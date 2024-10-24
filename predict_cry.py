import time
import ctypes
import numpy as np
import alsaaudio
from scipy.ndimage import zoom
from scipy.signal import stft


class PredictCry:
    MODEL_PATH = b"tflite_models/cry_detection_model_quant.tflite"
    SO_PATH = "/home/meow/repos/cry-detection/libtensorflowlite_c_2_14_1_amd64.so"

    def __init__(self, img_size=(64, 64), num_threads=2):
        self.lib = ctypes.cdll.LoadLibrary(self.SO_PATH)  # Load the shared library

        # Define types for the C API functions
        self.lib.TfLiteModelCreateFromFile.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.TfLiteModelCreateFromFile.argtypes = [ctypes.c_char_p]
        self.lib.TfLiteInterpreterCreate.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.TfLiteInterpreterCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.TfLiteInterpreterOptionsCreate.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.TfLiteInterpreterOptionsSetNumThreads.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.TfLiteInterpreterOptionsDelete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.TfLiteInterpreterDelete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.TfLiteModelDelete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.TfLiteInterpreterGetInputTensor.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.TfLiteInterpreterGetInputTensor.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.TfLiteInterpreterGetOutputTensor.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.TfLiteInterpreterGetOutputTensor.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.TfLiteInterpreterAllocateTensors.restype = ctypes.c_int
        self.lib.TfLiteInterpreterInvoke.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.TfLiteInterpreterInvoke.restype = ctypes.c_int
        self.lib.TfLiteTensorCopyFromBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t]
        self.lib.TfLiteTensorCopyFromBuffer.restype = ctypes.c_int
        self.lib.TfLiteTensorCopyToBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t]
        self.lib.TfLiteTensorCopyToBuffer.restype = ctypes.c_int

        # Create the model from file
        self.model = self.lib.TfLiteModelCreateFromFile(self.MODEL_PATH)
        if not self.model:
            raise RuntimeError("Failed to create TFLite model from file")

        # Create interpreter options
        self.options = self.lib.TfLiteInterpreterOptionsCreate()
        self.lib.TfLiteInterpreterOptionsSetNumThreads(self.options, num_threads)

        # Create the interpreter
        self.interpreter = self.lib.TfLiteInterpreterCreate(self.model, self.options)
        if not self.interpreter:
            raise RuntimeError("Failed to create TFLite interpreter")

        # Allocate tensors
        status = self.lib.TfLiteInterpreterAllocateTensors(self.interpreter)
        if status != 0:
            raise RuntimeError("Failed to allocate tensors")

        # Get input and output tensor pointers
        self.input_tensor = self.lib.TfLiteInterpreterGetInputTensor(self.interpreter, 0)
        if not self.input_tensor:
            raise RuntimeError("Failed to get input tensor")
        self.output_tensor = self.lib.TfLiteInterpreterGetOutputTensor(self.interpreter, 0)
        if not self.output_tensor:
            raise RuntimeError("Failed to get output tensor")

        # Set image size
        self.img_size = img_size

        self.sample_rate = 22050

    def preprocess_audio(self, audio):
        y = audio
        y = y / np.max(np.abs(y))  # Normalize

        # Compute STFT using scipy
        f, t, Zxx = stft(y, fs=self.sample_rate, nperseg=2048, noverlap=1536)
        D_magnitude = np.abs(Zxx)

        # Convert amplitude to dB scale
        D_dB = 20 * np.log10(D_magnitude + 1e-10)  # Add epsilon to avoid log(0)
        D_dB -= np.max(D_dB)  # Normalize to 0 dB max

        # Calculate zoom factors for resizing
        zoom_factors = [
            self.img_size[0] / D_dB.shape[0],
            self.img_size[1] / D_dB.shape[1]
        ]
        D_dB_resized = zoom(D_dB, zoom_factors, order=3)  # Cubic interpolation

        # Add channel dimension
        D_dB_resized = D_dB_resized[..., np.newaxis]

        return D_dB_resized

    def predict(self, audio):
        input_data = self.preprocess_audio(audio)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # Set the tensor to point to the input data to be inferred
        status = self.lib.TfLiteTensorCopyFromBuffer(
            self.input_tensor,
            ctypes.c_void_p(input_data.ctypes.data),
            ctypes.c_size_t(input_data.nbytes))
        if status != 0:
            raise RuntimeError("Failed to copy data to input tensor")

        # Run inference
        status = self.lib.TfLiteInterpreterInvoke(self.interpreter)
        if status != 0:
            raise RuntimeError("Failed to invoke interpreter")

        # Extract output data
        output_size = 1  # Assuming scalar output
        output_data = np.empty(output_size, dtype=np.float32)
        status = self.lib.TfLiteTensorCopyToBuffer(
            self.output_tensor,
            ctypes.c_void_p(output_data.ctypes.data),
            ctypes.c_size_t(output_data.nbytes))
        if status != 0:
            raise RuntimeError("Failed to copy data from output tensor")

        # print(f"Output: {output_data[0]}")
        return 'Cry' if output_data[0] > 0.5 else 'Not Cry'

    def __del__(self):
        # Clean up the interpreter, options, and model
        if hasattr(self, 'interpreter'):
            self.lib.TfLiteInterpreterDelete(self.interpreter)
        if hasattr(self, 'options'):
            self.lib.TfLiteInterpreterOptionsDelete(self.options)
        if hasattr(self, 'model'):
            self.lib.TfLiteModelDelete(self.model)


def main():
    # Below code is for testing on laptop only
    cry_predictor = PredictCry()
    duration = 5
    sample_rate = 22050
    print(f"Recording {duration} seconds")

    # Set up ALSA
    input_device = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
                                 channels=1, rate=sample_rate,
                                 format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=1024)

    # Capture audio data
    frames = []
    num_frames = int(sample_rate * duration / 1024)
    for _ in range(num_frames):
        length, data = input_device.read()
        if length > 0:
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            frames.append(audio_data)

    audio = np.concatenate(frames)
    t = time.time()
    prediction = cry_predictor.predict(audio)
    print(f"Prediction time: {time.time() - t:.2f} seconds")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
