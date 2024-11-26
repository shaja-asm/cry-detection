import os
import ctypes
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score
import scipy.fftpack
import sys


def load_tflite_c_library(lib_path):
    """Load the TensorFlow Lite C library."""
    lib = ctypes.cdll.LoadLibrary(lib_path)
    return lib


def define_c_api_functions(lib):
    """Define the required TensorFlow Lite C API functions."""
    # TfLiteModel
    lib.TfLiteModelCreate.restype = ctypes.c_void_p
    lib.TfLiteModelCreate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    lib.TfLiteModelDelete.restype = None
    lib.TfLiteModelDelete.argtypes = [ctypes.c_void_p]

    # TfLiteInterpreterOptions
    lib.TfLiteInterpreterOptionsCreate.restype = ctypes.c_void_p
    lib.TfLiteInterpreterOptionsCreate.argtypes = []

    lib.TfLiteInterpreterOptionsDelete.restype = None
    lib.TfLiteInterpreterOptionsDelete.argtypes = [ctypes.c_void_p]

    # TfLiteInterpreter
    lib.TfLiteInterpreterCreate.restype = ctypes.c_void_p
    lib.TfLiteInterpreterCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    lib.TfLiteInterpreterDelete.restype = None
    lib.TfLiteInterpreterDelete.argtypes = [ctypes.c_void_p]

    lib.TfLiteInterpreterAllocateTensors.restype = ctypes.c_int
    lib.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.c_void_p]

    lib.TfLiteInterpreterInvoke.restype = ctypes.c_int
    lib.TfLiteInterpreterInvoke.argtypes = [ctypes.c_void_p]

    lib.TfLiteInterpreterGetInputTensor.restype = ctypes.c_void_p
    lib.TfLiteInterpreterGetInputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int]

    lib.TfLiteInterpreterGetOutputTensor.restype = ctypes.c_void_p
    lib.TfLiteInterpreterGetOutputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # TfLiteTensor
    lib.TfLiteTensorCopyFromBuffer.restype = ctypes.c_int
    lib.TfLiteTensorCopyFromBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

    lib.TfLiteTensorCopyToBuffer.restype = ctypes.c_int
    lib.TfLiteTensorCopyToBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def compute_mfcc(y, sr, n_mfcc):
    """Compute MFCC features from an audio signal."""
    # Pre-emphasis filter
    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Frame parameters
    frame_size = 0.025  # 25 ms
    frame_stride = 0.010  # 10 ms
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    # Padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    # Framing
    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Windowing with Hamming window
    frames *= np.hamming(frame_length)

    # Fourier Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude spectrum
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power spectrum

    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, NFFT // 2 + 1))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]   # Left
        f_m = bin[m]             # Center
        f_m_plus = bin[m + 1]    # Right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    # Numerical stability
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # Convert to dB

    # MFCCs
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    # Mean normalization
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc


def preprocess_mfcc(mfcc, max_length, is_lstm):
    """Pad or truncate MFCC to max_length and add channel dimension if needed."""
    if mfcc.shape[0] < max_length:
        pad_width = ((0, max_length - mfcc.shape[0]), (0, 0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:max_length, :]

    if not is_lstm:
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension

    return mfcc


def preprocess_audio(file_path, num_mfcc, max_length, is_lstm):
    """Preprocess an audio file for inference."""
    y, sr = sf.read(file_path)
    y = y.astype(np.float32)
    y = y / np.max(np.abs(y))  # Normalize the audio signal
    mfcc = compute_mfcc(y, sr, n_mfcc=num_mfcc)
    mfcc = preprocess_mfcc(mfcc, max_length, is_lstm)
    return mfcc


def main():
    """Main function to perform inference using the TFLite C API."""
    # Set the audio folder and model type directly in the code
    audio_folder = '/home/garfield/CryCorpusFinal/Test_augmented'
    model_type = 'cnn'  # Choose 'cnn' or 'lstm'

    NUM_MFCC = 20
    MAX_LENGTH = 499
    TFLITE_MODEL_PATH = '/home/garfield/cry-detection/tflite_models/cnn_cry_detection_model_quant.tflite'
    TFLITE_C_LIBRARY_PATH = '/home/garfield/cry-detection/libtensorflowlite_c_2_14_1_amd64.so'

    # Load the TensorFlow Lite C library
    lib = load_tflite_c_library(TFLITE_C_LIBRARY_PATH)
    define_c_api_functions(lib)

    # Load the TFLite model
    with open(TFLITE_MODEL_PATH, 'rb') as f:
        tflite_model = f.read()

    model_data = ctypes.create_string_buffer(tflite_model)
    model_size = ctypes.c_size_t(len(tflite_model))

    # Create the model
    model = lib.TfLiteModelCreate(model_data, model_size)
    if not model:
        print('Failed to load model')
        sys.exit(1)

    # Create interpreter options
    options = lib.TfLiteInterpreterOptionsCreate()

    # Create the interpreter
    interpreter = lib.TfLiteInterpreterCreate(model, options)
    if not interpreter:
        print('Failed to create interpreter')
        sys.exit(1)

    # Allocate tensors
    status = lib.TfLiteInterpreterAllocateTensors(interpreter)
    if status != 0:
        print('Failed to allocate tensors')
        sys.exit(1)

    # Get input tensor
    input_index = 0  # Assuming the model has a single input
    input_tensor = lib.TfLiteInterpreterGetInputTensor(interpreter, input_index)
    if not input_tensor:
        print('Failed to get input tensor')
        sys.exit(1)

    # Prepare lists to store predictions and ground truth labels
    predictions = []
    ground_truths = []

    is_lstm_model = model_type == 'lstm'

    for file_name in os.listdir(audio_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(audio_folder, file_name)
            # Preprocess the audio file
            input_data = preprocess_audio(file_path, NUM_MFCC, MAX_LENGTH, is_lstm_model)

            # Adjust input shape for LSTM and CNN
            if is_lstm_model:
                input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            else:
                input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

            input_data_size = input_data.nbytes

            # Copy data into the input tensor
            status = lib.TfLiteTensorCopyFromBuffer(
                input_tensor,
                input_data.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(input_data_size),
            )
            if status != 0:
                print(f'Failed to copy data into input tensor for file {file_name}')
                continue

            # Invoke the interpreter
            status = lib.TfLiteInterpreterInvoke(interpreter)
            if status != 0:
                print(f'Failed to invoke interpreter for file {file_name}')
                continue

            # Get output tensor
            output_index = 0  # Assuming the model has a single output
            output_tensor = lib.TfLiteInterpreterGetOutputTensor(interpreter, output_index)
            if not output_tensor:
                print(f'Failed to get output tensor for file {file_name}')
                continue

            # Get output data
            output_data = np.zeros((1,), dtype=np.float32)
            output_data_size = output_data.nbytes

            # Copy output data from the tensor
            status = lib.TfLiteTensorCopyToBuffer(
                output_tensor,
                output_data.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(output_data_size),
            )
            if status != 0:
                print(f'Failed to copy output data for file {file_name}')
                continue

            # Get the prediction
            prediction = output_data[0]
            predictions.append(prediction)

            # Get the ground truth label from the file name
            if '_cry.wav' in file_name:
                ground_truth = 1
            elif '_notcry.wav' in file_name:
                ground_truth = 0
            else:
                print(f'Unknown label for file {file_name}, skipping')
                continue
            ground_truths.append(ground_truth)

            # Print the prediction
            prediction_label = 'Cry' if prediction > 0.5 else 'Not Cry'
            print(f"File: {file_name}, Prediction: {prediction_label}, Prediction fraction: {prediction}")

    # Compute metrics
    predictions_binary = [1 if p > 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(ground_truths, predictions_binary)
    f1 = f1_score(ground_truths, predictions_binary)

    # Compute false negative percentage
    false_negatives = sum(
        (np.array(ground_truths) == 1) & (np.array(predictions_binary) == 0)
    )
    total_positives = sum(np.array(ground_truths) == 1)
    false_negative_percentage = (
        (false_negatives / total_positives) * 100 if total_positives > 0 else 0
    )

    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Negative Percentage: {false_negative_percentage:.2f}%")

    # Clean up
    lib.TfLiteInterpreterDelete(interpreter)
    lib.TfLiteInterpreterOptionsDelete(options)
    lib.TfLiteModelDelete(model)


if __name__ == '__main__':
    main()
