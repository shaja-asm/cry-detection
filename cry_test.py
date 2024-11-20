import numpy as np
import soundfile as sf
import scipy.fftpack
import ctypes
import os

AUDIO_PATH = '/home/garfield/CryCorpusFinal'
CRY_FOLDER = os.path.join(AUDIO_PATH, 'cry_augmented')
NOTCRY_FOLDER = os.path.join(AUDIO_PATH, 'notcry_augmented')
NUM_MFCC = 20  #  Number of MFCC coefficients to extract
BATCH_SIZE = 32
EPOCHS = 50
MODEL = 'cnn'  # Choice: 'cnn' or 'lstm'

# Load the TFLite C library
lib = ctypes.cdll.LoadLibrary('/home/garfield/repos/cry-detection/libtensorflowlite_c_2_14_1_amd64.so')

# Define types for the C API functions
lib.TfLiteModelCreate.restype = ctypes.POINTER(ctypes.c_void_p)
lib.TfLiteInterpreterCreate.restype = ctypes.POINTER(ctypes.c_void_p)
lib.TfLiteInterpreterOptionsCreate.restype = ctypes.POINTER(ctypes.c_void_p)
lib.TfLiteInterpreterOptionsSetNumThreads.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
lib.TfLiteInterpreterOptionsDelete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.TfLiteInterpreterDelete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.TfLiteModelDelete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.TfLiteInterpreterGetInputTensor.restype = ctypes.POINTER(ctypes.c_void_p)
lib.TfLiteInterpreterGetOutputTensor.restype = ctypes.POINTER(ctypes.c_void_p)
lib.TfLiteTensorCopyFromBuffer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t]
lib.TfLiteTensorCopyToBuffer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t]
lib.TfLiteInterpreterInvoke.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.TfLiteInterpreterGetInputTensor.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
lib.TfLiteInterpreterGetOutputTensor.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

model_path = b"tflite_models/cry_detection_model_quant.tflite"
with open(model_path, 'rb') as f:
    model_data = f.read()

model = lib.TfLiteModelCreate(ctypes.c_char_p(model_data), ctypes.c_size_t(len(model_data)))

# Create interpreter options and set number of threads
options = lib.TfLiteInterpreterOptionsCreate()
lib.TfLiteInterpreterOptionsSetNumThreads(options, 2)

# Create the interpreter with the custom options
interpreter = lib.TfLiteInterpreterCreate(model, options)

# Allocate tensors
status = lib.TfLiteInterpreterAllocateTensors(interpreter)

# Get input and output tensor pointers
input_tensor = lib.TfLiteInterpreterGetInputTensor(interpreter, 0)
output_tensor = lib.TfLiteInterpreterGetOutputTensor(interpreter, 0)

def compute_mfcc(y, sr, n_mfcc=NUM_MFCC):
    # Pre-emphasis filter
    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
    # Frame parameters
    frame_size = 0.025  # 25 ms
    frame_stride = 0.010  # 10 ms
    frame_length, frame_step = int(round(frame_size * sr)), int(round(frame_stride * sr))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    # Padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    
    # Framing
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Windowing with Hamming window
    frames *= np.hamming(frame_length)
    
    # Fourier Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    
    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((nfilt, NFFT // 2 + 1))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    filter_banks = np.dot(pow_frames, fbank.T)
    # Numerical stability
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    # MFCCs
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    # Mean normalization
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    
    return mfcc

def preprocess_audio(file_path, num_mfcc=NUM_MFCC, max_length=99, is_lstm=False):
    y, sr = sf.read(file_path)
    y = y.astype(np.float32)
    y = y / np.max(np.abs(y))
    mfcc = compute_mfcc(y, sr, n_mfcc=num_mfcc)
    
    # Pad or truncate to fixed length
    if mfcc.shape[0] < max_length:
        pad_width = ((0, max_length - mfcc.shape[0]), (0, 0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:max_length, :]
    
    if not is_lstm:
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension for CNN
    return mfcc

def predict(file_path, num_mfcc=NUM_MFCC, max_length=99, is_lstm=False):
    input_data = preprocess_audio(file_path, num_mfcc, max_length, is_lstm)
    
    # Adjust input shape for LSTM and CNN
    if is_lstm:
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Shape: (1, time_steps, num_mfcc)
    else:
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Shape: (1, time_steps, num_mfcc, 1)
    
    # Set the tensor to point to the input data to be inferred
    lib.TfLiteTensorCopyFromBuffer(
        input_tensor,
        input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(input_data.nbytes)
    )
    
    # Run inference
    lib.TfLiteInterpreterInvoke(interpreter)
    
    # Extract output data
    output_size = 1
    output_data = np.empty(output_size, dtype=np.float32)
    lib.TfLiteTensorCopyToBuffer(
        output_tensor,
        output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(output_data.nbytes)
    )
    
    return output_data

def process_folder(folder_path, num_mfcc=NUM_MFCC, max_length=99, is_lstm=False):
    correct_predictions = 0
    total_files = 0
    results = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            prediction = predict(file_path, num_mfcc, max_length, is_lstm)
            prediction_label = 'Cry' if prediction > 0.5 else 'Not Cry'
            results.append((file_name, prediction_label))
            ground_truth = 'Cry' if '_cry.wav' in file_name else 'Not Cry'

            if prediction_label == ground_truth:
                correct_predictions += 1
                if prediction_label == 'Cry':
                    true_positives += 1
            else:
                if prediction_label == 'Cry':
                    false_positives += 1
                elif prediction_label == 'Not Cry' and ground_truth == 'Cry':
                    false_negatives += 1

            total_files += 1

    # Calculate metrics
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_negative_percentage = (false_negatives / (false_negatives + true_positives)) * 100 if (false_negatives + true_positives) > 0 else 0

    return results, accuracy, f1, false_negative_percentage

# Specify the folder containing test audio files
folder_path = '{0}/Test_augmented'.format(AUDIO_PATH)
if MODEL == 'lstm':
    is_lstm_model = True
else:
    is_lstm_model = False

# Set max_length based on training parameters (adjust as necessary)
max_length = 499  # This should match the max_length used during model training

predictions, accuracy, f1_score, false_negative_percentage = process_folder(
    folder_path,
    num_mfcc=NUM_MFCC,
    max_length=max_length,
    is_lstm=is_lstm_model
)

for file_name, prediction_label in predictions:
    print(f"File: {file_name}, Prediction: {prediction_label}")

print(f"Prediction Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1_score:.2f}")
print(f"False Negative Percentage: {false_negative_percentage:.2f}%")

# Clean up resources
lib.TfLiteInterpreterDelete(interpreter)
lib.TfLiteInterpreterOptionsDelete(options)
lib.TfLiteModelDelete(model)

print("All operations completed successfully.")
