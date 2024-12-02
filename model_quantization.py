import os
import numpy as np
import soundfile as sf
import tensorflow as tf
import scipy.fftpack
import pathlib
from sklearn.metrics import accuracy_score, f1_score

# Constants
NUM_MFCC = 20  # Number of MFCC coefficients to extract
MAX_LENGTH = 499  # Fixed maximum sequence length for padding/truncation
MODEL_TYPE = 'cnn'  # Model type
SAMPLE_RATE = 22050  # Sample rate used during training

def compute_mfcc(y, sr, n_mfcc=NUM_MFCC):
    """Compute MFCC features from an audio signal."""
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
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude spectrum
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))  # Power spectrum
    
    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Mel to Hz
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

def preprocess_mfcc(mfcc, max_length):
    """Pad or truncate MFCC to max_length and add channel dimension."""
    # Pad or truncate the MFCC sequence to max_length
    if mfcc.shape[0] < max_length:
        pad_width = ((0, max_length - mfcc.shape[0]), (0, 0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:max_length, :]

    mfcc = mfcc[..., np.newaxis]  # Add channel dimension

    return mfcc

def preprocess_audio(file_path, num_mfcc, max_length):
    """Preprocess an audio file for inference."""
    y, sr = sf.read(file_path)
    y = y.astype(np.float32)
    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    # Normalize the audio signal
    y = y / np.max(np.abs(y))
    # Compute MFCCs
    mfcc = compute_mfcc(y, sr, n_mfcc=num_mfcc)
    # Preprocess MFCC (pad/truncate, add channel dimension)
    mfcc = preprocess_mfcc(mfcc, max_length)
    return mfcc

def representative_dataset_gen(sample_audio_files):
    """Generator function for the representative dataset."""
    for file_path in sample_audio_files:
        try:
            mfcc = np.load(file_path)
            # Preprocess to match model input
            input_data = preprocess_mfcc(mfcc, max_length=MAX_LENGTH)
            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            yield [input_data]
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

def convert_and_save_tflite_model(model, sample_audio_files):
    """Convert the Keras model to a fully quantized TFLite model and save it."""
    # Create directory for TFLite models
    tflite_models_dir = pathlib.Path("tflite_models")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Define representative dataset generator
    def representative_dataset():
        for file_path in sample_audio_files:
            try:
                mfcc = np.load(file_path)
                # Preprocess to match model input
                input_data = preprocess_mfcc(mfcc, max_length=MAX_LENGTH)
                # Add batch dimension
                input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
                yield [input_data]
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    try:
        print("Converting to fully quantized TFLite model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_quantizer = True
        tflite_quant_model = converter.convert()
        
        # Save the quantized model to tflite_models_dir
        tflite_model_quant_file = tflite_models_dir / f"{MODEL_TYPE}_cry_detection_model_quant_alternative.tflite"
        tflite_model_quant_file.write_bytes(tflite_quant_model)
        print(f"Fully quantized TFLite model saved to {tflite_model_quant_file}")
    except Exception as e:
        print(f"Failed to convert quantized TFLite model: {e}")


def load_and_run_tflite_model_on_folder(tflite_model_path, audio_folder_path):
    """Load the TFLite model, perform inference, and calculate accuracy and F1-score."""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input and output quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    # Initialize lists to store predictions and ground truth
    predictions = []
    ground_truth = []
    
    # Iterate through all .wav files in the folder
    for file_name in os.listdir(audio_folder_path):
        if file_name.endswith('.wav'):  # Process only .wav files
            audio_file_path = os.path.join(audio_folder_path, file_name)
            
            # Infer ground truth from file name
            if '_cry.wav' in file_name:
                ground_truth.append(1)  # Label for CRY
            elif '_notcry.wav' in file_name:
                ground_truth.append(0)  # Label for NOT CRY
            else:
                print(f"Skipping file {file_name} (no label in filename).")
                continue
            
            # Preprocess the audio file
            input_data = preprocess_audio(audio_file_path, NUM_MFCC, MAX_LENGTH)
            input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
            
            # Quantize the input data
            quantized_input_data = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
            
            # Set the tensor to point to the input data
            interpreter.set_tensor(input_details[0]['index'], quantized_input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get the output data
            quantized_output_data = interpreter.get_tensor(output_details[0]['index'])
            # Dequantize the output data
            output_data = (quantized_output_data - output_zero_point) * output_scale
            
            # Apply sigmoid activation since the model uses linear activation
            prediction_score = 1 / (1 + np.exp(-output_data[0][0]))
            prediction = 1 if prediction_score > 0.5 else 0  # Threshold at 0.5
            
            # Store the prediction
            predictions.append(prediction)
            
            # Print results for the current file
            print(f"File: {file_name}")
            print(f"Prediction score: {prediction_score}")
            print(f"Predicted Label: {'CRY' if prediction == 1 else 'NOT CRY'}")
            print(f"Ground Truth: {'CRY' if ground_truth[-1] == 1 else 'NOT CRY'}")
            print("-" * 50)
    
    # Calculate accuracy and F1-score
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("--------------------------")

    return accuracy, f1

def main():
    # Load the Keras model
    model = tf.keras.models.load_model('cnn_cry_detection_model.keras')
    print("Keras model loaded.")
    
    # Path to the folder containing precomputed MFCC .npy files
    mfcc_folder_path = '/mnt/d/Datasets/CryCorpusFinal/mfccs'
    
    # Get a list of .npy files in the folder
    all_mfcc_files = [os.path.join(mfcc_folder_path, f) for f in os.listdir(mfcc_folder_path) if f.endswith('.npy')]
    
    # Select 20 files for the representative dataset
    sample_mfcc_files = all_mfcc_files[:20]  # You can randomize or select specific files as needed
    
    # Convert and save the TFLite model
    convert_and_save_tflite_model(model, sample_mfcc_files)
    
    # Path to the quantized TFLite model
    tflite_model_path = pathlib.Path("tflite_models") / f"{MODEL_TYPE}_cry_detection_model_quant_alternative.tflite"
    
    # Path to the folder containing audio files for inference
    audio_folder_path = '/mnt/d/Datasets/CryCorpusFinal/Test_augmented'
    
    # Run inference using the TFLite model on all files in the folder
    accuracy, f1 = load_and_run_tflite_model_on_folder(tflite_model_path, audio_folder_path)
    
    print(f"\nFinal Results: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

if __name__ == '__main__':
    main()
