import os
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, LSTM)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import datetime
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard, EarlyStopping)
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
import scipy.fftpack
import pathlib


# Constants
AUDIO_PATH = '/mnt/d/Datasets/CryCorpusNew/'
CRY_FOLDER = os.path.join(AUDIO_PATH, 'cry_augmented')
NOTCRY_FOLDER = os.path.join(AUDIO_PATH, 'notcry_augmented')
NUM_MFCC = 20  # Number of MFCC coefficients to extract
BATCH_SIZE = 32
EPOCHS = 50
MODEL_TYPE = 'cnn'  # Choices: 'cnn' or 'lstm'
MAX_LENGTH = 499  # Fixed maximum sequence length for padding/truncation


# Set GPU configuration (if applicable)
gpus = tf.config.experimental.list_physical_devices('GPU') 
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def load_audio_files(folder):
    """Load audio file paths from a folder."""
    files = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            files.append(os.path.join(folder, filename))
    return files


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


def save_mfcc_to_disk(mfcc, save_path):
    """Save MFCC array to disk as a .npy file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, mfcc)


def preprocess_mfcc(mfcc, max_length, is_lstm=False):
    """Pad or truncate MFCC to max_length and add channel dimension if needed."""
    # Pad or truncate the MFCC sequence to max_length
    if mfcc.shape[0] < max_length:
        pad_width = ((0, max_length - mfcc.shape[0]), (0, 0))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:max_length, :]

    if not is_lstm:
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension

    return mfcc


def process_and_save_mfcc(file_paths, label_prefix, save_dir, n_mfcc=NUM_MFCC):
    """Process audio files to compute and save MFCCs."""
    data_paths = []
    labels = []
    for idx, file in enumerate(file_paths):
        y, sr = sf.read(file)
        y = y.astype(np.float32)
        # Normalize the audio signal
        y = y / np.max(np.abs(y))
        mfcc = compute_mfcc(y, sr, n_mfcc=n_mfcc)
        save_path = os.path.join(save_dir, f'{label_prefix}_{idx}.npy')
        save_mfcc_to_disk(mfcc, save_path)
        data_paths.append(save_path)
        labels.append(1 if label_prefix == 'cry' else 0)
        del y, mfcc  # Free up memory
    return data_paths, labels


def load_and_preprocess_mfcc(file_path, max_length, is_lstm=False):
    """Load MFCC from .npy file and preprocess it."""
    mfcc = np.load(file_path)
    mfcc = preprocess_mfcc(mfcc, max_length, is_lstm)
    return mfcc


class OnTheFlyDataGenerator(tf.keras.utils.Sequence):
    """Data generator for loading and processing data on the fly."""
    def __init__(
        self, file_paths, labels, batch_size, num_mfcc, max_length, shuffle=True, augment=False, is_lstm=False, class_weights=None
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_mfcc = num_mfcc
        self.max_length = max_length
        self.shuffle = shuffle
        self.augment = augment
        self.is_lstm = is_lstm
        self.class_weights = class_weights  # Add class_weights
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : min((index + 1) * self.batch_size, len(self.file_paths))
        ]
        batch_file_paths = [self.file_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        X, y, sample_weights = self.__data_generation(batch_file_paths, batch_labels)
        return X, y, sample_weights  # Return sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_file_paths, batch_labels):
        X = []
        y = []
        sample_weights = []  # Initialize sample_weights

        for i, file_path in enumerate(batch_file_paths):
            mfcc = load_and_preprocess_mfcc(file_path, self.max_length, self.is_lstm)

            if self.augment:
                mfcc = self._augment_mfcc(mfcc)

            X.append(mfcc)
            y_label = batch_labels[i]
            y.append(y_label)

            # Assign sample weight based on the class
            if self.class_weights:
                sample_weight = self.class_weights[y_label]
                sample_weights.append(sample_weight)
            else:
                sample_weights.append(1.0)  # Default weight

        X = np.array(X)
        y = np.array(y)
        sample_weights = np.array(sample_weights)
        return X, y, sample_weights

    def _augment_mfcc(self, mfcc):
        noise_factor = 0.005
        noise = np.random.randn(*mfcc.shape)
        mfcc += noise_factor * noise

        time_shift = np.roll(mfcc, shift=np.random.randint(-10, 10), axis=0)

        mfcc = np.clip(mfcc, -1, 1)
        return time_shift


def build_cnn_model(hp):
    """Build a CNN model with hyperparameters."""
    l2_regularizer = tf.keras.regularizers.l2(
        hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
    )
    model = Sequential()

    model.add(
        Conv2D(
            filters=hp.Int('filters_1', min_value=32, max_value=96, step=32),
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            input_shape=(MAX_LENGTH, NUM_MFCC, 1),
            kernel_regularizer=l2_regularizer,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.3, max_value=0.6, step=0.1)))

    model.add(
        Conv2D(
            filters=hp.Int('filters_2', min_value=64, max_value=128, step=64),
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=l2_regularizer,
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1)))

    model.add(Flatten())
    model.add(
        Dense(
            units=128,
            activation='relu',
            kernel_regularizer=l2_regularizer,
        )
    )
    model.add(Dropout(rate=hp.Float('dropout_fc', min_value=0.3, max_value=0.6, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(
        learning_rate=hp.Float(
            'learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG'
        )
    )
    # optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)  # Gradient clipping
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


def build_lstm_model(hp):
    """Build an LSTM model with hyperparameters."""
    model = Sequential()

    # LSTM layers with regularization
    model.add(
        LSTM(
            units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=64),
            input_shape=(MAX_LENGTH, NUM_MFCC),
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
            ),
        )
    )
    model.add(
        Dropout(
            rate=hp.Float('dropout_lstm_1', min_value=0.3, max_value=0.6, step=0.1)
        )
    )

    model.add(
        LSTM(
            units=hp.Int('lstm_units_2', min_value=64, max_value=256, step=64),
            return_sequences=True,
        )
    )
    model.add(
        Dropout(
            rate=hp.Float('dropout_lstm_2', min_value=0.3, max_value=0.6, step=0.1)
        )
    )

    model.add(
        LSTM(
            units=hp.Int('lstm_units_3', min_value=64, max_value=256, step=64),
            return_sequences=False,
        )
    )
    model.add(
        Dropout(
            rate=hp.Float('dropout_lstm_3', min_value=0.3, max_value=0.6, step=0.1)
        )
    )

    # Fully connected layers
    model.add(
        Dense(
            units=hp.Int('dense_units', min_value=64, max_value=128, step=64),
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
            ),
        )
    )
    model.add(
        Dropout(rate=hp.Float('dropout_fc', min_value=0.3, max_value=0.6, step=0.1))
    )
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(
        learning_rate=hp.Float(
            'learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG'
        )
    )
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


def train_model_with_scheduler(model, model_name, train_generator, val_generator):
    """Train the model with learning rate scheduler and checkpoints."""
    log_dir = (
        "logs/fit/"
        + model_name
        + "_"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_callback = ModelCheckpoint(
        filepath=f'{model_name}_model',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
    )

    lr_callback = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=25, min_lr=1e-5
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=45, restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[
            tensorboard_callback,
            checkpoint_callback,
            lr_callback,
            early_stopping_callback,
        ],
    )
    return history


def build_and_train_cnn_model(train_generator, val_generator):
    """Build and train the CNN model with hyperparameter tuning."""
    tuner = RandomSearch(
        build_cnn_model,
        objective='val_accuracy',
        max_trials=30,
        executions_per_trial=1,
        directory='hyperparam_tuning_new',
        project_name='cnn_tuning_new',
    )
    
    tuner.search(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15)],
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = train_model_with_scheduler(model, 'cnn_best', train_generator, val_generator)
    return model


def build_and_train_lstm_model(train_generator, val_generator):
    """Build and train the LSTM model with hyperparameter tuning."""
    tuner = RandomSearch(
        build_lstm_model,
        objective='val_accuracy',
        max_trials=30,
        executions_per_trial=1,
        directory='hyperparam_tuning',
        project_name='lstm_tuning',
    )
    
    tuner.search(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15)],
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = train_model_with_scheduler(model, 'lstm_best', train_generator, val_generator)
    return model


def load_validation_data(file_paths, labels, num_mfcc, max_length, is_lstm):
    """Load and preprocess validation data."""
    X = []
    y = []
    for i, file_path in enumerate(file_paths):
        mfcc = load_and_preprocess_mfcc(file_path, max_length, is_lstm)
        X.append(mfcc)
        y.append(labels[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def evaluate_model(model, X_val, y_val, is_lstm=False):
    """Evaluate the model on validation data and save it."""
    # Load validation data
    X_val_data, y_true = load_validation_data(X_val, y_val, NUM_MFCC, MAX_LENGTH, is_lstm)
    # Make predictions
    y_pred = model.predict(X_val_data)
    y_pred = (y_pred > 0.5).astype(int)
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # Print the metrics
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation F1 Score: {f1:.4f}')
    # Save the model
    model_name = 'cnn' if not is_lstm else 'lstm'
    model.save(f'{model_name}_cry_detection_model.keras')
    print(f'Model saved as {model_name}_cry_detection_model.keras')


def convert_and_save_tflite_model(model, model_type, sample_audio_files, num_mfcc, max_length, is_lstm=False):
    """
    Convert the Keras model to a fully quantized TFLite format and save it.

    Args:
        model (tf.keras.Model): Trained Keras model to convert.
        model_type (str): Type of model ('cnn' or 'lstm').
        sample_audio_files (list): List of file paths for representative data.
        num_mfcc (int): Number of MFCC features used in the model.
        max_length (int): Maximum length of input sequences.
        is_lstm (bool): Whether the model is an LSTM model.
    """
    # Create directory for TFLite models
    tflite_models_dir = pathlib.Path("tflite_models")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Define representative dataset generator
    def representative_dataset():
        for file_path in sample_audio_files:
            try:
                # Load precomputed MFCCs from .npy file
                mfcc = np.load(file_path)

                # Preprocess to match model input
                input_data = preprocess_mfcc(mfcc, max_length=max_length, is_lstm=is_lstm)

                # Add batch dimension
                input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
                print(f"Representative input shape: {input_data.shape}")
                yield [input_data]
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    # Convert the model to fully quantized TFLite
    try:
        print("Converting to fully quantized TFLite model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable optimizations
        converter.representative_dataset = representative_dataset  # Use representative dataset for quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Restrict to int8 ops
        converter.inference_input_type = tf.int8  # Quantize input tensors to int8
        converter.inference_output_type = tf.int8  # Quantize output tensors to int8

        tflite_quant_model = converter.convert()
        tflite_model_quant_file = tflite_models_dir / f"{model_type}_cry_detection_model_quant_shifted.tflite"
        tflite_model_quant_file.write_bytes(tflite_quant_model)
        print(f"Fully quantized TFLite model saved to {tflite_model_quant_file}")
    except Exception as e:
        print(f"Failed to convert quantized TFLite model: {e}")

def preprocess_audio(file_path, num_mfcc, max_length, is_lstm):
    """Preprocess an audio file for inference."""
    y, sr = sf.read(file_path)
    y = y.astype(np.float32)
    # Normalize the audio signal
    y = y / np.max(np.abs(y))
    # Compute MFCCs
    mfcc = compute_mfcc(y, sr, n_mfcc=num_mfcc)
    # Preprocess MFCC (pad/truncate, add channel dimension)
    mfcc = preprocess_mfcc(mfcc, max_length, is_lstm)
    return mfcc


def predict_tflite(interpreter, input_details, output_details, file_path, num_mfcc, max_length, is_lstm):
    """Make a prediction using the TFLite interpreter."""
    input_data = preprocess_audio(file_path, num_mfcc, max_length, is_lstm)
    
    # Adjust input shape for LSTM and CNN
    if is_lstm:
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Shape: (1, time_steps, num_mfcc)
    else:
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Shape: (1, time_steps, num_mfcc, 1)
    
    # Get input quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    
    # Quantize the input data if the model expects INT8 input
    if input_details[0]['dtype'] == np.int8:
        input_data = input_data / input_scale + input_zero_point
        input_data = np.round(input_data).astype(np.int8)
    else:
        input_data = input_data.astype(input_details[0]['dtype'])
    
    # Set the tensor to point to the input data for inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output if necessary
    output_scale, output_zero_point = output_details[0]['quantization']
    if output_details[0]['dtype'] == np.int8:
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
    
    return output_data


def process_folder_tflite(interpreter, input_details, output_details, folder_path, num_mfcc, max_length, is_lstm):
    """Process a folder of audio files using the TFLite model."""
    correct_predictions = 0
    total_files = 0
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            prediction = predict_tflite(interpreter, input_details, output_details, file_path, num_mfcc, max_length, is_lstm)
            
            # Since the output might be quantized, ensure it's converted to float32
            if isinstance(prediction, np.ndarray) and prediction.dtype != np.float32:
                prediction = prediction.astype(np.float32)
            
            # Assuming binary classification with sigmoid activation
            prediction_prob = prediction[0][0]
            prediction_label = 'Cry' if prediction_prob > 0.5 else 'Not Cry'
            results.append((file_name, prediction_label))
            ground_truth = 'Cry' if '_cry.wav' in file_name else 'Not Cry'

            if prediction_label == ground_truth:
                correct_predictions += 1

            total_files += 1

    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0

    return results, accuracy


def tflite_inference(model_type):
    """Perform inference using the TFLite model."""
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=f"tflite_models/{model_type}_cry_detection_model_quant_shifted.tflite")
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Determine whether the model is LSTM
    is_lstm_model = True if model_type == 'lstm' else False
    
    # Define folder path for test data
    folder_path = os.path.join(AUDIO_PATH, 'test_augmented')
    
    predictions, accuracy = process_folder_tflite(interpreter, input_details, output_details, folder_path, NUM_MFCC, MAX_LENGTH, is_lstm_model)
    
    for file_name, prediction_label in predictions:
        print(f"File: {file_name}, Prediction: {prediction_label}")
    
    print(f"Prediction Accuracy: {accuracy:.2f}%")



def main():
    """Main function to execute the training and inference pipeline."""
    # Load audio file paths
    cry_files = load_audio_files(CRY_FOLDER)
    notcry_files = load_audio_files(NOTCRY_FOLDER)
    
    # Prepare directories for saving MFCCs
    mfcc_save_dir = os.path.join(AUDIO_PATH, 'mfccs')
    os.makedirs(mfcc_save_dir, exist_ok=True)
    
    # Process and save MFCCs for cry and notcry files
    cry_data_paths, cry_labels = process_and_save_mfcc(cry_files, 'cry', mfcc_save_dir, n_mfcc=NUM_MFCC)
    notcry_data_paths, notcry_labels = process_and_save_mfcc(notcry_files, 'notcry', mfcc_save_dir, n_mfcc=NUM_MFCC)
    
    # Combine data paths and labels
    data_paths = np.array(cry_data_paths + notcry_data_paths)
    labels = np.array(cry_labels + notcry_labels)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        data_paths, labels, test_size=0.2, random_state=42
    )
    
    # Compute class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(zip(np.unique(y_train), class_weights_array))
    
    # Initialize data generators with class weights
    train_generator = OnTheFlyDataGenerator(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        num_mfcc=NUM_MFCC,
        max_length=MAX_LENGTH,
        shuffle=True,
        augment=True,
        is_lstm=(MODEL_TYPE == 'lstm'),
        class_weights=class_weights_dict
    )
    
    val_generator = OnTheFlyDataGenerator(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        num_mfcc=NUM_MFCC,
        max_length=MAX_LENGTH,
        shuffle=False,
        augment=False,
        is_lstm=(MODEL_TYPE == 'lstm'),
        class_weights=None  # No need for sample weights in validation
    )
    
    # Build and train model
    if MODEL_TYPE == 'cnn':
        # Build and train CNN model
        model = build_and_train_cnn_model(train_generator, val_generator)
    elif MODEL_TYPE == 'lstm':
        # Build and train LSTM model
        model = build_and_train_lstm_model(train_generator, val_generator)
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose 'cnn' or 'lstm'.")
    
    # Evaluate model on validation data
    evaluate_model(model, X_val, y_val, is_lstm=(MODEL_TYPE == 'lstm'))
    
    # Define sample files for quantization
    # Use a subset of the training files for representative data
    sample_audio_files = X_train[:20]
    
    # Convert model to TFLite and save
    convert_and_save_tflite_model(
        model=model,
        model_type=MODEL_TYPE,
        sample_audio_files=sample_audio_files,
        num_mfcc=NUM_MFCC,
        max_length=MAX_LENGTH,
        is_lstm=(MODEL_TYPE == 'lstm')
    )
    
    # Perform inference using TFLite model
    tflite_inference(MODEL_TYPE)


if __name__ == '__main__':
    main()
