import os
import numpy as np
import soundfile as sf
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import zoom
from kerastuner.tuners import RandomSearch
import scipy.fftpack
import numpy as np
import soundfile as sf

# Set GPU configuration (if applicable)
gpus = tf.config.experimental.list_physical_devices('GPU') 
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


AUDIO_PATH = '/home/garfield/CryCorpusFinal'
CRY_FOLDER = os.path.join(AUDIO_PATH, 'cry_augmented')
NOTCRY_FOLDER = os.path.join(AUDIO_PATH, 'notcry_augmented')
NUM_MFCC = 20  #  Number of MFCC coefficients to extract
BATCH_SIZE = 32
EPOCHS = 50
MODEL = 'cnn'  # Choice: 'cnn' or 'lstm'




def load_audio_files(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            files.append(os.path.join(folder, filename))
    return files

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
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, mfcc)


cry_files = load_audio_files(CRY_FOLDER)
notcry_files = load_audio_files(NOTCRY_FOLDER)

data = []
labels = []

for idx, file in enumerate(cry_files):
    y, sr = sf.read(file)
    y = y.astype(np.float32)
    # Normalize the audio signal
    y = y / np.max(np.abs(y))
    mfcc = compute_mfcc(y, sr, n_mfcc=NUM_MFCC)
    save_path = os.path.join(f'{AUDIO_PATH}/mfccs', f'cry_{idx}.npy')
    save_mfcc_to_disk(mfcc, save_path)
    data.append(save_path)
    labels.append(1)
    del y, mfcc  # Free up memory

for idx, file in enumerate(notcry_files):
    y, sr = sf.read(file)
    y = y.astype(np.float32)
    # Normalize the audio signal
    y = y / np.max(np.abs(y))
    mfcc = compute_mfcc(y, sr, n_mfcc=NUM_MFCC)
    save_path = os.path.join(f'{AUDIO_PATH}/mfccs', f'notcry_{idx}.npy')
    save_mfcc_to_disk(mfcc, save_path)
    data.append(save_path)
    labels.append(0)
    del y, mfcc  # Free up memory

data = np.array(data)
labels = np.array(labels)


# Split data
X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Set the fixed maximum sequence length
MAX_LENGTH = 499

class OnTheFlyDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, file_paths, labels, batch_size, num_mfcc, max_length, shuffle=True, augment=False, is_lstm=False
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_mfcc = num_mfcc
        self.max_length = max_length
        self.shuffle = shuffle
        self.augment = augment
        self.is_lstm = is_lstm
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()
        self.cache = {}

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : min((index + 1) * self.batch_size, len(self.file_paths))
        ]
        batch_file_paths = [self.file_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        X, y = self.__data_generation(batch_file_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_file_paths, batch_labels):
        X = []
        y = []

        for i, file_path in enumerate(batch_file_paths):
            mfcc = np.load(file_path)

            if self.augment:
                mfcc = self._augment_mfcc(mfcc)

            if mfcc.shape[0] < self.max_length:
                pad_width = ((0, self.max_length - mfcc.shape[0]), (0, 0))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            else:
                mfcc = mfcc[: self.max_length, :]

            X.append(mfcc)
            y.append(batch_labels[i])
            del mfcc

        X = np.array(X)
        y = np.array(y)

        if self.is_lstm:
            return X, y
        else:
            X = X[..., np.newaxis]
            return X, y

    def _augment_mfcc(self, mfcc):
        noise_factor = 0.005
        noise = np.random.randn(*mfcc.shape)
        mfcc += noise_factor * noise

        time_shift = np.roll(mfcc, shift=np.random.randint(-10, 10), axis=0)

        mfcc = np.clip(mfcc, -1, 1)
        return time_shift


# CNN Model with Improved Regularization and Gradient Clipping
def build_cnn_model(hp):
    l2_regularizer = tf.keras.regularizers.l2(
        hp.Float('l2', min_value=1e-4, max_value=1e-2, sampling='LOG')
    )
    model = Sequential()

    # Reduced model complexity
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
            units=hp.Int('dense_units', min_value=64, max_value=128, step=64),
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
    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)  # Gradient clipping
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


# LSTM Model with Improvements
def build_lstm_model(hp):
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

train_generator = OnTheFlyDataGenerator(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    num_mfcc=NUM_MFCC,
    max_length=MAX_LENGTH,
    shuffle=True,
    augment=True,
    is_lstm=(MODEL == 'lstm'),
)

val_generator = OnTheFlyDataGenerator(
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    num_mfcc=NUM_MFCC,
    max_length=MAX_LENGTH,
    shuffle=False,
    augment=False,
    is_lstm=(MODEL == 'lstm'),
)

# Training Function
def train_model_with_scheduler(model, model_name):
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

    history = model.fit(train_generator,
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


# Hyperparameter Tuning and Training
if MODEL == 'cnn':
    tuner = RandomSearch(
        build_cnn_model,
        objective='val_accuracy',
        max_trials=30,
        executions_per_trial=1,
        directory='hyperparam_tuning',
        project_name='cnn_tuning',
    )

    tuner.search(
        train_generator,
        epochs=25,
        validation_data=val_generator,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15)],
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    history = train_model_with_scheduler(model, 'cnn_best')

elif MODEL == 'lstm':
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
    history = train_model_with_scheduler(model, 'lstm_best')


    def load_validation_data(file_paths, labels, num_mfcc, max_length, is_lstm):
        X = []
        y = []
        for i, file_path in enumerate(file_paths):
            mfcc = np.load(file_path)
            # Pad or truncate the MFCC sequence to max_length
            if mfcc.shape[0] < max_length:
                pad_width = ((0, max_length - mfcc.shape[0]), (0, 0))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            else:
                mfcc = mfcc[:max_length, :]

            if not is_lstm:
                mfcc = mfcc[..., np.newaxis]  # Add channel dimension

            X.append(mfcc)
            y.append(labels[i])

        X = np.array(X)
        y = np.array(y)
        return X, y

def evaluate_and_save_model(model, model_name, X_val_data, y_true, is_lstm=False):
    # Make predictions
    y_pred = model.predict(X_val_data)
    y_pred = (y_pred > 0.5).astype(int)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)  # Use a distinct name for the variable
    f1 = f1_score(y_true, y_pred)  # Ensure no shadowing of the function name

    # Print the metrics
    print(f'Accuracy for {model_name}: {accuracy:.4f}')
    print(f'F1 Score for {model_name}: {f1:.4f}')

    # Save the model
    model.save(f'{model_name}_cry_detection_model.keras')
    print(f'{model_name} model saved as {model_name}_cry_detection_model.keras')


# Model selection, training, and evaluation
if MODEL == 'cnn':
    # Load validation data
    X_val_data, y_true = load_validation_data(X_val, y_val, NUM_MFCC, MAX_LENGTH, is_lstm=False)
    # Evaluate and save the CNN model
    evaluate_and_save_model(model, 'cnn', X_val_data, y_true, is_lstm=False)

elif MODEL == 'lstm':
    # Load validation data
    X_val_data, y_true = load_validation_data(X_val, y_val, NUM_MFCC, MAX_LENGTH, is_lstm=True)
    # Evaluate and save the LSTM model
    evaluate_and_save_model(model, 'lstm', X_val_data, y_true, is_lstm=True)

import pathlib

# Create directory for TFLite models
tflite_models_dir = pathlib.Path("tflite_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow Select TF Ops for both CNN and LSTM models if necessary
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# For LSTM models, disable experimental lowering of tensor list ops
if MODEL == 'lstm':
    converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the model
tflite_model_file = tflite_models_dir / "cry_detection_model.tflite"
tflite_model_file.write_bytes(tflite_model)

# Apply optimizations and convert again
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir / "cry_detection_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)

print("TFLite conversion successful!")



# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="tflite_models/cry_detection_model_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(file_path, num_mfcc=NUM_MFCC, is_lstm=False):
    y, sr = sf.read(file_path)
    y = y.astype(np.float32)
    # Normalize the audio signal
    y = y / np.max(np.abs(y))
    # Compute MFCCs
    mfcc = compute_mfcc(y, sr, n_mfcc=num_mfcc)
    # Add channel dimension for CNN models
    if not is_lstm:
        mfcc = mfcc[..., np.newaxis]
    return mfcc

def predict(file_path, num_mfcc=NUM_MFCC, is_lstm=False):
    input_data = preprocess_audio(file_path, num_mfcc, is_lstm)
    
    # Adjust input shape for LSTM and CNN
    if is_lstm:
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Shape: (1, time_steps, num_mfcc)
    else:
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # Shape: (1, time_steps, num_mfcc, 1)
    
    # Set the tensor to point to the input data for inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

def process_folder(folder_path, num_mfcc=NUM_MFCC, is_lstm=False):
    correct_predictions = 0
    total_files = 0
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            prediction = predict(file_path, num_mfcc, is_lstm)
            prediction_label = 'Cry' if prediction > 0.5 else 'Not Cry'
            results.append((file_name, prediction_label))
            ground_truth = 'Cry' if '_cry.wav' in file_name else 'Not Cry'

            if prediction_label == ground_truth:
                correct_predictions += 1

            total_files += 1

    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0

    return results, accuracy


folder_path = '{0}/Test_augmented'.format(AUDIO_PATH)
if MODEL == 'lstm': 
    is_lstm_model = True  
else: 
    is_lstm_model = False
predictions, accuracy = process_folder(folder_path, num_mfcc=NUM_MFCC, is_lstm=is_lstm_model)

for file_name, prediction_label in predictions:
    print(f"File: {file_name}, Prediction: {prediction_label}")

print(f"Prediction Accuracy: {accuracy:.2f}%")
