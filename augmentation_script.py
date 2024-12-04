import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import stft, istft

def process_wav_file(input_file: str, output_file: str) -> None:
    """
    Processes a WAV file to simulate a frequency shift based on the distortion formula.

    Parameters:
    - input_file: Path to the input WAV file.
    - output_file: Path to save the processed WAV file.
    """
    # Read the WAV file
    sample_rate, data = wavfile.read(input_file)
    
    # Ensure data is in float32 format for processing
    if data.dtype != np.float32:
        # Convert data to float32 and normalize if it's in int format
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
    
    # Convert stereo to mono if necessary
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    # Perform Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(data, fs=sample_rate)
    
    # Apply the distortion formula to the frequency bins
    # y = 1.0579x - 0.0055 => x = (y + 0.0055) / 1.0579
    y = f
    x = (y + 0.0055) / 1.0579
    
    # Initialize the modified STFT matrix
    Zxx_modified = np.zeros_like(Zxx, dtype=np.complex64)
    
    # Adjust frequency components
    for i in range(Zxx.shape[1]):
        Zxx_modified[:, i] = np.interp(x, y, Zxx[:, i], left=0, right=0)
    
    # Perform inverse STFT
    _, data_modified = istft(Zxx_modified, fs=sample_rate)
    
    # Optional: Clip values to the [-1.0, 1.0] range to prevent extreme amplitudes
    data_modified = np.clip(data_modified, -1.0, 1.0)
    
    # Write the modified WAV file using float32 data
    # Since WAV files typically expect int data, specify subtype='FLOAT' to write float32 data
    wavfile.write(output_file, sample_rate, data_modified.astype(np.float32))
    
def main() -> None:
    """
    Main function to process all WAV files in the input folder and save them to the output folder.
    """
    input_folder = '/mnt/d/Datasets/CryCorpusNew/test/notcry'
    output_folder = '/mnt/d/Datasets/CryCorpusNew/test_augmented'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each WAV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            print(f'Processing {input_file}...')
            process_wav_file(input_file, output_file)
            print(f'Saved processed file to {output_file}')

if __name__ == '__main__':
    main()
