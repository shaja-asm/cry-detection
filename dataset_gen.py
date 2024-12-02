import os
import random
import shutil

def create_test_dataset(cry_folder, not_cry_folder, test_folder):
    # Create the test folder if it doesn't exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Create subfolders for cry and not cry within the test folder
    test_cry_folder = os.path.join(test_folder, 'cry')
    test_not_cry_folder = os.path.join(test_folder, 'not_cry')
    os.makedirs(test_cry_folder, exist_ok=True)
    os.makedirs(test_not_cry_folder, exist_ok=True)

    # Move 20% of cry files to the test folder
    cry_files = [f for f in os.listdir(cry_folder) if os.path.isfile(os.path.join(cry_folder, f))]
    cry_test_files = random.sample(cry_files, int(0.2 * len(cry_files)))
    
    for file_name in cry_test_files:
        source_path = os.path.join(cry_folder, file_name)
        destination_path = os.path.join(test_cry_folder, file_name)
        shutil.move(source_path, destination_path)

    # Move 20% of not cry files to the test folder
    not_cry_files = [f for f in os.listdir(not_cry_folder) if os.path.isfile(os.path.join(not_cry_folder, f))]
    not_cry_test_files = random.sample(not_cry_files, int(0.2 * len(not_cry_files)))
    
    for file_name in not_cry_test_files:
        source_path = os.path.join(not_cry_folder, file_name)
        destination_path = os.path.join(test_not_cry_folder, file_name)
        shutil.move(source_path, destination_path)

    print(f"Moved {len(cry_test_files)} cry files and {len(not_cry_test_files)} not cry files to the test folder.")

def main():
    # Hard coded folder paths
    cry_folder = '/mnt/d/deBarbaroCry/deBarbaroCry/CryCorpusNew/cry'
    not_cry_folder = '/mnt/d/deBarbaroCry/deBarbaroCry/CryCorpusNew/notcry'
    test_folder = '/mnt/d/deBarbaroCry/deBarbaroCry/CryCorpusNew/test'
    
    create_test_dataset(cry_folder, not_cry_folder, test_folder)

if __name__ == "__main__":
    main()
