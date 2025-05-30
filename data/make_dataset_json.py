import os
import json


def list_files_in_directory(directory_path):
    # List all files in the directory
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files


def save_files_to_json(files, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(files, json_file, indent=4)


def make_json(directory_path, output_file):
    # Get the list of files and save to JSON
    files = list_files_in_directory(directory_path)
    save_files_to_json(files, output_file)

# create training set json


def main():
    prefix = "data/VCTK_dataset/"
    # train_clean
    make_json(
        os.path.join(prefix, 'clean_trainset_28spk_wav/'),
        'data/train_clean.json'
    )

    # train_noisy
    make_json(
        os.path.join(prefix, 'noisy_trainset_28spk_wav/'),
        'data/train_noisy.json'
    )
    # ----------------------------------------------------------#
    # create valid set json
    # valid_clean
    make_json(
         os.path.join(prefix, 'clean_validationset_28spk_wav/'),
         'data/valid_clean.json'
    )

    # valid_noisy
    make_json(
        os.path.join(prefix, 'noisy_validationset_28spk_wav/'),
        'data/valid_noisy.json'
    )
    # ----------------------------------------------------------#
    # create testing set json
    # test_clean
    make_json(
       os.path.join(prefix, 'clean_testset_wav/'),
       'data/test_clean.json'
    )

    # test_noisy
    make_json(
       os.path.join(prefix, 'noisy_testset_wav/'),
       'data/test_noisy.json'
    )
    # ----------------------------------------------------------#


if __name__ == '__main__':
    main()
