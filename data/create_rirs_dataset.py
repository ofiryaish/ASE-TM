import requests
import os
import zipfile


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # URLs of the dataset
    urls = [
        "https://www.openslr.org/resources/26/sim_rir_16k.zip",
        # "https://www.openslr.org/resources/28/rirs_noises.zip"
    ]

    for url in urls:
        # Path to save the downloaded file
        output_path = os.path.join(SCRIPT_DIR, url.split("/")[-1])

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping download.")
        else:
            # Download the file
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"File downloaded successfully and saved as {output_path}")

            else:
                print(f"Failed to download file. Status code: {response.status_code}")
                continue

        extract_path = os.path.join(SCRIPT_DIR, output_path.replace(".zip", ""))
        # Unzip the file
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"Extracted {output_path} to {extract_path}")


if __name__ == "__main__":
    main()
