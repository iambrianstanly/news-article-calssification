
import gdown
import zipfile
import os



def download_and_extract_zip_from_gdrive(file_id, output_zip_path, extract_to='.'):

    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download ZIP file
    print("Downloading...")
    gdown.download(url, output_zip_path, quiet=False)

    # Extract ZIP file
    print("Extracting...")
    with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted to: {os.path.abspath(extract_to)}")

# Example usage
if __name__ == "__main__":
    file_id = '1GC_fIc9WM9tkp5ka7ZKXflkcDNAEu13e'  # Replace with your actual file ID
    download_and_extract_zip_from_gdrive(file_id, './data/downloaded.zip', extract_to='./data/raw/')

