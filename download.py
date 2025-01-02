import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    try:
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get("content-length", 0))
        
        with requests.get(url, stream=True) as response, open(save_path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=64*1024):
                f.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Finish Downloading: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {url}\nInfo: {e}")

def download_gpt2_files(target_dir):
    base_url = "https://huggingface.co/openai-community/gpt2/resolve/main/"
    
    files = [
        "config.json",
        "merges.txt",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json"
    ]
    
    os.makedirs(target_dir, exist_ok=True)
    
    for file in files:
        file_url = f"{base_url}{file}?download=true"
        save_path = os.path.join(target_dir, file)
        download_file(file_url, save_path)

if __name__ == "__main__":
    
    target_directory = "./models/gpt2"
    download_gpt2_files(target_directory)
