import os
import urllib.request
import ssl


def download_file(url, path):
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return

    print(f"Downloading {url} to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create unverified context to avoid SSL errors on some Macs
    ssl_context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(url, context=ssl_context) as response, open(
            path, "wb"
        ) as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    base_assets_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
    )

    models = [
        (
            "https://huggingface.co/rvc-boss/uvr5_weights/resolve/main/hubert_base.pt",
            os.path.join(base_assets_path, "hubert", "hubert_base.pt"),
        ),
        (
            "https://huggingface.co/rvc-boss/uvr5_weights/resolve/main/rmvpe.pt",
            os.path.join(base_assets_path, "rmvpe", "rmvpe.pt"),
        ),
    ]

    for url, path in models:
        download_file(url, path)
