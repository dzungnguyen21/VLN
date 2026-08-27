import os
import sys
import tarfile
from huggingface_hub import hf_hub_download

def download_and_extract_mp3d_ce():
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    token = None
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            token = f.read().strip()

    dest_dir = os.path.abspath("data/scene_data")
    os.makedirs(dest_dir, exist_ok=True)

    print("=" * 70)
    print("  Downloading Matterport3D CE Scene Data (mp3d_ce.tar.gz ~15GB)")
    print("  Repo: InternRobotics/Scene-N1")
    print("=" * 70)

    try:
        downloaded_path = hf_hub_download(
            repo_id="InternRobotics/Scene-N1",
            filename="mp3d_ce.tar.gz",
            repo_type="dataset",
            token=token,
            local_dir=dest_dir,
        )
        print(f"\nDownload completed: {downloaded_path}")
    except Exception as e:
        print(f"\n[Error] Failed to download mp3d_ce.tar.gz: {e}")
        sys.exit(1)

    print(f"\nExtracting archive into {dest_dir}...")
    try:
        with tarfile.open(downloaded_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction completed successfully!")
    except Exception as e:
        print(f"[Error] Failed to extract archive: {e}")
        sys.exit(1)

    # Clean up archive to save disk space
    if os.path.exists(downloaded_path):
        os.remove(downloaded_path)
        print("Cleaned up temporary archive.")

    print("\nScene data directory ready at: data/scene_data/mp3d_ce")

if __name__ == "__main__":
    download_and_extract_mp3d_ce()

