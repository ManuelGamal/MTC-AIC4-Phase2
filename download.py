"""
Download the UETrack checkpoint required by predictor.py.
"""

from __future__ import annotations

from pathlib import Path

import gdown


GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/18uOU8gPKn1ejLtfVWgncdKaGvbkavUjC"


def download_checkpoint(file_id=None, checkpoint_dir="checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "model_final.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        return str(checkpoint_path)

    print("Downloading UETrack checkpoint folder...")
    gdown.download_folder(
        url=GOOGLE_DRIVE_FOLDER_URL,
        output=str(checkpoint_dir),
        quiet=False,
        use_cookies=False,
    )

    if not checkpoint_path.exists():
        matches = sorted(checkpoint_dir.rglob("model_final.pth"))
        if matches:
            matches[0].replace(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "model_final.pth was not found after download. "
            "Check the Google Drive folder contents or place the file under checkpoints/ manually."
        )

    return str(checkpoint_path)


if __name__ == "__main__":
    download_checkpoint()
