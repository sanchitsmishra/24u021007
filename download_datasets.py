"""
Dataset Downloader & Preparer for Minor Project Evaluation
Downloads and unpacks official evaluation splits for:
1. MS-COCO val2017 (5,000 images + annotations)
2. MSR-VTT 1k-Test (1,000 videos + annotations)
3. UCF-101 Recognition Test Split (101 action classes + videos)
"""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

DATASETS_ROOT = Path(__file__).resolve().parent / "datasets"

URLS = {
    "coco_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "coco_val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "msrvtt_annotations": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/msrvtt_test_1k.json",
    "msrvtt_test_csv": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/raw_data/MSRVTT_JSFUSION_test.csv",
    "msrvtt_videos_zip": "https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/MSRVTT_Videos.zip",
    "ucf101_splits": "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF101TrainTestSplits-RecognitionTask.zip",
    "ucf101_videos_zip": "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip",
}


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def safe_unlink(path: Path) -> None:
    """Safely removes a file on Windows handling lingering file handles."""
    if not path.exists():
        return
    import gc
    gc.collect()
    for _ in range(5):
        try:
            path.unlink()
            return
        except Exception:
            time.sleep(0.5)
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[-] Warning: could not delete temporary archive {path}: {e}")


def download_with_progress(url: str, output_path: Path, description: str) -> None:
    """Downloads a file with dynamic console progress bar and resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    # Check remote file size
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"[-] Failed to query remote size for {url}: {e}")
        total_size = 0

    if output_path.exists() and total_size > 0:
        local_size = output_path.stat().st_size
        if local_size == total_size:
            print(f"[+] {description} already downloaded ({format_bytes(local_size)}). Skipping.")
            return

    downloaded = 0
    if temp_path.exists():
        downloaded = temp_path.stat().st_size

    headers = {"User-Agent": "Mozilla/5.0"}
    if downloaded > 0 and total_size > 0:
        if downloaded < total_size:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"[*] Resuming {description} from {format_bytes(downloaded)} / {format_bytes(total_size)}...")
        else:
            temp_path.replace(output_path)
            print(f"[+] {description} completed ({format_bytes(total_size)}).")
            return
    else:
        print(f"[*] Downloading {description} ({format_bytes(total_size)})...")

    req = urllib.request.Request(url, headers=headers)
    block_size = 1024 * 1024  # 1 MB chunk
    start_time = time.time()
    last_print = start_time

    mode = "ab" if downloaded > 0 else "wb"
    with urllib.request.urlopen(req) as resp, open(temp_path, mode) as out_file:
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 0.5:
                speed = downloaded / (now - start_time) if (now - start_time) > 0 else 0
                pct = (downloaded / total_size * 100) if total_size > 0 else 0.0
                bar_len = 25
                filled = int(bar_len * pct / 100)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(
                    f"\r    [{bar}] {pct:5.1f}% | {format_bytes(downloaded)}/{format_bytes(total_size)} | {format_bytes(int(speed))}/s",
                    end="",
                    flush=True,
                )
                last_print = now

    print(f"\r    [=========================] 100.0% | {format_bytes(downloaded)} | Done!             ")
    if temp_path.exists():
        temp_path.replace(output_path)


def setup_coco() -> None:
    """Downloads COCO val2017 images and annotations."""
    print("\n=======================================================")
    print("           Setting up MS-COCO (val2017)                ")
    print("=======================================================")
    coco_dir = DATASETS_ROOT / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotations
    ann_zip = coco_dir / "annotations_trainval2017.zip"
    captions_file = coco_dir / "annotations" / "captions_val2017.json"
    if not captions_file.exists():
        download_with_progress(URLS["coco_annotations"], ann_zip, "COCO 2017 Annotations")
        print("[*] Extracting COCO validation annotations...")
        with zipfile.ZipFile(ann_zip, "r") as z:
            # Extract only val2017 annotations needed for evaluation to save disk space
            for member in z.namelist():
                if "val2017" in member:
                    z.extract(member, coco_dir)
        print(f"[+] COCO annotations ready at {captions_file}")
    else:
        print(f"[+] COCO annotations already exist at {captions_file}")

    # 2. Validation Images (5,000 images)
    val_zip = coco_dir / "val2017.zip"
    val_images_dir = coco_dir / "val2017"
    if not val_images_dir.exists() or len(list(val_images_dir.glob("*.jpg"))) < 5000:
        download_with_progress(URLS["coco_val_images"], val_zip, "COCO val2017 Images (5,000 images)")
        print("[*] Extracting val2017 images...")
        with zipfile.ZipFile(val_zip, "r") as z:
            z.extractall(coco_dir)
        num_images = len(list(val_images_dir.glob("*.jpg")))
        print(f"[+] Extracted {num_images} images into {val_images_dir}")
        if val_zip.exists():
            val_zip.unlink()  # remove zip to conserve disk space
    else:
        print(f"[+] COCO val2017 images already extracted ({len(list(val_images_dir.glob('*.jpg')))} images).")


def setup_msrvtt() -> None:
    """Downloads MSR-VTT 1k-Test annotations and extracts test video clips."""
    print("\n=======================================================")
    print("           Setting up MSR-VTT (1k-Test Split)          ")
    print("=======================================================")
    msrvtt_dir = DATASETS_ROOT / "msrvtt"
    msrvtt_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = msrvtt_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotations (test_1k json)
    ann_path = ann_dir / "msrvtt_test_1k.json"
    download_with_progress(URLS["msrvtt_annotations"], ann_path, "MSR-VTT 1k-Test Annotations")

    # 2. Test video split CSV
    csv_path = ann_dir / "MSRVTT_JSFUSION_test.csv"
    download_with_progress(URLS["msrvtt_test_csv"], csv_path, "MSR-VTT JSFusion Test CSV")

    # 3. Videos
    videos_dir = msrvtt_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Read test video IDs from JSON or CSV
    with open(ann_path, "r", encoding="utf-8") as f:
        test_info = json.load(f)
    test_video_ids = set()
    if isinstance(test_info, list):
        for item in test_info:
            if "video_id" in item:
                test_video_ids.add(item["video_id"])
    elif isinstance(test_info, dict) and "videos" in test_info:
        for item in test_info["videos"]:
            test_video_ids.add(item["video_id"])

    print(f"[*] Target test video split size: {len(test_video_ids)} videos")

    existing_test_videos = [v for v in videos_dir.glob("*.mp4") if v.stem in test_video_ids]
    if len(existing_test_videos) >= len(test_video_ids) and len(test_video_ids) > 0:
        print(f"[+] MSR-VTT test videos already extracted ({len(existing_test_videos)} clips).")
        return

    # Download MSRVTT_Videos.zip
    video_zip = msrvtt_dir / "MSRVTT_Videos.zip"
    download_with_progress(URLS["msrvtt_videos_zip"], video_zip, "MSR-VTT Video Archive")

    print("[*] Extracting 1k-Test video clips from archive...")
    with zipfile.ZipFile(video_zip, "r") as z:
        members = z.namelist()
        extracted_count = 0
        for m in members:
            fname = Path(m).name
            stem = Path(m).stem
            # Extract if it belongs to test split or extract all if test IDs empty
            if (test_video_ids and stem in test_video_ids) or not test_video_ids:
                if fname.endswith(".mp4"):
                    dest = videos_dir / fname
                    if not dest.exists():
                        with z.open(m) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted_count += 1
                        if extracted_count % 100 == 0:
                            print(f"    Extracted {extracted_count} test videos...")

    print(f"[+] Total MSR-VTT test videos available: {len(list(videos_dir.glob('*.mp4')))} in {videos_dir}")
    if video_zip.exists():
        video_zip.unlink()  # remove archive to save disk space


def setup_ucf101() -> None:
    """Downloads UCF-101 official test splits and recognition test videos."""
    print("\n=======================================================")
    print("           Setting up UCF-101 (Test Split)             ")
    print("=======================================================")
    ucf_dir = DATASETS_ROOT / "ucf101"
    ucf_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = ucf_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Recognition Splits (classInd.txt, testlist01.txt, etc.)
    splits_zip = ucf_dir / "UCF101TrainTestSplits-RecognitionTask.zip"
    class_ind_file = splits_dir / "classInd.txt"
    if not class_ind_file.exists():
        download_with_progress(URLS["ucf101_splits"], splits_zip, "UCF-101 Split Lists")
        print("[*] Extracting split definitions...")
        with zipfile.ZipFile(splits_zip, "r") as z:
            for m in z.namelist():
                if not m.endswith("/"):
                    fname = Path(m).name
                    with z.open(m) as src, open(splits_dir / fname, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        safe_unlink(splits_zip)
        print(f"[+] UCF-101 split files ready in {splits_dir}")
    else:
        print(f"[+] UCF-101 split files already exist in {splits_dir}")

    # Read the 101 action classes
    classes = {}
    with open(class_ind_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                classes[int(parts[0])] = parts[1]
    print(f"[+] Loaded {len(classes)} action categories from classInd.txt")

    # Read testlist01.txt (the official recognition benchmark test split)
    testlist_file = splits_dir / "testlist01.txt"
    test_videos = []
    with open(testlist_file, "r", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if val:
                test_videos.append(val)
    print(f"[*] Official test split contains {len(test_videos)} video clips across 101 classes.")

    videos_dir = ucf_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    existing_videos = list(videos_dir.rglob("*.avi")) + list(videos_dir.rglob("*.mp4"))
    if len(existing_videos) >= len(test_videos):
        print(f"[+] UCF-101 test videos already extracted ({len(existing_videos)} clips).")
        return

    # Download UCF-101.zip
    video_zip = ucf_dir / "UCF-101.zip"
    download_with_progress(URLS["ucf101_videos_zip"], video_zip, "UCF-101 Video Archive (~6.5 GB)")

    print("[*] Extracting test videos into action folders...")
    # Map video filename to relative path in test split
    test_rel_set = set(test_videos)
    with zipfile.ZipFile(video_zip, "r") as z:
        extracted = 0
        for m in z.namelist():
            # e.g., UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
            parts = Path(m).parts
            if len(parts) >= 3 and parts[-1].endswith(".avi"):
                rel_key = f"{parts[-2]}/{parts[-1]}"
                if rel_key in test_rel_set:
                    dest_file = videos_dir / parts[-2] / parts[-1]
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if not dest_file.exists():
                        with z.open(m) as src, open(dest_file, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted += 1
                        if extracted % 200 == 0:
                            print(f"    Extracted {extracted} / {len(test_videos)} test clips...")

    print(f"[+] Extracted {extracted} UCF-101 test videos into {videos_dir}")
    safe_unlink(video_zip)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Evaluation Datasets for Minor Project")
    parser.add_argument(
        "--dataset",
        choices=["coco", "msrvtt", "ucf101", "all"],
        default="all",
        help="Which dataset to download and set up (default: all)",
    )
    args = parser.parse_args()

    print(f"Target datasets directory: {DATASETS_ROOT.resolve()}")
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["coco", "all"]:
        setup_coco()

    if args.dataset in ["msrvtt", "all"]:
        setup_msrvtt()

    if args.dataset in ["ucf101", "all"]:
        setup_ucf101()

    print("\n=======================================================")
    print("               Dataset Setup Completed!                ")
    print(f"  All datasets are structured under: {DATASETS_ROOT.resolve()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
