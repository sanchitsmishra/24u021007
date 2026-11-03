"""
Evaluation Runner for Video Caption Generator
Evaluates on MSR-VTT, MS-COCO, and UCF-101 datasets.
Outputs academic benchmark scores, detailed metrics, and qualitative error analysis.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

from PIL import Image
import torch

# Ensure app imports resolve
APP_DIR = Path(__file__).resolve().parent / "video-caption-generator-main"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.caption_service import VideoCaptionService
from metrics import CaptionMetrics, tokenize

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate_msrvtt(service: VideoCaptionService, num_samples: int = 50) -> Dict[str, Any]:
    """Evaluates video captioning pipeline on MSR-VTT 1k-Test split."""
    print("\n" + "=" * 65)
    print("           Evaluating on MSR-VTT (Video Captioning)            ")
    print("=" * 65)

    msr_dir = DATASETS_DIR / "msrvtt"
    full_ann_path = msr_dir / "annotations" / "MSRVTT_data.json"
    ann_path = full_ann_path if full_ann_path.exists() else (msr_dir / "annotations" / "msrvtt_test_1k.json")
    videos_dir = msr_dir / "videos"

    if not ann_path.exists() or not videos_dir.exists():
        raise FileNotFoundError(f"MSR-VTT dataset not found at {msr_dir}. Run download_datasets.py first.")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build reference mapping: video_id -> [20 captions]
    ground_truth: Dict[str, List[str]] = {}
    if isinstance(data, list):
        for item in data:
            v_id = item["video_id"]
            ground_truth.setdefault(v_id, []).append(item["caption"])
    elif isinstance(data, dict):
        if "sentences" in data:
            for s in data["sentences"]:
                ground_truth.setdefault(s["video_id"], []).append(s["caption"])
        elif "videos" in data:
            for v in data["videos"]:
                ground_truth[v["video_id"]] = v.get("captions", [])

    available_videos = sorted(list(videos_dir.glob("*.mp4")))
    if not available_videos:
        raise FileNotFoundError("No .mp4 videos found in MSR-VTT videos directory.")

    # Limit to num_samples if specified
    eval_videos = available_videos if num_samples <= 0 else available_videos[:num_samples]
    print(f"[*] Running evaluation on {len(eval_videos)} video clips...")

    metric_evaluator = CaptionMetrics()
    predictions: Dict[str, str] = {}
    detailed_results = []
    latencies = []

    for i, v_path in enumerate(eval_videos, 1):
        v_id = v_path.stem
        start_t = time.time()
        try:
            caption, _, _ = service.caption_video(v_path)
        except Exception as e:
            print(f"[-] Error processing {v_path.name}: {e}")
            caption = ""
        duration = time.time() - start_t
        latencies.append(duration)

        predictions[v_id] = caption
        refs = ground_truth.get(v_id, [])

        sample_scores = metric_evaluator.compute_sentence_metrics(caption, refs)
        detailed_results.append({
            "video_id": v_id,
            "filename": v_path.name,
            "prediction": caption,
            "references": refs[:5],  # store top 5 references for review
            "metrics": sample_scores,
            "latency_seconds": round(duration, 2),
        })

        log_step = 10 if len(eval_videos) > 50 else 5
        if i % log_step == 0 or i == len(eval_videos):
            print(f"    [{i:4d}/{len(eval_videos)}] {v_id} | {duration:.2f}s | Pred: \"{caption[:50]}...\"")

        # Checkpoint every 50 videos to safeguard progress
        if i % 50 == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            chk_path = RESULTS_DIR / "msrvtt_checkpoint.json"
            try:
                with open(chk_path, "w", encoding="utf-8") as chk_f:
                    json.dump({"completed": i, "total": len(eval_videos), "predictions": predictions}, chk_f)
            except Exception:
                pass

    # Compute overall corpus metrics
    overall_cider, per_item_cider = metric_evaluator.compute_corpus_cider(predictions, ground_truth)
    for res in detailed_results:
        res["metrics"]["CIDEr"] = per_item_cider.get(res["video_id"], 0.0)

    avg_bleu1 = sum(r["metrics"]["BLEU-1"] for r in detailed_results) / len(detailed_results)
    avg_bleu2 = sum(r["metrics"]["BLEU-2"] for r in detailed_results) / len(detailed_results)
    avg_bleu3 = sum(r["metrics"]["BLEU-3"] for r in detailed_results) / len(detailed_results)
    avg_bleu4 = sum(r["metrics"]["BLEU-4"] for r in detailed_results) / len(detailed_results)
    avg_rouge = sum(r["metrics"]["ROUGE-L"] for r in detailed_results) / len(detailed_results)
    avg_meteor = sum(r["metrics"]["METEOR"] for r in detailed_results) / len(detailed_results)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    summary = {
        "dataset": "MSR-VTT",
        "total_samples": len(eval_videos),
        "scores": {
            "CIDEr": round(overall_cider, 2),
            "BLEU-4": round(avg_bleu4, 2),
            "BLEU-3": round(avg_bleu3, 2),
            "BLEU-2": round(avg_bleu2, 2),
            "BLEU-1": round(avg_bleu1, 2),
            "METEOR": round(avg_meteor, 2),
            "ROUGE-L": round(avg_rouge, 2),
        },
        "performance": {
            "average_latency_sec": round(avg_latency, 2),
            "device": service.device,
        },
        "details": detailed_results,
    }

    _print_score_card("MSR-VTT Video Captioning Benchmark", summary["scores"])
    return summary


def evaluate_coco(service: VideoCaptionService, num_samples: int = 50) -> Dict[str, Any]:
    """Evaluates BLIP image captioning & YOLO count verification on MS-COCO val2017."""
    print("\n" + "=" * 65)
    print("           Evaluating on MS-COCO (Image Captioning)            ")
    print("=" * 65)

    coco_dir = DATASETS_DIR / "coco"
    captions_path = coco_dir / "annotations" / "captions_val2017.json"
    images_dir = coco_dir / "val2017"

    if not captions_path.exists() or not images_dir.exists():
        raise FileNotFoundError(f"COCO dataset not found at {coco_dir}. Run download_datasets.py first.")

    with open(captions_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map image_id -> [5 captions] & filename
    ground_truth: Dict[str, List[str]] = {}
    id_to_filename = {}
    for img in coco_data["images"]:
        id_to_filename[str(img["id"])] = img["file_name"]

    for ann in coco_data["annotations"]:
        img_id_str = str(ann["image_id"])
        ground_truth.setdefault(img_id_str, []).append(ann["caption"])

    available_img_ids = [img_id for img_id in ground_truth if (images_dir / id_to_filename[img_id]).exists()]
    eval_ids = available_img_ids if num_samples <= 0 else available_img_ids[:num_samples]
    print(f"[*] Running evaluation on {len(eval_ids)} images...")

    service._ensure_model_loaded()
    metric_evaluator = CaptionMetrics()
    chk_path = RESULTS_DIR / "coco_checkpoint.json"
    predictions: Dict[str, str] = {}
    latencies: List[float] = []
    sentence_metrics: List[Dict[str, float]] = []
    detailed_results: List[Dict[str, Any]] = []
    start_idx = 0

    if chk_path.exists():
        try:
            with open(chk_path, "r", encoding="utf-8") as chk_f:
                chk = json.load(chk_f)
            predictions = chk.get("predictions", {})
            latencies = chk.get("latencies", [])
            sentence_metrics = chk.get("sentence_metrics", [])
            detailed_results = chk.get("detailed_results", [])
            start_idx = chk.get("completed", len(predictions))
            print(f"[*] Resumed COCO checkpoint at {start_idx}/{len(eval_ids)} images.")
        except Exception as e:
            print(f"[-] Could not load COCO checkpoint: {e}")

    for i, img_id in enumerate(eval_ids[start_idx:], start_idx + 1):
        filename = id_to_filename[img_id]
        img_path = images_dir / filename
        start_t = time.time()
        try:
            with Image.open(img_path).convert("RGB") as img:
                raw_caption = service._caption_image(img)
                count = service._count_people(img)
                verified_caption = service._apply_person_count(raw_caption, count)
        except Exception as e:
            print(f"[-] Error processing {filename}: {e}")
            verified_caption = ""
        duration = time.time() - start_t
        latencies.append(duration)

        predictions[img_id] = verified_caption
        refs = ground_truth.get(img_id, [])

        sample_scores = metric_evaluator.compute_sentence_metrics(verified_caption, refs)
        sentence_metrics.append(sample_scores)
        if len(detailed_results) < 50:
            detailed_results.append({
                "image_id": img_id,
                "filename": filename,
                "prediction": verified_caption,
                "references": refs[:5],
                "metrics": sample_scores,
                "latency_seconds": round(duration, 3),
            })

        log_step = 25 if len(eval_ids) > 100 else 5
        if i % log_step == 0 or i == len(eval_ids):
            print(f"    [{i:4d}/{len(eval_ids)}] {filename} | {duration:.2f}s | Pred: \"{verified_caption[:45]}\"")

        if i % 100 == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with open(chk_path, "w", encoding="utf-8") as chk_f:
                    json.dump({
                        "completed": i,
                        "total": len(eval_ids),
                        "predictions": predictions,
                        "latencies": latencies,
                        "sentence_metrics": sentence_metrics,
                        "detailed_results": detailed_results,
                    }, chk_f)
            except Exception:
                pass

    if chk_path.exists():
        try:
            chk_path.unlink()
        except Exception:
            pass

    overall_cider, per_item_cider = metric_evaluator.compute_corpus_cider(predictions, ground_truth)
    for res in detailed_results:
        res["metrics"]["CIDEr"] = per_item_cider.get(res["image_id"], 0.0)

    avg_bleu1 = sum(m.get("BLEU-1", 0.0) for m in sentence_metrics) / len(sentence_metrics) if sentence_metrics else 0.0
    avg_bleu4 = sum(m.get("BLEU-4", 0.0) for m in sentence_metrics) / len(sentence_metrics) if sentence_metrics else 0.0
    avg_rouge = sum(m.get("ROUGE-L", 0.0) for m in sentence_metrics) / len(sentence_metrics) if sentence_metrics else 0.0
    avg_meteor = sum(m.get("METEOR", 0.0) for m in sentence_metrics) / len(sentence_metrics) if sentence_metrics else 0.0

    summary = {
        "dataset": "MS-COCO val2017",
        "total_samples": len(eval_ids),
        "scores": {
            "CIDEr": round(overall_cider, 2),
            "BLEU-4": round(avg_bleu4, 2),
            "BLEU-1": round(avg_bleu1, 2),
            "METEOR": round(avg_meteor, 2),
            "ROUGE-L": round(avg_rouge, 2),
        },
        "performance": {
            "average_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "device": service.device,
        },
        "details": detailed_results,
    }

    _print_score_card("MS-COCO Image Captioning Benchmark", summary["scores"])
    return summary


def evaluate_ucf101(service: VideoCaptionService, num_samples: int = 50) -> Dict[str, Any]:
    """Evaluates video captioning pipeline on UCF-101 Action Recognition."""
    print("\n" + "=" * 65)
    print("           Evaluating on UCF-101 (Action Recognition)          ")
    print("=" * 65)

    ucf_dir = DATASETS_DIR / "ucf101"
    class_ind_file = ucf_dir / "splits" / "classInd.txt"
    test_list_file = ucf_dir / "splits" / "testlist01.txt"
    videos_dir = ucf_dir / "videos"

    if not class_ind_file.exists() or not videos_dir.exists():
        raise FileNotFoundError(f"UCF-101 dataset not found at {ucf_dir}. Run download_datasets.py first.")

    # Read class mappings
    classes = {}
    with open(class_ind_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                classes[int(parts[0])] = parts[1]
    all_class_names = list(classes.values())

    # Find test videos present on disk
    test_entries = []
    with open(test_list_file, "r", encoding="utf-8") as f:
        for line in f:
            rel_p = line.strip()
            if rel_p and (videos_dir / rel_p).exists():
                category = rel_p.split("/")[0] if "/" in rel_p else rel_p.split("\\")[0]
                test_entries.append((videos_dir / rel_p, category))

    if not test_entries:
        # Fallback to scanning action directories
        for act_dir in sorted(videos_dir.iterdir()):
            if act_dir.is_dir():
                for v in act_dir.glob("*.avi"):
                    test_entries.append((v, act_dir.name))

    eval_entries = test_entries if num_samples <= 0 else test_entries[:num_samples]
    print(f"[*] Running action recognition evaluation on {len(eval_entries)} action clips...")

    # Action keywords pre-parsing for semantic matching
    def parse_action_words(label: str) -> set:
        import re
        words = re.findall(r"[A-Z][a-z]*", label)
        return {w.lower() for w in words if w}

    class_keywords = {name: parse_action_words(name) for name in all_class_names}

    chk_path = RESULTS_DIR / "ucf101_checkpoint.json"
    correct_top1 = 0
    correct_top5 = 0
    keyword_hits = 0
    detailed_results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    start_idx = 0

    if chk_path.exists():
        try:
            with open(chk_path, "r", encoding="utf-8") as chk_f:
                chk = json.load(chk_f)
            correct_top1 = chk.get("correct_top1", 0)
            correct_top5 = chk.get("correct_top5", 0)
            keyword_hits = chk.get("keyword_hits", 0)
            detailed_results = chk.get("detailed_results", [])
            latencies = chk.get("latencies", [])
            start_idx = chk.get("completed", 0)
            print(f"[*] Resumed UCF-101 checkpoint at {start_idx}/{len(eval_entries)} videos.")
        except Exception as e:
            print(f"[-] Could not load UCF-101 checkpoint: {e}")

    for i, (v_path, target_action) in enumerate(eval_entries[start_idx:], start_idx + 1):
        start_t = time.time()
        try:
            caption, _, _ = service.caption_video(v_path)
        except Exception as e:
            print(f"[-] Error processing {v_path.name}: {e}")
            caption = ""
        duration = time.time() - start_t
        latencies.append(duration)

        pred_tokens = set(tokenize(caption))
        target_words = class_keywords.get(target_action, {target_action.lower()})

        # Check action keyword hit
        has_hit = bool(pred_tokens & target_words)
        if has_hit:
            keyword_hits += 1

        # Score all 101 classes based on word overlap with generated caption
        scores = []
        for c_name in all_class_names:
            c_words = class_keywords[c_name]
            overlap = len(pred_tokens & c_words) / len(c_words) if c_words else 0.0
            scores.append((overlap, c_name))

        scores.sort(key=lambda x: x[0], reverse=True)
        top5_predictions = [c for _, c in scores[:5]]

        is_top1 = top5_predictions[0] == target_action and scores[0][0] > 0
        is_top5 = target_action in top5_predictions and any(s[0] > 0 for s in scores[:5] if s[1] == target_action)

        if is_top1:
            correct_top1 += 1
        if is_top5:
            correct_top5 += 1

        if len(detailed_results) < 50:
            detailed_results.append({
                "video_file": v_path.name,
                "ground_truth_action": target_action,
                "predicted_caption": caption,
                "top5_matched_actions": top5_predictions,
                "top1_match": is_top1,
                "top5_match": is_top5,
                "action_keyword_hit": has_hit,
                "latency_seconds": round(duration, 2),
            })

        status = "TOP-1" if is_top1 else "TOP-5" if is_top5 else "MISS"
        log_step = 25 if len(eval_entries) > 100 else 5
        if i % log_step == 0 or i == len(eval_entries):
            print(f"    [{i:4d}/{len(eval_entries)}] Target: {target_action:20s} | [{status:5s}] | Pred: \"{caption[:45]}...\"")

        if i % 50 == 0:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                with open(chk_path, "w", encoding="utf-8") as chk_f:
                    json.dump({
                        "completed": i,
                        "total": len(eval_entries),
                        "correct_top1": correct_top1,
                        "correct_top5": correct_top5,
                        "keyword_hits": keyword_hits,
                        "detailed_results": detailed_results,
                        "latencies": latencies,
                    }, chk_f)
            except Exception:
                pass

    if chk_path.exists():
        try:
            chk_path.unlink()
        except Exception:
            pass

    total = len(eval_entries)
    summary = {
        "dataset": "UCF-101",
        "total_samples": total,
        "scores": {
            "Top-1 Action Accuracy (%)": round((correct_top1 / total) * 100.0, 2) if total else 0.0,
            "Top-5 Action Accuracy (%)": round((correct_top5 / total) * 100.0, 2) if total else 0.0,
            "Action Keyword Recall (%)": round((keyword_hits / total) * 100.0, 2) if total else 0.0,
        },
        "performance": {
            "average_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "device": service.device,
        },
        "details": detailed_results,
    }

    _print_score_card("UCF-101 Action Recognition Benchmark", summary["scores"])
    return summary


def _print_score_card(title: str, scores: Dict[str, Any]) -> None:
    """Pretty prints score summary table."""
    print("\n" + "-" * 50)
    print(f"  RESULT: {title}")
    print("-" * 50)
    for metric, val in scores.items():
        print(f"  {metric:30s} : {val}")
    print("-" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Video Caption Generator on Benchmark Datasets")
    parser.add_argument(
        "--dataset",
        choices=["msrvtt", "coco", "ucf101", "all"],
        default="msrvtt",
        help="Which benchmark to run (default: msrvtt)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples to evaluate (default: 30, -1 for full dataset)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of benchmarks even if existing results are cached",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("        INITIALIZING VIDEO CAPTION GENERATOR BENCHMARK         ")
    print("=" * 65)

    service = VideoCaptionService()
    print(f"[*] Inference Device: {service.device.upper()}")

    results_all: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "device": service.device,
    }
    master_path = RESULTS_DIR / "evaluation_report.json"

    # Pre-load existing report if available to keep all datasets merged
    if master_path.exists():
        try:
            with open(master_path, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
                if isinstance(existing_report, dict):
                    results_all.update(existing_report)
        except Exception:
            pass

    if args.dataset in ["msrvtt", "all"]:
        msr_res_file = RESULTS_DIR / "msrvtt_eval_results.json"
        if not args.force and msr_res_file.exists():
            try:
                with open(msr_res_file, "r", encoding="utf-8") as f:
                    cached_msr = json.load(f)
                cached_count = cached_msr.get("total_samples", 0)
                avail_clips = len(list((DATASETS_DIR / "msrvtt" / "videos").glob("*.mp4")))
                needed_clips = avail_clips if args.num_samples <= 0 else min(args.num_samples, avail_clips)
                if cached_count >= needed_clips:
                    print(f"[+] Reusing existing MSR-VTT evaluation results ({cached_count} samples). Use --force to re-run.")
                    msr_res = cached_msr
                else:
                    msr_res = evaluate_msrvtt(service, num_samples=args.num_samples)
            except Exception:
                msr_res = evaluate_msrvtt(service, num_samples=args.num_samples)
        else:
            msr_res = evaluate_msrvtt(service, num_samples=args.num_samples)

        results_all["msrvtt"] = msr_res
        with open(RESULTS_DIR / "msrvtt_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(msr_res, f, indent=2)
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=2)

    if args.dataset in ["coco", "all"]:
        coco_res_file = RESULTS_DIR / "coco_eval_results.json"
        if not args.force and coco_res_file.exists():
            try:
                with open(coco_res_file, "r", encoding="utf-8") as f:
                    cached_coco = json.load(f)
                cached_count = cached_coco.get("total_samples", 0)
                avail_imgs = len(list((DATASETS_DIR / "coco" / "val2017").glob("*.jpg")))
                needed_imgs = avail_imgs if args.num_samples <= 0 else min(args.num_samples, avail_imgs)
                if cached_count >= needed_imgs:
                    print(f"[+] Reusing existing COCO evaluation results ({cached_count} samples). Use --force to re-run.")
                    coco_res = cached_coco
                else:
                    coco_res = evaluate_coco(service, num_samples=args.num_samples)
            except Exception:
                coco_res = evaluate_coco(service, num_samples=args.num_samples)
        else:
            coco_res = evaluate_coco(service, num_samples=args.num_samples)

        results_all["coco"] = coco_res
        with open(RESULTS_DIR / "coco_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(coco_res, f, indent=2)
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=2)

    if args.dataset in ["ucf101", "all"]:
        ucf_res_file = RESULTS_DIR / "ucf101_eval_results.json"
        if not args.force and ucf_res_file.exists():
            try:
                with open(ucf_res_file, "r", encoding="utf-8") as f:
                    cached_ucf = json.load(f)
                cached_count = cached_ucf.get("total_samples", 0)
                avail_vids = len(list((DATASETS_DIR / "ucf101" / "videos").rglob("*.avi")))
                needed_vids = avail_vids if args.num_samples <= 0 else min(args.num_samples, avail_vids)
                if cached_count >= needed_vids:
                    print(f"[+] Reusing existing UCF-101 evaluation results ({cached_count} samples). Use --force to re-run.")
                    ucf_res = cached_ucf
                else:
                    ucf_res = evaluate_ucf101(service, num_samples=args.num_samples)
            except Exception:
                ucf_res = evaluate_ucf101(service, num_samples=args.num_samples)
        else:
            ucf_res = evaluate_ucf101(service, num_samples=args.num_samples)

        results_all["ucf101"] = ucf_res
        with open(RESULTS_DIR / "ucf101_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(ucf_res, f, indent=2)
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  [+] Complete evaluation report saved to: {master_path}")

    # Automatically generate PDF and HTML scorecards
    try:
        from generate_scorecard import build_pdf_scorecard, build_html_scorecard, generate_benchmark_charts
        chart_paths = generate_benchmark_charts(results_all, RESULTS_DIR / "charts")
        pdf_path = RESULTS_DIR / "evaluation_scorecard.pdf"
        html_path = RESULTS_DIR / "evaluation_scorecard.html"
        build_pdf_scorecard(results_all, chart_paths, pdf_path)
        build_html_scorecard(results_all, html_path)
        print(f"  [+] PDF Scorecard generated: {pdf_path}")
        print(f"  [+] HTML Scorecard generated: {html_path}")
    except Exception as e:
        print(f"  [-] Note: Scorecard generation skipped: {e}")

    print("=" * 65)


if __name__ == "__main__":
    main()
