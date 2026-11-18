#!/usr/bin/env python3
"""Evaluate a saved per-glyph policy and its EMA geometry without new training."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "callirewrite_verifier", Path(__file__).with_name("verify_callirewrite_rl.py"))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    run = args.run_dir.resolve()
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    input_dir = args.input_dir or Path(report["provenance"]["input_image"]).parent
    checkpoint = run / "best_policy.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert not args.require_gpu or device == "cuda", "CUDA unavailable"
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    policy, log_alpha, _ = verifier.build_policy(device, saved["config"])
    policy.load_state_dict(saved["policy_state_dict"], strict=True)
    log_alpha.data.copy_(saved["log_alpha"])
    policy._alpha = log_alpha.detach().exp()
    policy.eval()
    seed = saved["config"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    skeleton = np.load(run / "best_skeleton_rc.npy", allow_pickle=False)
    provenance = dict(report["provenance"], policy_checkpoint=str(checkpoint),
                      policy_sha256=verifier.digest(checkpoint),
                      evaluation_runner_sha256=verifier.digest(__file__),
                      additional_training_steps=0)
    cases = [
        ("zero_action_on_selected_ema_skeleton", None, skeleton, "baseline"),
        ("selected_policy_on_fresh_coarse_skeleton", policy, None, "refined"),
        ("selected_policy_on_selected_ema_skeleton", policy, skeleton, "refined"),
    ]
    results = {}
    for name, candidate, geometry, source_kind in cases:
        result, _ = verifier.evaluate(
            input_dir, run / "ablations" / name, candidate, device,
            skeleton=geometry, seed=seed, source_kind=source_kind, save=True,
            provenance=dict(provenance, ablation_role=name))
        results[name] = result
    original = report["refined"]["iou"]
    reproduced = results["selected_policy_on_selected_ema_skeleton"]["iou"]
    assert abs(original - reproduced) < 1e-12, "Checkpoint evaluation changed"
    results["same_checkpoint_combined_result_reproduced_exactly"] = True
    results["additional_training_steps"] = 0
    results["policy_effect_on_selected_ema_skeleton_iou"] = (
        reproduced - results["zero_action_on_selected_ema_skeleton"]["iou"])
    results["policy_effect_on_fresh_coarse_skeleton_iou"] = (
        results["selected_policy_on_fresh_coarse_skeleton"]["iou"] -
        report["baseline"]["iou"])
    results["interpretation"] = (
        "Fresh independent environments; EMA disabled during each evaluation. "
        "The combined policy/EMA candidate was selected by same-glyph IoU. "
        "These conditional comparisons separate the saved policy's immediate "
        "action effect from the selected EMA geometry; EMA geometry itself was "
        "produced during training and is not an independent optimization method.")
    with np.load(run / "baseline/trajectory.npz", allow_pickle=False) as coarse, \
            np.load(run / "refined/trajectory.npz", allow_pickle=False) as refined:
        original_skeleton = coarse["skeleton_with_stroke_start"]
        selected_skeleton = refined["skeleton_with_stroke_start"]
        displacement = np.linalg.norm(skeleton - original_skeleton[:, 1:], axis=1)
        width = report["baseline"]["native_canvas_width"]
        consistency = {
            "coarse_skeleton_shape": list(original_skeleton.shape),
            "selected_skeleton_shape": list(skeleton.shape),
            "same_skeleton_point_count": len(skeleton) == len(original_skeleton),
            "same_stroke_start_flags": bool(np.array_equal(
                original_skeleton[:, 0], selected_skeleton[:, 0])),
            "same_per_sample_stroke_ids": bool(np.array_equal(
                coarse["stroke_id"], refined["stroke_id"])),
            "same_per_sample_skeleton_indices": bool(np.array_equal(
                coarse["skeleton_index"], refined["skeleton_index"])),
            "same_contact_sample_count": len(coarse["xy_uv"]) == len(refined["xy_uv"]),
            "skeleton_displacement_normalized_image_width_mean": float(displacement.mean()),
            "skeleton_displacement_normalized_image_width_max": float(displacement.max()),
            "skeleton_displacement_native_pixels_mean": float(displacement.mean() * width),
            "skeleton_displacement_native_pixels_max": float(displacement.max() * width),
            "geometry_note": "Sequence consistency only; this does not establish semantic correctness of the extracted stroke order.",
        }
    verifier.dump(run / "skeleton_consistency.json", consistency)
    verifier.dump(run / "ablation_metrics.json", results)
    report["ablations"] = results
    report["skeleton_consistency"] = consistency
    verifier.dump(report_path, report)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
