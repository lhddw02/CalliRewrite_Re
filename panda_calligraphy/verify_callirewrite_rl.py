#!/usr/bin/env python3
"""Bounded per-glyph SAC optimization and faithful native CalliEnv export.

This utility changes no official environment, reward, geometry or raster code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "rl_finetune"
sys.path.insert(0, str(RL))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_env(input_dir, output_dir, *, train=False, image_iter=10):
    import gym
    import Callienv
    from Callienv.envs.tools import Writing_Brush

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = Writing_Brush(**json.loads((RL / "tool_property/brush.json").read_text()))
    return gym.make(
        "CalliEnv-v0", tool=tool,
        folder_path=str(Path(input_dir).resolve()) + os.sep,
        output_path=str(output_dir), visualize_path=None,
        env_num=1, env_rank=(0, 1), render_mode=None,
        image_iter=image_iter,
        start_update=1 if train else 10**9,
        update=1, ema_gamma=0.9, new_step_api=True,
    )


def build_policy(device, config):
    import numpy as np
    import torch
    from MLP.model import My_MLP
    from tianshou.policy import SACPolicy
    from tianshou.utils.net.continuous import ActorProb, Critic

    actor = ActorProb(My_MLP(8, hidden_sizes=(128, 256), device=device), 2,
                      hidden_sizes=(64,), device=device).to(device)
    critics = [Critic(My_MLP(8, 2, hidden_sizes=(128, 256, 256), concat=True,
                             device=device), device=device).to(device) for _ in range(2)]
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config["actor_lr"])
    optim1 = torch.optim.Adam(critics[0].parameters(), lr=config["critic_lr"])
    optim2 = torch.optim.Adam(critics[1].parameters(), lr=config["critic_lr"])
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=config["actor_lr"])
    target_entropy = 0.98 * float(np.log(2.0))
    policy = SACPolicy(actor, actor_optim, critics[0], optim1, critics[1], optim2,
                       gamma=0.9, tau=0.005,
                       alpha=(target_entropy, log_alpha, alpha_optim))
    return policy, log_alpha, [actor_optim, optim1, optim2, alpha_optim]


def metrics(mask, target):
    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(mask, target).sum())
    union = int(np.logical_or(mask, target).sum())
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = []
    for image in (mask, target):
        raw = image.astype(np.uint8)
        edges.append(raw.astype(bool) & ~cv2.erode(raw, kernel).astype(bool))
    if not edges[0].any() or not edges[1].any():
        chamfer = None
    else:
        distances = [cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2,
                                          cv2.DIST_MASK_PRECISE) for edge in edges]
        chamfer = float((distances[1][edges[0]].mean() +
                         distances[0][edges[1]].mean()) / 2)
    return {
        "iou": intersection / union if union else 1.0,
        "chamfer_symmetric_mean_pixels": chamfer,
        "chamfer_normalized_by_width": chamfer / mask.shape[1] if chamfer is not None else None,
        "xor_pixels": int(np.logical_xor(mask, target).sum()),
        "native_canvas_width": int(mask.shape[1]),
        "target_ink_pixels": int(target.sum()),
        "rendered_ink_pixels": int(mask.sum()),
        "metric_note": "Symmetric mean Euclidean contour distance on native 300px masks; not claimed identical to paper Chamfer implementation.",
    }


def evaluate(input_dir, output_dir, policy, device, *, skeleton=None, seed=42,
             source_kind="refined", save=False, provenance=None):
    import numpy as np
    import torch
    from tianshou.data import Batch

    env = make_env(input_dir, Path(output_dir) / "env_outputs", train=False)
    raw = env.unwrapped
    samples = []
    rewards = []
    action_norms = []
    try:
        obs = env.reset(seed=seed)
        if skeleton is not None:
            raw.skel_list = np.asarray(skeleton).copy()
            # Recompute reset state for the supplied geometry without triggering a reload.
            raw.counter = 1
            obs = env.reset(seed=seed)
        initial_skeleton = raw.skel_list.copy()
        points_with_markers = np.column_stack((raw.pt_list, initial_skeleton))
        for step in range(20000):
            if policy is None:
                action = np.zeros(2, dtype=np.float32)
            else:
                with torch.no_grad():
                    action = policy(Batch(obs=np.asarray(obs)[None], info=Batch())).act[0]
                action = action.detach().cpu().numpy().astype(np.float32)
            # These are the actual requested offsets computed in official step().
            index = raw.skel_list_cnter + 1
            theta = float(action[1]) * math.pi
            requested_rc = np.clip(
                raw.skel_list[index] + np.array([-math.sin(theta), math.cos(theta)]) *
                float(action[0]) * raw.r_prime_bound, 0.0, 0.99)
            obs, reward, terminated, truncated, info = env.step(action)
            state = np.asarray(raw.state)
            footprint, radius = raw.calc_four_points(state)
            anchor_rc = np.asarray(footprint[0])
            stroke_id = int(np.searchsorted(raw.pt_idx[1:], raw.skel_list_cnter,
                                            side="right"))
            samples.append({
                "xy_uv": anchor_rc[::-1].copy(),
                "requested_xy_uv": requested_rc[::-1].copy(),
                "radius_norm": float(radius), "length_norm": float(state[2]),
                "heading_deg": float(state[3] * raw.theta_max),
                "stroke_id": stroke_id, "action": action.copy(),
                "footprint_points_uv": np.asarray(footprint)[:, ::-1].copy(),
                "skeleton_index": int(raw.skel_list_cnter),
            })
            rewards.append(float(reward))
            action_norms.append(float(np.linalg.norm(action)))
            assert np.isfinite(state).all() and np.isfinite(reward)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Evaluation did not terminate")
        mask = np.asarray(raw.record_canvas > 0).copy()
        target = np.asarray(raw.stroke_downsample > 0.5).copy()
        result = metrics(mask, target)
        result.update(steps=len(samples), reward_sum=float(sum(rewards)),
                      mean_action_norm=float(np.mean(action_norms)),
                      stroke_count=len(set(sample["stroke_id"] for sample in samples)))
        if save:
            import cv2
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / "native_rendered.png"), (~mask).astype(np.uint8)*255)
            cv2.imwrite(str(output_dir / "target.png"), (~target).astype(np.uint8)*255)
            overlay = np.full((*mask.shape, 3), 255, dtype=np.uint8)
            overlay[target & ~mask] = (80, 80, 230)
            overlay[mask & ~target] = (230, 130, 50)
            overlay[mask & target] = (25, 25, 25)
            cv2.imwrite(str(output_dir / "overlay.png"), overlay)
            arrays = {key: np.asarray([sample[key] for sample in samples])
                      for key in samples[0]}
            # Reconstruct the native union independently, using EXACT official raster semantics.
            reconstructed = np.zeros_like(mask)
            raster_calls = 0
            for sample in samples:
                rr, cc = raw.tool.draw_canvas(raw.canvas_width,
                                               sample["footprint_points_uv"][:, ::-1],
                                               sample["radius_norm"])
                reconstructed[rr, cc] = True
                raster_calls += 1 + int(int(sample["radius_norm"] * raw.canvas_width) != 0)
            assert np.array_equal(reconstructed, mask), "Export/native mask mismatch"
            metadata = dict(provenance or {})
            metadata.update({
                "schema": "callirewrite.native-brush-contact.v1",
                "source_kind": source_kind, "coordinate_order": "u=image column, v=image row",
                "coordinate_units": "normalized image width", "heading_units": "degrees",
                "xy_uv_semantics": "Exact native rendered/reward footprint anchor from calc_four_points, paired with SAME returned state's radius/length/heading.",
                "requested_xy_uv_semantics": "Separate step() action-offset skeleton request; differs from native footprint anchor in official code.",
                "native_canvas_width": raw.canvas_width,
                "native_stamp_count": len(samples), "native_rasterizer_call_count": raster_calls,
                "native_raster_convention": "tools.Writing_Brush.draw_canvas: integer center and polygon vertices multiplied by canvas_width; disk radius is normalized current_r passed directly as pixel radius (preserved official artifact).",
                "native_reconstruction_exact": True,
                "timing": "No physical timestamps exported; each row is one official environment contact sample.",
                "stroke_id_semantics": "Contiguous zero-based stroke from original pt_idx and current skeleton index; no bridge across ids.",
                "ema_during_evaluation": False,
                "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in
                                   [RL/"Callienv/envs/Callienv.py", RL/"Callienv/envs/tools.py"]},
            })
            arrays.update(metadata_json=np.asarray(json.dumps(metadata)),
                          skeleton_with_stroke_start=points_with_markers,
                          native_ink_mask=mask, target_ink_mask=target,
                          reward=np.asarray(rewards))
            np.savez_compressed(output_dir / "trajectory.npz", **arrays)
            dump(output_dir / "metrics.json", result)
            dump(output_dir / "manifest.json", metadata)
        return result, initial_skeleton
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--glyph-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=512)
    parser.add_argument("--max-training-seconds", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--evaluate-skeleton", type=Path)
    args = parser.parse_args()
    import numpy as np
    import torch
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.input_dir = args.input_dir.resolve()
    assert (args.input_dir/"0.png").is_file() and (args.input_dir/"0.npy").is_file()
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert not args.require_gpu or device == "cuda", "CUDA unavailable"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    config = {"actor_lr": 3e-5, "critic_lr": 1e-4, "gamma": 0.9, "tau": 0.005,
              "independent_critic_backbones": True,
              "actor_hidden": [128, 256], "critic_hidden": [128, 256, 256],
              "actor_distribution_hidden": [64], "batch_size": args.batch_size,
              "buffer_size": args.buffer_size, "update_per_step": 1,
              "image_iter": 10, "start_update": 1, "ema_gamma": 0.9,
              "seed": args.seed, "requested_steps": args.steps,
              "eval_every": args.eval_every, "warmup_steps": args.warmup,
              "max_training_seconds": args.max_training_seconds}
    provenance = {"glyph_id": args.glyph_id, "experiment": "per-glyph optimization, not held-out generalization",
                  "input_image": str(args.input_dir/"0.png"),
                  "input_image_sha256": digest(args.input_dir/"0.png"),
                  "input_global_curves": str(args.input_dir/"0.npy"),
                  "input_global_curves_sha256": digest(args.input_dir/"0.npy"),
                  "config": config, "reward": "Unmodified released CalliEnv.calc_reward",
                  "argv": sys.argv,
                  "verification_runner_sha256": digest(__file__)}
    baseline, coarse_skeleton = evaluate(args.input_dir, args.output/"baseline", None,
                                        device, seed=args.seed, source_kind="baseline",
                                        save=True, provenance=provenance)
    print(json.dumps({"stage": "baseline", "device": device, "metrics": baseline}), flush=True)
    if args.evaluate_checkpoint:
        checkpoint = args.evaluate_checkpoint.resolve()
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        loaded, loaded_log_alpha, _ = build_policy(device, saved["config"])
        loaded.load_state_dict(saved["policy_state_dict"], strict=True)
        loaded_log_alpha.data.copy_(saved["log_alpha"])
        loaded._alpha = loaded_log_alpha.detach().exp()
        loaded.eval()
        skeleton_path = args.evaluate_skeleton or checkpoint.parent/"best_skeleton_rc.npy"
        skeleton = np.load(skeleton_path, allow_pickle=False)
        refined, _ = evaluate(args.input_dir, args.output/"refined", loaded, device,
                              skeleton=skeleton, seed=args.seed, source_kind="refined", save=True,
                              provenance=dict(provenance, policy_checkpoint=str(checkpoint),
                                              policy_sha256=digest(checkpoint),
                                              original_training=saved["provenance"]))
        report = {"status": "passed", "mode": "checkpoint evaluation only",
                  "baseline": baseline, "refined": refined,
                  "iou_change": refined["iou"]-baseline["iou"],
                  "checkpoint": str(checkpoint), "skeleton": str(skeleton_path),
                  "new_training_steps": 0, "provenance": provenance}
        dump(args.output/"report.json", report)
        print(json.dumps({"stage": "finished", "report": report}), flush=True)
        return
    policy, log_alpha, optimizers = build_policy(device, config)
    train_env = make_env(args.input_dir, args.output/"training_env_outputs", train=True)
    vector = DummyVectorEnv([lambda: train_env])
    vector.seed(args.seed)
    buffer = VectorReplayBuffer(args.buffer_size, 1)
    collector = Collector(policy, vector, buffer, exploration_noise=True)
    warmup_result = collector.collect(n_step=args.warmup, random=True)
    start = time.monotonic()
    collected = 0
    updates = 0
    history = []
    best = None
    best_state = None
    best_skeleton = None
    best_log_alpha = None
    next_eval = min(args.eval_every, args.steps)
    latest_losses = {}
    wall_start = time.monotonic()
    try:
        while collected < args.steps:
            policy.train()
            result = collector.collect(n_step=min(4, args.steps-collected))
            n = int(result["n/st"])
            collected += n
            for _ in range(n):
                latest_losses = policy.update(args.batch_size, buffer)
                if not all(np.isfinite(value) for value in latest_losses.values()):
                    raise RuntimeError(f"Nonfinite loss: {latest_losses}")
                updates += 1
            if collected >= next_eval or collected == args.steps:
                elapsed = time.monotonic()-wall_start
                policy.eval()
                # Separate environments prevent baseline or policy comparisons from EMA leakage.
                learned_skeleton = train_env.unwrapped.skel_list.copy()
                coarse_metrics, _ = evaluate(args.input_dir, args.output/"evaluation_tmp_fresh",
                                             policy, device, seed=args.seed)
                ema_metrics, _ = evaluate(args.input_dir, args.output/"evaluation_tmp_ema",
                                          policy, device, skeleton=learned_skeleton, seed=args.seed)
                entry = {"steps": collected, "updates": updates, "wall_seconds": elapsed,
                         "collected_steps_per_wall_second": collected/elapsed,
                         "policy_on_fresh_coarse": coarse_metrics,
                         "policy_with_training_ema": ema_metrics,
                         "losses": latest_losses}
                history.append(entry)
                for label, value, geometry in [("fresh_coarse", coarse_metrics, coarse_skeleton),
                                                ("training_ema", ema_metrics, learned_skeleton)]:
                    if best is None or value["iou"] > best["metrics"]["iou"]:
                        best = {"variant": label, "steps": collected, "metrics": value}
                        best_state = {key: tensor.detach().cpu().clone()
                                      for key, tensor in policy.state_dict().items()}
                        best_skeleton = geometry.copy()
                        best_log_alpha = log_alpha.detach().cpu().clone()
                dump(args.output/"history.json", history)
                print(json.dumps({"stage": "evaluation", **entry, "best": best}), flush=True)
                next_eval += args.eval_every
                if time.monotonic()-start >= args.max_training_seconds:
                    print("Reached bounded training wall time", flush=True)
                    break
        final_state = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
        torch.save({"policy_state_dict": final_state,
                    "log_alpha": log_alpha.detach().cpu(), "config": config,
                    "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
                    "steps": collected, "provenance": provenance}, args.output/"final_policy.pth")
        assert best_state is not None
        checkpoint = args.output/"best_policy.pth"
        torch.save({"policy_state_dict": best_state, "log_alpha": best_log_alpha,
                    "config": config, "steps": best["steps"], "selected": best,
                    "provenance": provenance}, checkpoint)
        np.save(args.output/"best_skeleton_rc.npy", best_skeleton)
        # Explicitly load from disk into a NEW policy instance before final reported export.
        loaded, loaded_log_alpha, _ = build_policy(device, config)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        loaded.load_state_dict(saved["policy_state_dict"], strict=True)
        loaded_log_alpha.data.copy_(saved["log_alpha"])
        loaded._alpha = loaded_log_alpha.detach().exp()
        loaded.eval()
        final_provenance = dict(provenance, policy_checkpoint=str(checkpoint.resolve()),
                                policy_sha256=digest(checkpoint), selected=best)
        refined, _ = evaluate(args.input_dir, args.output/"refined", loaded, device,
                              skeleton=best_skeleton, seed=args.seed, source_kind="refined",
                              save=True, provenance=final_provenance)
        assert abs(refined["iou"]-best["metrics"]["iou"]) < 1e-12
        report = {"status": "passed", "device": device,
                  "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                  "scope": "Per-glyph optimization; best evaluation selected on the same glyph, not held-out generalization.",
                  "baseline": baseline, "refined": refined,
                  "iou_change": refined["iou"]-baseline["iou"],
                  "improved_iou": refined["iou"] > baseline["iou"],
                  "selected": best, "collected_training_steps": collected,
                  "random_warmup_steps": int(warmup_result["n/st"]), "gradient_updates": updates,
                  "wall_seconds_including_evaluations": time.monotonic()-wall_start,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
                  "checkpoint_roundtrip_metrics_identical": True,
                  "baseline_export": str((args.output/"baseline/trajectory.npz").resolve()),
                  "refined_export": str((args.output/"refined/trajectory.npz").resolve()),
                  "policy_checkpoint": str(checkpoint.resolve()),
                  "provenance": provenance}
        dump(args.output/"report.json", report)
        print(json.dumps({"stage": "finished", "report": report}), flush=True)
    finally:
        vector.close()


if __name__ == "__main__":
    main()
