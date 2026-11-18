"""Restore cached weights safely, run official extraction, and check decoding.

From the repository root, in the existing coarse environment:
  source .local_setup/activate.sh coarse
  python panda_calligraphy/verify_callirewrite_extraction.py --require-gpu

The default input is bundled test_data/0.png; --input selects another PNG.
--name defaults to a timestamp; a pre-existing named run is never overwritten.
--restore-only verifies cached ZIP hashes and restores missing model files
without importing TensorFlow or running inference. No packages/downloads are
installed. --cache-dir and --output-root can select other local directories.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import os
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".local_setup/verification/integration/coarse"

CACHED_ARCHIVES = {
    "parameters.zip": "23aaee432debfd27e421ac720e1a85cd3dd82db9cb0013c86c16649d0bdbbb3d",
    "pretrain_perceptual_model.zip": "d79a32968e4cff94a99ae63ced566302593f762289d564a0a63c6047c065b22d",
}


def _digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b""):
        digest.update(block)
    return digest.hexdigest()


def restore_cached_checkpoints(cache_dir, destination):
    """Pinned ZIP hashes, safe paths, CRCs, exclusive creation; never overwrite."""
    restored = []
    for name, expected_hash in CACHED_ARCHIVES.items():
        archive = cache_dir / name
        with archive.open("rb") as stream:
            actual_hash = _digest_stream(stream)
        if actual_hash != expected_hash:
            raise ValueError("Cached checkpoint archive SHA-256 mismatch: "+str(archive))
        with zipfile.ZipFile(archive) as package:
            selected = []
            for member in package.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe checkpoint archive member: "+member.filename)
                if member.is_dir():
                    continue
                if name == "parameters.zip":
                    if not relative.parts or relative.parts[0] != "new_train_phase_2":
                        continue
                    output_relative = Path(member.filename)
                else:
                    output_relative = Path("pretrain_perceptual_model") / member.filename
                selected.append((member, destination / output_relative))
            # Validate all selected archive data and existing destination files
            # before writing anything from this archive.
            checked = []
            for member, target in selected:
                with package.open(member) as stream:
                    expected_member_hash = _digest_stream(stream)  # also checks ZIP CRC
                if target.exists():
                    with target.open("rb") as stream:
                        if _digest_stream(stream) != expected_member_hash:
                            raise FileExistsError("Refusing to overwrite different checkpoint file: "+str(target))
                checked.append((member, target, expected_member_hash))
            for member, target, member_hash in checked:
                existed = target.exists()
                if not existed:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024*1024)
                restored.append({"path": str(target), "sha256": member_hash,
                                 "bytes": member.file_size, "already_present": existed})
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "rl_finetune/data/test_data/0.png")
    parser.add_argument("--name", default="glyph_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".local_setup/downloads")
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        raise ValueError("--name must contain only letters, digits, underscores and hyphens")
    restored = restore_cached_checkpoints(args.cache_dir.resolve(), ROOT / "seq_extract/outputs/snapshot")
    if args.restore_only:
        print(json.dumps({"status": "passed", "restored_model_files": restored}, indent=2))
        return
    source = args.input.resolve()
    output_root = args.output_root.resolve()
    run_dir = output_root / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    work = run_dir / "work"
    (work / "outputs").mkdir(parents=True)
    (work / "tools").mkdir()
    (work / "outputs/snapshot").symlink_to(ROOT / "seq_extract/outputs/snapshot", target_is_directory=True)
    input_path = run_dir / "0.png"
    shutil.copy2(source, input_path)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / "seq_extract"), str(ROOT)]
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    import test_vectorization as official
    from panda_calligraphy.callirewrite_io import load_coarse, rasterize_coarse, rasterize_sampled
    devices = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not devices:
        raise RuntimeError("No TensorFlow GPU is available")
    os.chdir(work)
    start = time.monotonic()
    # The released main_testing path, model, lengths, and stop settings are unchanged.
    official.main_testing("", "", str(input_path), str(run_dir / "sampling"),
                          str(ROOT / "seq_extract/outputs/snapshot"), "new_train_phase_2", 1,
                          longer_infer_lens=[500]*10, round_stop_state_num=12,
                          stroke_acc_threshold=.95, draw_seq=False, draw_order=True,
                          state_dependent=False)
    inference_seconds = time.monotonic()-start
    output = run_dir / "sampling/__new_train_phase_2"
    npz_path = output / "seq_data/0.npz"
    decoded = load_coarse(npz_path)
    raster = rasterize_coarse(decoded)
    original_gray = np.asarray(Image.open(output / "0.png").convert("L"), dtype=np.float32)/255.
    original = 1-original_gray
    predicted_gray = np.round((1-raster)*255).astype(np.uint8)
    Image.fromarray(predicted_gray).save(run_dir / "adapter_reconstruction.png")
    # Compare both rasters after the same 8-bit output quantization; classifying
    # unquantized 0.5 against quantized 127/255 gives misleading edge mismatches.
    mask_a, mask_b = original >= .5, predicted_gray <= 127
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(np.logical_and(mask_a, mask_b).sum()/max(1, union))
    differences = np.abs(original-raster)
    sampled_raster = rasterize_sampled(decoded)
    sampled_mask = sampled_raster >= .5
    sampled_union = np.logical_or(mask_a, sampled_mask).sum()
    sampled_iou = float(np.logical_and(mask_a, sampled_mask).sum()/max(1, sampled_union))
    Image.fromarray(np.round((1-sampled_raster)*255).astype(np.uint8)).save(run_dir / "sampled_trajectory_reconstruction.png")
    # Pixel differences can reflect float32 crop arithmetic, but not axis/state errors.
    assert iou >= .995, (iou, float(differences.max()))
    assert float(differences.mean()) < .001, float(differences.mean())
    assert sampled_iou >= .95, ("sampled geometry mismatch", sampled_iou)
    assert sum(decoded["round_length"]) == len(decoded["curves_px"])
    assert len(decoded["curves_px"]) > 0 and decoded["strokes"]
    rl_input = output_root / "rl_input" / args.name
    rl_input.mkdir(parents=True, exist_ok=False)
    shutil.copy2(input_path, rl_input / "0.png")
    np.save(rl_input / "0.npy", decoded["curves_px"])
    arrays = {"curves_px": decoded["curves_px"], "round_length": decoded["round_length"]}
    for stroke in decoded["strokes"]:
        prefix = "stroke_%d_" % stroke["stroke_id"]
        arrays[prefix + "xy_uv"] = stroke["xy_uv"]
        arrays[prefix + "radius_norm"] = stroke["radius_norm"]
    np.savez_compressed(run_dir / "decoded_geometry.npz", **arrays)
    def display_path(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)
    report = {"status": "passed", "input_source": display_path(source),
              "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "method": "Released test_vectorization.main_testing; unmodified inference/rendering functions",
              "model": "new_train_phase_2/p2s-30000", "perceptual_model": "iter_18.tfmodel",
              "device": "GPU:0" if devices else "CPU", "python": sys.version,
              "tensorflow": tf.__version__, "inference_and_render_seconds": inference_seconds,
              "npz": display_path(npz_path), "image_size": decoded["image_size"],
              "rows": len(decoded["curves_px"]), "draw_rows": int((decoded["curves_px"][:, 0] == 0).sum()),
              "move_rows": len(decoded["moves"]), "stroke_groups": len(decoded["strokes"]),
              "round_length": decoded["round_length"].tolist(),
              "renderer_comparison": {"foreground_iou_at_0_5": iou,
                                      "iou_quantization": "Both images quantized to 8-bit grayscale before thresholding at foreground >= 0.5",
                                      "mean_absolute_error": float(differences.mean()),
                                      "max_absolute_error": float(differences.max()),
                                      "same_8bit_pixels_fraction": float((predicted_gray == np.round(original_gray*255).astype(np.uint8)).mean()),
                                      "sampled_xy_radius_iou_at_0_5": sampled_iou},
              "rl_pair_dir": display_path(rl_input), "width_mapping": decoded["width_mapping"],
              "restored_model_files": restored}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("INTEGRATION_REPORT", json.dumps(report))


if __name__ == "__main__":
    main()
