"""Collect the two completed verification runs into README assets.

Run with the Panda Python environment from any working directory. This only
reads the saved measurements and copies/plots actual outputs; it does not train.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / ".local_setup/verification/integration"
DOCS = ROOT / "panda_calligraphy/docs"


def read(path):
    return json.loads(path.read_text())


def main():
    cases = [("glyph_0", "glyph_0_extended", "panda_coarse", "panda_refined_glyph0"),
             ("glyph_6", "glyph_6", "panda_coarse_glyph6", "panda_refined_glyph6")]
    summary = {"date": "2026-11-17", "scope": "Two-glyph feasibility pilot; same-glyph checkpoint selection, no held-out or hardware claim", "cases": {}}
    figure = Image.new("RGB", (1500, 870), "#f3f5f8")
    draw = ImageDraw.Draw(figure)
    fonts = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(fonts, 19)
    small = ImageFont.truetype(fonts, 15)
    draw.text((25, 15), "CalliRewrite to Panda: measured outputs from the verification run", fill="#14293d", font=font)
    labels = ["Input glyph", "Official coarse output", "Native brush baseline", "Trained policy + EMA", "Panda achieved"]
    for col, label in enumerate(labels):
        draw.text((col*300+18, 58), label, fill="#14293d", font=font)
    for row, (glyph, rl_name, coarse_panda, refined_panda) in enumerate(cases):
        extraction = RESULTS / "coarse/runs" / glyph
        rl = RESULTS / "rl" / rl_name
        panda = RESULTS / refined_panda
        entry = {"extraction": read(extraction / "report.json"), "refinement": read(rl / "report.json"),
                 "panda_coarse": read(RESULTS / coarse_panda / "verification.json"),
                 "panda_refined": read(panda / "verification.json")}
        summary["cases"][glyph] = entry
        paths = [RESULTS / "coarse/rl_input" / glyph / "0.png",
                 extraction / "sampling/__new_train_phase_2/0.png",
                 rl / "baseline/native_rendered.png", rl / "refined/native_rendered.png", panda / "panda_achieved.png"]
        y = 104+row*365
        for col, path in enumerate(paths):
            picture = Image.open(path).convert("RGB").resize((270,270))
            figure.paste(picture, (col*300+15,y))
        report = entry["refinement"]
        text = (f"{glyph}: native overlap {report['baseline']['iou']:.1%} → {report['refined']['iou']:.1%}; "
                f"{report['collected_training_steps']:,} training steps. Panda/native replay overlap: "
                f"{entry['panda_refined']['source_to_achieved']['iou']:.1%}.")
        draw.text((25,y+285), text, fill="#14293d", font=small)
        shutil.copy2(rl / "comparison.png", DOCS / f"callirewrite_rl_{glyph}.png")
    draw.text((25,827), "Coarse and native brush columns use different renderers. Training gains are measured only against the same native brush baseline.", fill="#47586a", font=small)
    figure.save(DOCS / "callirewrite_verification_comparison.png")
    demo = RESULTS / "rl/glyph_6/refined/trajectory.npz"
    copies = {demo: "callirewrite_demo_refined.npz",
              RESULTS / "coarse/rl_input/glyph_6/0.png": "callirewrite_demo_input.png",
              RESULTS / "coarse/runs/glyph_6/sampling/__new_train_phase_2/seq_data/0.npz": "callirewrite_demo_coarse.npz"}
    for source, name in copies.items():
        shutil.copy2(source, DOCS / name)
    live = RESULTS / "panda_refined_glyph6/viser_completed.png"
    offline = RESULTS / "panda_refined_glyph6/panda_achieved.png"
    summary["browser"] = {"refined_glyph6_completed": live.exists(),
        "live_canvas_matches_offline": bool(live.exists() and np.array_equal(np.asarray(Image.open(live)), np.asarray(Image.open(offline)))),
        "paper_frame": "image right +Y_base, image down +X_base; image plane faces +Z",
        "nominal_playback_duration_s": summary["cases"]["glyph_6"]["panda_refined"]["duration_s"]}
    interpolation = RESULTS / "interpolation_check.json"
    if interpolation.exists():
        summary["interpolation"] = read(interpolation)
    summary["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/"panda_calligraphy").glob("*.py") if not p.name.startswith("test_")}
    (DOCS / "callirewrite_verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(DOCS / "callirewrite_verification_comparison.png")


if __name__ == "__main__":
    main()
