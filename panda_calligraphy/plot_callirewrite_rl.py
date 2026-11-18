"""Plot actual native masks and optimization history from verify_integration_rl.py."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/callirewrite-integration-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/callirewrite-integration-fonts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
report = json.loads((args.run_dir/"report.json").read_text())
history = json.loads((args.run_dir/"history.json").read_text())
base = np.load(args.run_dir/"baseline/trajectory.npz", allow_pickle=False)
refined = np.load(args.run_dir/"refined/trajectory.npz", allow_pickle=False)
fig = plt.figure(figsize=(12, 7), facecolor="#f5f7fa")
grid = fig.add_gridspec(2, 3, height_ratios=(1.6, 1), left=.055, right=.98,
                       bottom=.145, top=.81, wspace=.1, hspace=.46)
titles = ["Target glyph", f"Zero-action baseline\nIoU {report['baseline']['iou']:.4f}",
          f"Trained candidate ({report['selected']['steps']:,} steps)\nIoU {report['refined']['iou']:.4f}"]
for index, (mask, title) in enumerate(zip(
        [base['target_ink_mask'], base['native_ink_mask'], refined['native_ink_mask']], titles)):
    ax = fig.add_subplot(grid[0, index])
    ax.imshow(~mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
ax = fig.add_subplot(grid[1, :])
steps = [entry['steps'] for entry in history]
ax.plot(steps, [entry['policy_on_fresh_coarse']['iou'] for entry in history],
        '-o', color='#2f6b9a', markersize=3, label='Trained policy / fresh coarse skeleton')
ax.plot(steps, [entry['policy_with_training_ema']['iou'] for entry in history],
        '-o', color='#b46729', markersize=3, label='Trained policy / training EMA skeleton')
ax.axhline(report['baseline']['iou'], color='#58616b', linestyle='--', label='Zero-action baseline')
ax.set(xlabel='Collected training steps (random warmup excluded)', ylabel='Native mask IoU')
ax.grid(alpha=.2)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8, ncol=3, loc='lower left',
           bbox_to_anchor=(.055, .36), framealpha=1, facecolor='white')
ax.spines[['top', 'right']].set_visible(False)
glyph = report['provenance']['glyph_id']
fig.suptitle(f"CalliRewrite native 2D optimization | {glyph}", fontsize=17, weight='bold', x=.055, ha='left')
fig.text(.055, .925, f"Per-glyph optimization, no held-out claim. Best trained candidate IoU change: {report['iou_change']:+.4f}.", fontsize=10)
fig.text(.055, .025, 'Unmodified released reward/raster; independent SAC critics. Native contact exports reconstruct each mask exactly.', fontsize=9, color='#536171')
destination = args.run_dir/'comparison.png'
fig.savefig(destination, dpi=150, facecolor=fig.get_facecolor())
print(destination)
