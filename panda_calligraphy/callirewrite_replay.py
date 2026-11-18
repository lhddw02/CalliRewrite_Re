"""Verify and replay real CalliRewrite outputs with a URDF-consistent Panda.

Example (from the repository root)::
    panda_calligraphy/venv/bin/python panda_calligraphy/callirewrite_replay.py \
        --input panda_calligraphy/docs/callirewrite_demo_refined.npz \
        --target panda_calligraphy/docs/callirewrite_demo_input.png --serve

The saved input is the interface between the extraction/RL and viewer environments.
No neural-network libraries are imported here. Native RL raster conventions are
preserved, including the released normalized-radius disk behavior. The tuft is a
geometric illustration; the radius/compression conversion is a virtual calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
import queue

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, binary_erosion

from panda_replay_kinematics import PandaReplayKinematics

PAPER_Z = 0.002
PAPER_SIDE = 0.36
PAPER_WXYZ = (0.0, 2**-.5, 2**-.5, 0.0)
RADIUS_COMPRESSION_GAIN = 2.0


def uv_to_base(uv, side=0.18):
    """Image columns right -> +Y; image rows down -> +X, in Panda metres.

    This reverses image handedness onto the upward-facing XY paper plane.
    Viewed from +X toward the robot, the glyph is upright rather than mirrored.
    """
    uv = np.asarray(uv, dtype=float)
    return np.stack((0.55 + side * (uv[..., 1] - 0.5),
                     side * (uv[..., 0] - 0.5)), axis=-1)


def base_to_uv(xy, side=0.18):
    xy = np.asarray(xy, dtype=float)
    return np.stack((xy[..., 1] / side + 0.5,
                     0.5 + (xy[..., 0] - 0.55) / side), axis=-1)


def read_source(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "strokes_data" in data:
            from callirewrite_io import load_coarse
            decoded = load_coarse(path)
            strokes = decoded["strokes"]
            if not strokes:
                raise ValueError("The extraction contains no drawing strokes")
            source = dict(
                xy_uv=np.concatenate([s["xy_uv"] for s in strokes]),
                radius_norm=np.concatenate([s["radius_norm"] for s in strokes]),
                stroke_id=np.concatenate([np.full(len(s["xy_uv"]), i) for i, s in enumerate(strokes)]),
                metadata={"source_kind": "coarse", "footprint": "coarse_circle",
                          "width_mapping": decoded.get("width_mapping", "official local-window width")},
            )
        else:
            source = {k: data[k].copy() for k in data.files}
            raw = source.pop("metadata", source.pop("metadata_json", "{}"))
            source["metadata"] = json.loads(str(np.asarray(raw).item()))
    for name in ("xy_uv", "radius_norm", "stroke_id"):
        if name not in source:
            raise ValueError(f"Missing trajectory field: {name}")
    n = len(source["xy_uv"])
    if n == 0 or source["xy_uv"].shape != (n, 2):
        raise ValueError("xy_uv must be a non-empty Nx2 image-coordinate array")
    for name in ("radius_norm", "stroke_id"):
        if source[name].shape != (n,):
            raise ValueError(f"{name} must have shape ({n},)")
    for name in ("xy_uv", "radius_norm", "stroke_id"):
        if not np.all(np.isfinite(source[name])):
            raise ValueError(f"Non-finite values in {name}")
    if np.any(source["radius_norm"] < 0):
        raise ValueError("Negative brush radius")
    ids = source["stroke_id"]
    if np.any(ids != np.floor(ids)) or np.any(np.diff(ids) < 0):
        raise ValueError("stroke_id must contain ordered integer stroke groups")
    source["metadata"].update(source_file=str(path.resolve()),
                              source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source["native"] = source["metadata"].get("source_kind") != "coarse"
    if source["native"]:
        for name in ("length_norm", "heading_deg"):
            if name not in source or source[name].shape != (n,) or not np.all(np.isfinite(source[name])):
                raise ValueError(f"Native brush exports require finite {name} of shape ({n},)")
        if np.any(source["length_norm"] < 0):
            raise ValueError("Negative brush length")
    return source


def footprint_points(uv, radius, length, heading):
    """Released Writing_Brush.calc_four_points, expressed in (column,row)."""
    if length <= 0:
        return np.repeat(np.asarray(uv)[None], 4, axis=0)
    rad = np.deg2rad(heading)
    phi = np.arccos(np.clip(radius / length, -1.0, 1.0))
    direction = lambda a: np.array([np.cos(a), -np.sin(a)])
    return np.array([uv, uv + radius * direction(rad-phi),
                     uv + length * direction(rad), uv + radius * direction(rad+phi)])


def stamp(mask, uv, radius, points=None, native=False):
    """Integer triangle and strict-interior disk, matching skimage.draw semantics.

    The released RL implementation passes *normalized* radius to disk(), while
    scaling triangle coordinates by canvas size. Preserve that convention here.
    Coarse extraction instead uses its decoded, pixel-scaled circular radius.
    """
    size = len(mask)
    center = np.clip((np.asarray(uv) * size).astype(int), 0, size-1)
    rp = float(radius if native else radius * size)
    if rp > 0:
        lo = np.maximum(0, np.floor(center-rp).astype(int))
        hi = np.minimum(size-1, np.ceil(center+rp).astype(int))
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1] |= ((xx-center[0])**2 + (yy-center[1])**2 < rp**2)
    if points is not None:
        tri = np.clip((np.asarray(points)[1:] * size).astype(int), 0, size-1)
        area = ((tri[1,0]-tri[0,0])*(tri[2,1]-tri[0,1])
                -(tri[1,1]-tri[0,1])*(tri[2,0]-tri[0,0]))
        if area == 0:
            # skimage.draw.polygon emits only the supplied vertices in this case.
            mask[tri[:,1], tri[:,0]] = True
            return
        lo, hi = tri.min(axis=0), tri.max(axis=0)
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        cross = [(b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0])
                 for a, b in zip(tri, np.roll(tri, -1, axis=0))]
        inside = (np.logical_and.reduce([c >= 0 for c in cross]) |
                  np.logical_and.reduce([c <= 0 for c in cross]))
        mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1] |= inside


def mask_rgb(mask):
    rgb = np.full((*mask.shape, 3), (251, 247, 238), dtype=np.uint8)
    rgb[mask] = (18, 18, 20)
    return rgb


def compare_masks(a, b):
    union = int(np.count_nonzero(a | b))
    iou = float(np.count_nonzero(a & b) / union) if union else 1.0
    ea, eb = a & ~binary_erosion(a), b & ~binary_erosion(b)
    if ea.any() and eb.any():
        chamfer = float((distance_transform_edt(~eb)[ea].mean() +
                         distance_transform_edt(~ea)[eb].mean()) / 2)
    else:
        chamfer = None
    return {"iou": iou, "symmetric_contour_distance_px": chamfer,
            "different_pixels": int(np.count_nonzero(a != b)), "resolution": len(a)}


@dataclass
class Replay:
    source: dict
    side: float
    time_s: np.ndarray
    contact_index: np.ndarray
    target_tip: np.ndarray
    kinematics: PandaReplayKinematics
    solution: object
    resolution: int

    def footprint(self, frame, achieved=True):
        i = int(self.contact_index[frame])
        if i < 0:
            return None
        requested_r = float(self.source["radius_norm"][i])
        if achieved:
            tip = self.solution.achieved_tip[frame]
            compression = max(0.0, PAPER_Z-tip[2])
            if compression <= 0:
                return None
            uv = base_to_uv(tip[:2], self.side)
            radius = RADIUS_COMPRESSION_GAIN * compression / self.side
        else:
            uv, radius = self.source["xy_uv"][i], requested_r
        points = None
        if self.source["native"]:
            length = self.source["length_norm"][i]
            if requested_r > 0:
                length *= radius / requested_r
            points = footprint_points(uv, radius, length, self.source["heading_deg"][i])
        return uv, radius, points

    def deposit(self, mask, frame, achieved=True):
        footprint = self.footprint(frame, achieved)
        if footprint is not None:
            stamp(mask, *footprint, native=self.source["native"])


def prepare(source, side=0.18, writing_speed=0.025, resolution=300):
    if not 0 < side <= 0.26 or not np.isfinite(writing_speed) or writing_speed <= 0 or resolution < 32:
        raise ValueError("Use glyph side in (0,0.26] m, positive speed and resolution >=32")
    if source["native"] and "native_ink_mask" in source and source["native_ink_mask"].shape != (resolution, resolution):
        raise ValueError("Use the exported native canvas resolution for a faithful raster comparison")
    xy = uv_to_base(source["xy_uv"], side)
    radii = source["radius_norm"] * side
    compression = radii / RADIUS_COMPRESSION_GAIN
    if compression.max() > 0.012:
        raise ValueError("Requested width needs >12 mm virtual compression; reduce --glyph-side")
    if np.any(np.abs(xy-np.array([0.55, 0.0])) + radii[:, None] > PAPER_SIDE/2-0.005):
        raise ValueError("Trajectory/footprint exceeds the paper; reduce --glyph-side")
    if source["native"]:
        for i, uv in enumerate(source["xy_uv"]):
            points = footprint_points(uv, source["radius_norm"][i], source["length_norm"][i], source["heading_deg"][i])
            if np.any(np.abs(uv_to_base(points, side)-np.array([0.55, 0.0])) > PAPER_SIDE/2-0.005):
                raise ValueError("Directional brush footprint exceeds the paper; reduce --glyph-side")
    nominal = np.column_stack((xy, PAPER_Z-compression))
    lift_z = PAPER_Z + 0.025
    targets, indices, times = [], [], []
    def append(p, index, speed):
        t = 0.0 if not targets else times[-1]+max(0.01, np.linalg.norm(p-targets[-1])/speed)
        targets.append(np.array(p, dtype=float)); indices.append(index); times.append(t)
    def travel(end):
        start = targets[-1]
        count = max(1, int(np.ceil(np.linalg.norm(end-start)/0.002)))
        for p in np.linspace(start, end, count+1)[1:]:
            append(p, -1, 0.08)
    ids = source["stroke_id"]
    for i, p in enumerate(nominal):
        if i == 0 or ids[i] != ids[i-1]:
            above = np.r_[p[:2], lift_z]
            if targets:
                travel(np.r_[targets[-1][:2], lift_z])
                travel(above)
            else:
                append(above, -1, 0.08)
            # Approach to just above the surface without drawing; first source
            # contact event follows. Lift/landing phases never add extra stamps.
            travel(np.r_[p[:2], PAPER_Z+0.001])
        append(p, i, writing_speed)
    travel(np.r_[targets[-1][:2], lift_z])
    kin = PandaReplayKinematics()
    targets = np.asarray(targets)
    solved = kin.solve_path(targets)
    if not solved.success:
        bad = np.flatnonzero(~solved.converged)
        raise ValueError(f"Panda IK failed at {len(bad)} samples; first={bad[:5].tolist()}")
    # Stretch timing to honor each URDF joint velocity bound, including travel.
    durations = np.maximum(np.diff(times), np.max(np.abs(np.diff(solved.q, axis=0)) / kin.velocity_limits, axis=1)*1.05)
    times = np.r_[0.0, np.cumsum(durations)]
    return Replay(source, side, times, np.array(indices), targets, kin, solved, resolution)


def verify(replay, output, target=None):
    output.mkdir(parents=True, exist_ok=True)
    ideal = np.zeros((replay.resolution,)*2, bool)
    achieved = ideal.copy()
    pen_up_ink = 0
    achieved_contacts = 0
    for frame in range(len(replay.time_s)):
        before = np.count_nonzero(achieved) if replay.contact_index[frame] < 0 else None
        replay.deposit(ideal, frame, achieved=False)
        replay.deposit(achieved, frame)
        if before is not None:
            pen_up_ink += int(np.count_nonzero(achieved) != before)
        elif replay.footprint(frame) is not None:
            achieved_contacts += 1
    Image.fromarray(mask_rgb(ideal)).save(output / "source_reconstruction.png")
    Image.fromarray(mask_rgb(achieved)).save(output / "panda_achieved.png")
    source = replay.source
    q = replay.solution.q
    report = {"source": source["metadata"],
              "scope": "kinematic replay; no dynamics or mesh collision validation",
              "contact_samples": int(np.sum(replay.contact_index >= 0)),
              "stroke_groups": int(len(np.unique(source["stroke_id"]))),
              "trajectory_samples": len(q), "duration_s": float(replay.time_s[-1]),
              "glyph_side_m": replay.side, "paper_side_m": PAPER_SIDE,
              "image_to_base": "X=.55+side*(v-.5), Y=side*(u-.5), Z=.002-compression",
              "virtual_calibration": "radius_m = 2 * compression_m; illustrative, not physical calibration",
              "max_position_error_m": float(replay.solution.position_error_m.max()),
              "max_orientation_error_rad": float(replay.solution.orientation_error_rad.max()),
              "minimum_joint_limit_margin_rad": float(np.min(np.minimum(q-replay.kinematics.joint_limits[:, 0], replay.kinematics.joint_limits[:, 1]-q))),
              "max_joint_velocity_ratio": float(np.max(np.abs(np.diff(q, axis=0))/np.diff(replay.time_s)[:, None]/replay.kinematics.velocity_limits)),
              "minimum_ferrule_clearance_m": float(0.045+np.min(replay.solution.achieved_tip[:, 2])-PAPER_Z),
              "achieved_contact_samples": achieved_contacts,
              "pen_up_ink_samples": pen_up_ink,
              "source_to_achieved": compare_masks(ideal, achieved),
              "urdf_sha256": replay.kinematics.urdf_sha256,
              "raster_convention": "native integer triangle plus normalized-radius disk" if source["native"] else "approximate sampled coarse circles; exact extraction raster is evaluated separately"}
    expected_contacts = int(np.count_nonzero(source["radius_norm"] > 0))
    checks = {
        "all_ik_samples_converged": bool(np.all(replay.solution.converged)),
        "finite_solution": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(replay.solution.achieved_tip))),
        "position_error_below_20_um": report["max_position_error_m"] < 2e-5,
        "orientation_error_below_0_01_rad": report["max_orientation_error_rad"] < .01,
        "inside_joint_limits": report["minimum_joint_limit_margin_rad"] >= 0,
        "within_joint_velocity_limits": report["max_joint_velocity_ratio"] <= 1.0+1e-9,
        "rigid_ferrule_above_paper": report["minimum_ferrule_clearance_m"] > 0,
        "all_expected_contacts_achieved": achieved_contacts == expected_contacts,
        "no_ink_during_pen_up": pen_up_ink == 0,
        "nonempty_drawing": bool(achieved.any()),
        "source_achieved_iou_at_least_0_98": report["source_to_achieved"]["iou"] >= .98,
    }
    if source["native"] and "native_ink_mask" in source:
        report["native_to_source"] = compare_masks(np.asarray(source["native_ink_mask"], bool), ideal)
        checks["export_matches_official_native_raster"] = report["native_to_source"]["different_pixels"] == 0
    report["checks"] = checks
    report["status"] = "passed" if all(checks.values()) else "failed"
    panels = [("Source replay", mask_rgb(ideal)), ("Panda achieved", mask_rgb(achieved))]
    if target is not None:
        im = Image.open(target).convert("L").resize((replay.resolution,)*2, Image.Resampling.BILINEAR)
        target_mask = np.asarray(im) < 128
        report["target_to_source"] = compare_masks(target_mask, ideal)
        report["target_to_achieved"] = compare_masks(target_mask, achieved)
        panels.insert(0, ("Input glyph", np.repeat(np.array(im)[..., None], 3, axis=-1)))
    cell = 360
    figure = Image.new("RGB", (cell*len(panels), 420), "#f2f4f7")
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    except OSError:
        font = ImageFont.load_default()
    for j, (label, rgb) in enumerate(panels):
        draw.text((j*cell+25, 22), label, fill="#172a3a", font=font)
        figure.paste(Image.fromarray(rgb).resize((320, 320)), (j*cell+20, 67))
    figure.save(output / "comparison.png")
    np.savez_compressed(output / "panda_trajectory.npz", time_s=replay.time_s,
                        contact_index=replay.contact_index, q=q,
                        target_tip=replay.target_tip, achieved_tip=replay.solution.achieved_tip,
                        metadata=json.dumps(report))
    (output / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def tuft_geometry(hand, rotation, nominal_tip, radius, heading, length, contact,
                  stiffness=1.0, splay=1.0):
    """Cubic-spine visual with a directional contact ellipse; no material solver."""
    root = hand + 0.18*rotation[:, 2]
    end = nominal_tip.copy()
    if contact:
        end[2] = PAPER_Z
    direction = np.array([-np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading)), 0.0])
    controls = np.array([root, root+0.015*rotation[:, 2],
                         end-(0.01*direction/stiffness if contact else 0.015*rotation[:, 2]), end])
    vertices = []
    rings, sides = 12, 20
    lateral = np.array([-direction[1], direction[0], 0.0])
    for t in np.linspace(0, 1, rings):
        center = ((1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]
                  +3*(1-t)*t*t*controls[2]+t**3*controls[3])
        blend = t**3 if contact else 0
        base_radius = 0.007*(1-t)**0.7
        for a in np.linspace(0, 2*np.pi, sides, endpoint=False):
            off = (base_radius+blend*radius*splay)*np.sin(a)*lateral
            off += (base_radius+blend*max(radius, length/2)*splay)*np.cos(a)*direction
            p = center+off
            if contact:
                p[2] = max(PAPER_Z, p[2])
            vertices.append(p)
    vertices = np.array(vertices, dtype=np.float32)
    faces = []
    for r in range(rings-1):
        for a in range(sides):
            b = (a+1) % sides
            faces.extend([[r*sides+a,r*sides+b,(r+1)*sides+b], [r*sides+a,(r+1)*sides+b,(r+1)*sides+a]])
    fibers = np.array([[vertices[r*sides+a], vertices[(r+1)*sides+a]]
                       for a in range(0, sides, 2) for r in range(rings-1)])
    return vertices, np.array(faces), fibers


def serve(replay, output, target, port=8080, autoplay=False):
    import viser
    import viser.transforms as tf
    from viser.extras import ViserUrdf
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    from replay_studio import (NATIVE_MODE, STUDIO_MODE, ReplaySettings,
                               ReplayPlayback, PreviewCanvas, save_preview_settings)
    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.initial_camera.position = (1.25, -0.95, 1.05)
    server.initial_camera.look_at = (0.45, 0.0, 0.16)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.scene.add_grid("/floor", width=2, height=2, position=(0,0,-0.42))
    server.scene.add_frame("/panda_base", axes_length=0.10, axes_radius=0.003)
    server.scene.add_box("/table", dimensions=(0.62,0.72,0.04), position=(0.55,0,-0.02), color=(72,51,41))
    urdf = load_robot_description("panda_description", load_meshes=True, build_scene_graph=True)
    robot = ViserUrdf(server, urdf, root_node_name="/panda")
    names = list(robot.get_actuated_joint_limits())
    settings = ReplaySettings()
    playback = ReplayPlayback(replay, autoplay)
    canvas = PreviewCanvas(replay)
    rgb, _ = canvas.update(playback.frame, settings)
    def paper_image(image):
        border = max(1, int(round(replay.resolution*(PAPER_SIDE/replay.side-1)/2)))
        return np.pad(image, ((border,border),(border,border),(0,0)), constant_values=248)
    # Viser images internally flip around local X. This rotation maps image
    # columns to +Y and rows to +X, with the image facing upward.
    paper = server.scene.add_image("/paper", paper_image(rgb), PAPER_SIDE, PAPER_SIDE,
                                   position=(0.55,0,PAPER_Z), wxyz=PAPER_WXYZ, format="png")
    guide_points = []
    for offset in np.linspace(-replay.side/2, replay.side/2, 5):
        guide_points += [[[.55+offset,-replay.side/2,PAPER_Z+.0002], [.55+offset,replay.side/2,PAPER_Z+.0002]],
                         [[.55-replay.side/2,offset,PAPER_Z+.0002], [.55+replay.side/2,offset,PAPER_Z+.0002]]]
    guides = server.scene.add_line_segments("/paper_guides", points=np.asarray(guide_points),
                colors=(187,105,105), thickness=.8, thickness_units="screen", visible=False)
    handle = server.scene.add_cylinder("/brush/handle", radius=.0055, height=.14, color=(168,120,63))
    ferrule = server.scene.add_cylinder("/brush/ferrule", radius=.0072, height=.025, color=(30,30,30))
    commands = queue.SimpleQueue()
    controls = {}
    with server.gui.add_folder("CalliRewrite playback"):
        kind = replay.source["metadata"].get("source_kind", "unknown")
        server.gui.add_markdown(f"**{kind.title()} model output** · {len(np.unique(replay.source['stroke_id']))} stroke groups\n\nImage right = +Y of Panda base; image up = −X.")
        play = server.gui.add_button("Pause" if autoplay else "Play")
        restart = server.gui.add_button("Restart and clear")
        step = server.gui.add_button("Play one stroke")
        controls["speed"] = server.gui.add_slider("Playback speed", min=.25, max=4, step=.25, initial_value=1)
        controls["loop"] = server.gui.add_checkbox("Loop playback", False)
        progress = server.gui.add_slider("Progress", min=0, max=100, step=.1, initial_value=0, disabled=True)
        status = server.gui.add_markdown("Ready — trajectory checks passed.")
    # The mode is outside collapsible folders so it stays easy to find.
    controls["mode"] = server.gui.add_dropdown("Replay mode", (NATIVE_MODE, STUDIO_MODE), initial_value=NATIVE_MODE)
    mode_note = server.gui.add_markdown("**Native faithful:** exact exported ink raster. Ink styling is disabled.")
    with server.gui.add_folder("Ink studio", expand_by_default=False):
        controls["opacity"] = server.gui.add_slider("Ink opacity", min=0, max=1, step=.05, initial_value=1, disabled=True)
        controls["ink_level"] = server.gui.add_slider("Ink level (%)", min=0, max=100, step=5, initial_value=100, disabled=True)
        controls["absorbency"] = server.gui.add_slider("Paper absorbency / bleed", min=0, max=3, step=.1, initial_value=0, disabled=True)
        controls["ink_color"] = server.gui.add_rgb("Ink color", (18,18,20), disabled=True)
        controls["footprint_scale"] = server.gui.add_slider("Footprint scale", min=.25, max=2.5, step=.05, initial_value=1, disabled=True)
        dip = server.gui.add_button("Dip ink (100%)", disabled=True)
        server.gui.add_markdown("Display only. Studio uses pixel-scaled round footprints plus the exported triangles. Settings redraw all accumulated ink at current progress. Ink level is uniform; it is not consumed. Bleed is a soft raster halo, not fluid simulation.")
    with server.gui.add_folder("Brush and scene", expand_by_default=False):
        controls["stiffness"] = server.gui.add_slider("Bristle stiffness (visual)", min=.3, max=3, step=.1, initial_value=1)
        controls["splay"] = server.gui.add_slider("Tuft splay (visual)", min=.25, max=2.5, step=.05, initial_value=1)
        controls["guide_grid"] = server.gui.add_checkbox("Guide grid", False)
        controls["fibers"] = server.gui.add_checkbox("Show fibers", True)
        controls["show_arm"] = server.gui.add_checkbox("Show arm", True)
        overview = server.gui.add_button("Scene overview")
        overhead = server.gui.add_button("Paper view")
        brush_view = server.gui.add_button("Brush view")
        server.gui.add_markdown("Stiffness changes the illustrative Bézier bend; splay changes tuft geometry. Neither changes the source trajectory, IK, compression, or verified ink mask.")
    with server.gui.add_folder("Paper and settings", expand_by_default=False):
        clear = server.gui.add_button("Clear paper")
        defaults = server.gui.add_button("Restore defaults")
        save = server.gui.add_button("Save preview and settings")
        server.gui.add_markdown("Clear paper keeps the current pose and discards earlier ink. Restore defaults keeps playback progress. Preview files are saved separately under output/preview; verification files stay unchanged.")
    with server.gui.add_folder("Drawing comparison", expand_by_default=False):
        if target:
            server.gui.add_image(np.array(Image.open(target).convert("RGB")), label="Input glyph")
        server.gui.add_image(np.array(Image.open(output/"source_reconstruction.png")), label="Source reconstruction")
        live_image = server.gui.add_image(rgb, label="Panda drawing / preview")
    def bind_setting(name, control):
        @control.on_update
        def _(event):
            commands.put(("setting", (name, event.target.value)))
    for name, control in controls.items():
        bind_setting(name, control)
    def bind_button(control, command):
        @control.on_click
        def _(_event):
            commands.put((command, None))
    for control, command in ((play,"play"), (restart,"restart"), (step,"step"), (clear,"clear"),
                             (dip,"dip"), (defaults,"defaults"), (save,"save")):
        bind_button(control, command)
    def set_camera(event, position, look_at, up):
        for client in ([event.client] if event.client else server.get_clients().values()):
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up
    @overview.on_click
    def _(event):
        set_camera(event, (1.25,-.95,1.05), (.45,0,.16), (0,0,1))
    @overhead.on_click
    def _(event):
        set_camera(event, (.55,0,.9), (.55,0,0), (-1,0,0))
    @brush_view.on_click
    def _(event):
        set_camera(event, (.60,-.50,.16), (.55,0,.095), (0,0,1))
    ink_controls = ("opacity", "ink_level", "absorbency", "ink_color", "footprint_scale")
    def sync_controls():
        for name, control in controls.items():
            control.value = getattr(settings, name)
        disabled = settings.mode == NATIVE_MODE
        for name in ink_controls:
            controls[name].disabled = disabled
        dip.disabled = disabled
        mode_note.content = ("**Native faithful:** exact exported ink raster. Ink styling is disabled."
            if disabled else "**Ink studio:** artistic preview. Source motion is unchanged; styled ink is not a verification result.")
    def save_preview(stem):
        destination = output/"preview"
        destination.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(destination/f"{stem}.png")
        save_preview_settings(destination/f"{stem}.json", settings, playback)
    print(f"[READY] CalliRewrite {kind} replay: http://localhost:{server.get_port()}", flush=True)
    last = time.monotonic()
    previous_render = None
    loop_pending = False
    notice, notice_until = "", 0.0
    try:
        while True:
            now = time.monotonic(); dt = min(.25, now-last); last = now
            if loop_pending:
                playback.restart(playing=True)
                loop_pending = False
            save_requested = False
            while not commands.empty():
                command, value = commands.get()
                if command == "setting":
                    name, setting_value = value
                    settings = replace(settings, **{name: setting_value})
                    if name == "mode":
                        sync_controls()
                elif command == "play":
                    playback.toggle()
                elif command == "restart":
                    playback.restart()
                elif command == "step":
                    if not playback.step_stroke():
                        notice, notice_until = "No remaining stroke — use Restart and clear.", now+4
                elif command == "clear":
                    playback.clear()
                    notice, notice_until = "Paper cleared; only future contacts will add ink.", now+4
                elif command == "dip":
                    settings = replace(settings, ink_level=100.0)
                    sync_controls()
                elif command == "defaults":
                    settings = ReplaySettings()
                    sync_controls()
                    notice, notice_until = "Defaults restored; playback progress is unchanged.", now+4
                elif command == "save":
                    save_requested = True
            event = playback.advance(dt, settings)
            frame = playback.frame
            q = replay.solution.q[frame]
            if frame+1 < len(replay.time_s):
                fraction = (playback.time-replay.time_s[frame])/(replay.time_s[frame+1]-replay.time_s[frame])
                q = q*(1-fraction)+replay.solution.q[frame+1]*fraction
            values = dict(zip(replay.kinematics.joint_names, q))
            robot.update_cfg(np.array([values.get(name, .006) for name in names]))
            robot.show_visual = settings.show_arm
            guides.visible = settings.guide_grid
            hand, rotation, tip = replay.kinematics.forward(q)
            quaternion = tf.SO3.from_matrix(rotation).wxyz
            handle.position = hand+.085*rotation[:,2]; handle.wxyz = quaternion
            ferrule.position = hand+.1675*rotation[:,2]; ferrule.wxyz = quaternion
            i = int(np.max(replay.contact_index[:frame+1], initial=0))
            contact = bool(tip[2] < PAPER_Z)
            radius = max(0.0, 2*(PAPER_Z-tip[2])) if contact else 0
            heading = float(replay.source.get("heading_deg", np.zeros(len(replay.source["xy_uv"])))[i])
            length = float(replay.source.get("length_norm", replay.source["radius_norm"])[i])*replay.side
            width_scale = settings.footprint_scale if settings.mode == STUDIO_MODE else 1.0
            vertices, faces, fibers = tuft_geometry(hand, rotation, tip, radius*width_scale, heading,
                length*width_scale, contact, settings.stiffness, settings.splay)
            tuft_color = settings.ink_color if settings.mode == STUDIO_MODE else (24,22,22)
            server.scene.add_mesh_simple("/brush/tuft", vertices=vertices, faces=faces, color=tuft_color, flat_shading=False)
            server.scene.add_line_segments("/brush/fibers", points=fibers, colors=(55,48,42), thickness=.6,
                                           thickness_units="screen", visible=settings.fibers)
            render_key = (frame, playback.paint_start, settings)
            if render_key != previous_render:
                rgb, _ = canvas.update(frame, settings, playback.paint_start)
                paper.image = paper_image(rgb); live_image.image = rgb
                previous_render = render_key
            if save_requested:
                save_preview(f"preview_{time.time_ns()}")
                notice, notice_until = "Preview PNG and settings saved under output/preview.", now+5
            if event == "completed":
                save_preview("last_completed")
                # Preserve the historical exact output only for a complete,
                # faithful run. Customized/cleared canvases stay in preview/.
                if settings.mode == NATIVE_MODE and playback.paint_start == 0:
                    Image.fromarray(rgb).save(output/"viser_completed.png")
                loop_pending = settings.loop
            progress.value = round(100*playback.time/replay.time_s[-1], 1)
            play.label = "Pause" if playback.playing else ("Replay" if playback.time >= replay.time_s[-1] else "Play")
            status.content = (notice if now < notice_until else
                ("**Completed** — preview saved." if playback.time >= replay.time_s[-1] else
                 f"{playback.time:.1f} / {replay.time_s[-1]:.1f} s · {'Brush touching paper' if contact else 'Brush lifted'}"))
            time.sleep(.03)
    except KeyboardInterrupt:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw coarse NPZ or faithful RL rollout NPZ")
    parser.add_argument("--target", type=Path, help="original glyph PNG for comparison")
    parser.add_argument("--output", type=Path, default=Path(".local_setup/verification/integration/panda"))
    parser.add_argument("--glyph-side", type=float, default=.18, help="glyph frame width in metres")
    parser.add_argument("--writing-speed", type=float, default=.025, help="nominal contact speed in m/s")
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--serve", action="store_true", help="open interactive Viser server after verification")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    replay = prepare(read_source(args.input), args.glyph_side, args.writing_speed, args.resolution)
    report = verify(replay, args.output, args.target)
    print(json.dumps({"status": report["status"], "stroke_groups": report["stroke_groups"],
                      "max_position_error_m": report["max_position_error_m"],
                      "source_to_achieved": report["source_to_achieved"],
                      "output": str(args.output.resolve())}), flush=True)
    if report["status"] != "passed":
        raise SystemExit("Replay verification failed; inspect verification.json before playback")
    if args.serve:
        serve(replay, args.output, args.target, args.port, args.autoplay)


if __name__ == "__main__":
    main()
