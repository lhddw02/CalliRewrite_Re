"""Read official CalliRewrite coarse NPZs without TensorFlow or PyTorch.

Raw model rows are [pen_up, a_row, a_col, d_row, d_col, width_end, next_scale].
They are local quadratic parameters, not global control points or XYZ poses.
The decoder follows the released rasterizer's rounded 2*128-pixel control
points, OpenCV resize pixel centers, and inverse TensorFlow crop-and-resize
mapping. ``curves_px`` is [pen_up, start_row, start_col, control_row,
control_col, end_row, end_col], in the padded image's pixel coordinates.

``radius_norm`` is the area-equivalent image radius of the rasterizer's circle:
  r_hi(t) = floor((1-t)*(1+floor(w_start*128/2))
                 +t*(1+floor(w_end*128/2)))
  radius_norm = 0.5*r_hi(t)*sqrt(paste_scale_x*paste_scale_y)/image_size.
It is a rendering-space radius, not a measured force, pressure, or SI length.
Quantization and the rasterizer's additive radius are deliberately retained.

Only ``rasterize_coarse`` needs OpenCV; loading/sampling uses NumPy alone.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np

RASTER_SIZE = 128
MIN_WINDOW_SIZE = 32.0


def _quadratic(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    return (1-t)**2*points[0] + 2*t*(1-t)*points[1] + t*t*points[2]


def _paste_transform(cursor_uv: np.ndarray, window: float) -> dict:
    """The inverse affine of DiffPastingV3.image_pasting_sampling_v3."""
    r = float(RASTER_SIZE)
    low = np.floor(cursor_uv-window/2.0)
    high = np.ceil(cursor_uv+window/2.0)
    extent = high-low
    crop_cursor = ((low+high)/2.0-(cursor_uv-window/2.0))/window*r
    crop_window = r*extent/window
    crop_start = crop_cursor-(crop_window-1.0)/2.0
    crop_end = crop_start+crop_window-1.0
    scale_uv = (extent-1.0)/(crop_end-crop_start)
    return {"low_uv": low.astype(np.int64), "extent_uv": extent.astype(np.int64),
            "crop_start_uv": crop_start, "crop_end_uv": crop_end,
            "scale_uv": scale_uv}


def _sample_segment(segment: dict, image_size: int, spacing: float) -> dict:
    points = segment["points_rc"]
    # ||B'(t)|| <= 2*max(||Q-P0||, ||P2-Q||), so adjacent samples stay
    # within the requested spatial step even with uneven control handles.
    length_bound = 2*max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[1]))
    count = max(2, int(math.ceil(length_bound/spacing))+1)
    t = np.linspace(0.0, 1.0, count)
    rc = _quadratic(points, t)
    # Explicit endpoints protect even degenerate/short curves from resampling loss.
    rc[0], rc[-1] = points[0], points[-1]
    hi_radii = segment["highres_radii"]
    radius = np.floor((1.0-t)*hi_radii[0]+t*hi_radii[1])
    radius *= .5*float(np.sqrt(np.prod(segment["paste"]["scale_uv"]))) / image_size
    if segment["pen_up"]:
        radius[:] = 0.0
    return {"xy_uv": rc[:, ::-1]/float(image_size), "radius_norm": radius,
            "curve_index": segment["curve_index"], "round_index": segment["round_index"],
            "pen_up": bool(segment["pen_up"])}


def load_coarse(path: str | Path, *, sample_spacing_px: float = 1.0) -> dict:
    """Load, decode and sample every inference round of an official coarse NPZ.

    ``strokes`` groups consecutive pen-down curves; ``moves`` retains each
    pen-up curve. ``segments`` preserves all source rows and round boundaries.
    Each stroke has ``xy_uv`` (normalized column,row), ``radius_norm`` and an
    integer ``stroke_id``. Global geometry is not clipped: the renderer clips
    pixels and the next cursor, while a curve itself may extend past the image.
    Consumers must choose a safe robot workspace before replaying it.
    """
    if not np.isfinite(sample_spacing_px) or sample_spacing_px <= 0:
        raise ValueError("sample_spacing_px must be positive and finite")
    path = Path(path)
    required = {"strokes_data", "init_cursors", "image_size", "round_length", "init_width"}
    archive = np.load(path, allow_pickle=False)
    if not hasattr(archive, "files"):
        raise ValueError("load_coarse expects a local-parameter NPZ, not a global-curve NPY")
    with archive:
        missing = required-set(archive.files)
        if missing:
            raise ValueError("Missing coarse NPZ keys: "+", ".join(sorted(missing)))
        raw = np.array(archive["strokes_data"], dtype=np.float64, copy=True)
        cursors = np.array(archive["init_cursors"], dtype=np.float64, copy=True)
        rounds_raw = np.atleast_1d(archive["round_length"]).astype(np.float64)
        image_raw = np.asarray(archive["image_size"])
        width_raw = np.asarray(archive["init_width"])
    if raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError("strokes_data must be a finite (M,7) array")
    if image_raw.size != 1 or width_raw.size != 1:
        raise ValueError("image_size and init_width must be scalars")
    n_float = float(image_raw.reshape(-1)[0])
    if not np.isfinite(n_float) or n_float < 2 or n_float != int(n_float):
        raise ValueError("image_size must be an integer >= 2")
    image_size = int(n_float)
    init_width = float(width_raw.reshape(-1)[0])
    if not np.isfinite(init_width) or init_width < 0:
        raise ValueError("init_width must be finite and nonnegative")
    if rounds_raw.ndim != 1 or not np.isfinite(rounds_raw).all() or np.any(rounds_raw < 0):
        raise ValueError("round_length must contain nonnegative integers")
    rounds = rounds_raw.astype(np.int64)
    if not np.array_equal(rounds, rounds_raw) or int(rounds.sum()) != len(raw):
        raise ValueError("round_length must contain integers summing to the number of rows")
    if cursors.ndim == 1 and cursors.size == 2:
        cursors = cursors.reshape(1, 2)
    if cursors.shape != (len(rounds), 2) or not np.isfinite(cursors).all():
        raise ValueError("init_cursors must be finite (number_of_rounds,2)")
    if np.any((raw[:, 0] != 0) & (raw[:, 0] != 1)):
        raise ValueError("pen state must be 0 (draw) or 1 (move)")
    if np.any(raw[:, 5:] < 0):
        raise ValueError("width and next_scale must be nonnegative")

    segments = []
    curves = []
    offset = 0
    for round_index, round_length in enumerate(rounds):
        cursor_uv = cursors[round_index]*image_size
        previous_width, previous_scale, previous_window = init_width, 1.0, float(RASTER_SIZE)
        for inner_index in range(int(round_length)):
            index = offset+inner_index
            row = raw[index]
            pen_up = int(row[0])
            a_rc, d_rc, end_width, next_scale = row[1:3], row[3:5], row[5], row[6]
            window = float(np.clip(previous_scale*previous_window, MIN_WINDOW_SIZE, image_size))
            start_local = np.array([.5, .5])
            end_local = (d_rc+1.0)/2.0
            control_local = start_local+(end_local-start_local)*a_rc
            local = np.array([start_local, control_local, end_local])
            # Same half-up control-point conversion as utils.normal(..., 256).
            highres_rc = np.floor(local*(2*RASTER_SIZE-1)+.5)
            texture_rc = (highres_rc-.5)/2.0  # OpenCV's 2:1 resize pixel-center convention
            paste = _paste_transform(cursor_uv, window)
            points_rc = (paste["low_uv"][::-1]
                         +(texture_rc-paste["crop_start_uv"][::-1])*paste["scale_uv"][::-1])
            radii_hi = np.array([int(1+previous_width*RASTER_SIZE//2),
                                 int(1+end_width*RASTER_SIZE//2)], dtype=np.int64)
            segments.append({"curve_index": index, "round_index": round_index,
                             "round_start": inner_index == 0, "pen_up": pen_up,
                             "points_rc": points_rc, "highres_points_rc": highres_rc,
                             "highres_radii": radii_hi, "window_size_px": window,
                             "width_start": previous_width, "width_end": end_width,
                             "paste": paste})
            curves.append(np.r_[pen_up, points_rc.reshape(-1)])
            next_window = float(np.clip(next_scale*window, MIN_WINDOW_SIZE, image_size))
            previous_width = end_width*window/next_window
            previous_scale, previous_window = next_scale, window
            cursor_uv = np.clip(cursor_uv+d_rc[::-1]*(window/2.0), 0.0, image_size-1.0)
        offset += int(round_length)

    strokes, moves, group = [], [], []
    def flush_group():
        if not group:
            return
        strokes.append({"stroke_id": len(strokes), "round_index": group[0]["round_index"],
                        "xy_uv": np.concatenate([s["xy_uv"] for s in group]),
                        "radius_norm": np.concatenate([s["radius_norm"] for s in group]),
                        "curve_indices": [s["curve_index"] for s in group]})
        group.clear()
    for segment in segments:
        sampled = _sample_segment(segment, image_size, sample_spacing_px)
        segment["sampled"] = sampled
        if segment["round_start"]:
            flush_group()
        if segment["pen_up"]:
            flush_group()
            moves.append(sampled)
        else:
            group.append(sampled)
    flush_group()
    return {"source": str(path.resolve()), "image_size": image_size,
            "curves_px": np.array(curves, dtype=np.float64).reshape(-1, 7),
            "strokes": strokes, "moves": moves, "segments": segments,
            "round_length": rounds, "init_cursors": cursors,
            "init_width": init_width, "raw_strokes_data": raw,
            "coordinate_order": "curves_px=row,column; xy_uv=column,row normalized by image_size",
            "width_mapping": "Quantized official high-resolution circle radius, divided by two "
                             "for resize, multiplied by area-equivalent inverse crop scale and "
                             "divided by image_size. Rendering radius, not calibrated pressure."}


def _crop_and_resize(patch: np.ndarray, paste: dict) -> np.ndarray:
    """NumPy bilinear equivalent of the source's tf.image.crop_and_resize."""
    width, height = paste["extent_uv"]
    # TF receives float32 boxes normalized by 127, then expands them again.
    start = np.asarray(paste["crop_start_uv"]/(RASTER_SIZE-1), np.float32)*(RASTER_SIZE-1)
    end = np.asarray(paste["crop_end_uv"]/(RASTER_SIZE-1), np.float32)*(RASTER_SIZE-1)
    xs = np.linspace(start[0], end[0], int(width), dtype=np.float32)
    ys = np.linspace(start[1], end[1], int(height), dtype=np.float32)
    valid_x = (xs >= 0) & (xs <= RASTER_SIZE-1)
    valid_y = (ys >= 0) & (ys <= RASTER_SIZE-1)
    xs, ys = np.clip(xs, 0, RASTER_SIZE-1), np.clip(ys, 0, RASTER_SIZE-1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.ceil(xs).astype(int), np.ceil(ys).astype(int)
    dx, dy = xs-x0, ys-y0
    top = patch[y0[:, None], x0]*(1-dx)+patch[y0[:, None], x1]*dx
    bottom = patch[y1[:, None], x0]*(1-dx)+patch[y1[:, None], x1]*dx
    out = top*(1-dy[:, None])+bottom*dy[:, None]
    out[~valid_y] = 0
    out[:, ~valid_x] = 0
    return out


def rasterize_coarse(coarse: dict | str | Path) -> np.ndarray:
    """Recreate official accumulated grayscale ink as float32 [0 background,1 ink].

    This intentionally follows all 100 original circle stamps, integer radii,
    2:1 OpenCV resizing and crop/paste operations, rather than rasterizing the
    sampled path approximation. Requires OpenCV in the extraction environment.
    """
    import cv2

    if not isinstance(coarse, dict):
        coarse = load_coarse(coarse)
    n = coarse["image_size"]
    canvas = np.zeros((n, n), dtype=np.float32)
    for segment in coarse["segments"]:
        if segment["pen_up"]:
            continue
        high = np.zeros((2*RASTER_SIZE, 2*RASTER_SIZE), dtype=np.float32)
        controls = segment["highres_points_rc"]
        z0, z1 = segment["highres_radii"]
        for i in range(100):
            # Source uses i*(1/100), which differs from i/100 at some binary
            # floats; its int-truncated centers can differ by a whole pixel.
            t = i*(1.0/100.0)
            # Keep the released scalar expression/order: replacing a*a with
            # a**2 can move an exact-integer center across int()'s boundary.
            rc = (1-t)*(1-t)*controls[0]+2*t*(1-t)*controls[1]+t*t*controls[2]
            radius = int((1-t)*z0+t*z1)
            cv2.circle(high, (int(rc[1]), int(rc[0])), radius, 1.0, -1)
        # utils.draw returns 1-resize(high), and the caller subtracts again.
        patch = 1.0-(1.0-cv2.resize(high, (RASTER_SIZE, RASTER_SIZE)))
        pasted = _crop_and_resize(patch, segment["paste"])
        x0, y0 = segment["paste"]["low_uv"]
        height, width = pasted.shape
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(n, x0+width), min(n, y0+height)
        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            canvas[dst_y0:dst_y1, dst_x0:dst_x1] += pasted[
                dst_y0-y0:dst_y1-y0, dst_x0-x0:dst_x1-x0]
    return np.clip(canvas, 0.0, 1.0)


def rasterize_sampled(coarse: dict | str | Path, *, supersample: int = 4) -> np.ndarray:
    """Diagnostic raster of decoded sampled XY/radii, without using local rows.

    This approximates continuous circular stamps and is useful to check that
    the exported robot path follows the official image. It is intentionally
    separate from the pixel-faithful ``rasterize_coarse`` implementation.
    """
    import cv2

    if not isinstance(coarse, dict):
        coarse = load_coarse(coarse)
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    n = coarse["image_size"]
    high = np.zeros((n*supersample, n*supersample), dtype=np.float32)
    for stroke in coarse["strokes"]:
        xy = stroke["xy_uv"]*n*supersample
        radii = stroke["radius_norm"]*n*supersample
        for point, radius in zip(xy, radii):
            cv2.circle(high, tuple(np.rint(point).astype(int)),
                       max(1, int(round(float(radius)))), 1.0, -1)
    return cv2.resize(high, (n, n), interpolation=cv2.INTER_AREA)
