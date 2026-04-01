"""
Shared inference pipeline for BEV-based segment dose prediction.

This module contains all geometry helpers, CUDA kernels, and the main run()
function. Each model-specific script imports from here and supplies only the
model class and output filename.
"""

import argparse
import contextlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import cupy as cp
import cupyx.scipy.ndimage as cpndi
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------

def read_mha(path: str, option: str = "info"):
    img = sitk.ReadImage(path)
    if option == "info":
        return img.GetSpacing(), img.GetOrigin(), img.GetSize(), img
    if option == "array":
        return sitk.GetArrayFromImage(img)
    raise ValueError("option must be 'info' or 'array'")


def write_mha(path: str, arr: np.ndarray, ref_img: sitk.Image) -> None:
    im = sitk.GetImageFromArray(arr)
    im.SetOrigin(ref_img.GetOrigin())
    im.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(im, path)


def extract_gps(
    mac_file: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Parse a Geant4 MAC file and return (focus_point, direction_x, rot1)."""
    fp = r"/gps/ang/focuspoint\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+mm"
    dx = r"/gps/direction\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
    r1 = r"/gps/pos/rot1\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
    t = open(mac_file, "r").read()
    s = tuple(map(float, re.search(fp, t).groups()))
    x = tuple(map(float, re.search(dx, t).groups()))
    y = tuple(map(float, re.search(r1, t).groups()))
    return s, x, y


# ------------------------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------------------------

def _norm_np(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def _basis_np(dx: Tuple[float, float, float], dy: Tuple[float, float, float]) -> np.ndarray:
    ux  = _norm_np(np.asarray(dx, dtype=np.float32))
    uy0 = _norm_np(np.asarray(dy, dtype=np.float32))
    uz  = _norm_np(np.cross(ux, uy0))
    uy  = _norm_np(np.cross(uz, ux))
    return np.stack([ux, uy, uz], axis=0).astype(np.float32)


def _norm(v: cp.ndarray) -> cp.ndarray:
    return v / cp.linalg.norm(v)


def _basis(dx: cp.ndarray, dy: cp.ndarray):
    ux  = _norm(dx)
    uy0 = _norm(dy)
    uz  = _norm(cp.cross(ux, uy0))
    uy  = _norm(cp.cross(uz, ux))
    return ux.astype(cp.float32), uy.astype(cp.float32), uz.astype(cp.float32)


# ------------------------------------------------------------------------------
# MAC geometry cache
# ------------------------------------------------------------------------------

def build_mac_cache(
    patient: str, sim_root: str, out_dir: str,
    NX: int = 256, NY: int = 200, NZ: int = 200,
) -> str:
    """Pre-compute and cache BEV geometry (rotation matrix, source position) for all segments."""
    dose_dir  = os.path.join(sim_root, patient, "dose_mha")
    setup_dir = os.path.join(sim_root, patient, "setup")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "mac_cache.json")
    if not os.path.isdir(dose_dir):
        raise FileNotFoundError(dose_dir)
    if not os.path.isdir(setup_dir):
        raise FileNotFoundError(setup_dir)

    crop = float(1000.0 - NX)
    off  = [0.0, -(NY // 2) + 0.5, -(NZ // 2) + 0.5]

    segments: Dict[str, Dict[str, List[float]]] = {}
    for f in sorted(os.listdir(dose_dir)):
        m = re.search(r"MU(\d+\.?\d*)_G", f)
        if not m:
            continue
        seg      = f[5:-4]
        mac_file = os.path.join(setup_dir, f"{seg}.mac")
        if not os.path.exists(mac_file):
            continue
        s, dx, dy = extract_gps(mac_file)
        U   = _basis_np(dx, dy)
        src = np.asarray(s, dtype=np.float32) + crop * U[0]
        segments[seg] = {
            "s":   list(map(float, s)),
            "dx":  list(map(float, dx)),
            "dy":  list(map(float, dy)),
            "U":   U.reshape(-1).astype(np.float32).tolist(),
            "src": src.astype(np.float32).tolist(),
            "off": [float(v) for v in off],
            "NX":  int(NX), "NY": int(NY), "NZ": int(NZ),
        }
    if not segments:
        raise RuntimeError("no segments parsed")
    with open(cache_path, "w") as f:
        json.dump(
            {"patient": patient, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "segments": segments},
            f,
        )
    return cache_path


def load_mac_cache(out_dir: str) -> Optional[Dict[str, Dict[str, List[float]]]]:
    p = os.path.join(out_dir, "mac_cache.json")
    if not os.path.exists(p):
        return None
    try:
        obj  = json.load(open(p, "r"))
        segs = obj.get("segments", None)
        return segs if isinstance(segs, dict) and segs else None
    except Exception:
        return None


# ------------------------------------------------------------------------------
# BEV index grid
# ------------------------------------------------------------------------------

def build_bev_index_grid(NX: int, NY: int, NZ: int) -> cp.ndarray:
    """Build a flat (N, 3) grid of BEV voxel centres in mm."""
    x = cp.arange(NX, dtype=cp.float32)
    y = cp.arange(NY, dtype=cp.float32)
    z = cp.arange(NZ, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")
    off   = cp.asarray([0.0, -(NY // 2) + 0.5, -(NZ // 2) + 0.5], cp.float32)
    scale = cp.asarray([2.0, 2.0, 2.0], cp.float32)
    G_lin = (cp.stack((X, Y, Z), -1) + off) * scale
    del X, Y, Z
    cp.cuda.Stream.null.synchronize()
    return G_lin.reshape(-1, 3)


# ------------------------------------------------------------------------------
# ROI utilities
# ------------------------------------------------------------------------------

def cuboid_corners_world(
    nx: int, ny: int, nz: int,
    dx: float, dy: float, dz: float,
    U: cp.ndarray, plane_origin: cp.ndarray, off: cp.ndarray,
) -> cp.ndarray:
    """Return the 8 world-space corners of the BEV cuboid."""
    c = cp.asarray(
        [[0, 0, 0], [nx-1, 0, 0], [0, ny-1, 0], [0, 0, nz-1],
         [nx-1, ny-1, 0], [nx-1, 0, nz-1], [0, ny-1, nz-1], [nx-1, ny-1, nz-1]],
        cp.float32,
    )
    sc = cp.asarray([dx, dy, dz], cp.float32)
    return plane_origin + ((c + off) * sc) @ U


def clamp_box(lo: cp.ndarray, hi: cp.ndarray, shape: Tuple[int, int, int]):
    lo = cp.clip(lo, 0, cp.asarray(shape) - 1)
    hi = cp.clip(hi, 0, cp.asarray(shape) - 1)
    return lo.astype(cp.int32), hi.astype(cp.int32)


# ------------------------------------------------------------------------------
# CUDA projection kernel (bilinear interpolation on the 2-D segment map)
# ------------------------------------------------------------------------------

PROJECTION_KERNEL = cp.ElementwiseKernel(
    in_params=(
        "raw float32 P, raw float32 seg, "
        "float32 sx, float32 sy, float32 sz, "
        "float32 iso, "
        "float32 nx, float32 ny, float32 nz, "
        "float32 ox, float32 oy, float32 oz, "
        "float32 pyx, float32 pyy, float32 pyz, "
        "float32 pzx, float32 pzy, float32 pzz, "
        "float32 dy, float32 dz, "
        "float32 offy, float32 offz, "
        "int32 Ny, int32 Nz, float32 eps"
    ),
    out_params="float32 out",
    operation=r'''
        float px=P[i*3+0], py=P[i*3+1], pz=P[i*3+2];
        float rx=px-sx, ry=py-sy, rz=pz-sz;
        float dp=rx*nx+ry*ny+rz*nz; if (abs(dp)<eps){out=0;return;}
        float t=iso/dp; if (t<=0){out=0;return;}
        float wx=sx+t*rx, wy=sy+t*ry, wz=sz+t*rz;
        float dx_=wx-ox, dy_=wy-oy, dz_=wz-oz;
        float yl=dx_*pyx+dy_*pyy+dz_*pyz;
        float zl=dx_*pzx+dy_*pzy+dz_*pzz;
        float yi=yl/dy - offy, zi=zl/dz - offz;
        if (yi<0||yi>=(float)(Ny-1)||zi<0||zi>=(float)(Nz-1)){out=0;return;}
        int y0=(int)floorf(yi), z0=(int)floorf(zi), y1=y0+1, z1=z0+1;
        float wyf=yi-(float)y0, wzf=zi-(float)z0;
        float v00=seg[y0*Nz+z0], v10=seg[y1*Nz+z0], v01=seg[y0*Nz+z1], v11=seg[y1*Nz+z1];
        float a=(1.0f-wzf)*v00 + wzf*v01;
        float b=(1.0f-wzf)*v10 + wzf*v11;
        out=(1.0f-wyf)*a + wyf*b;
    ''',
    name="proj2D",
)


# ------------------------------------------------------------------------------
# Per-segment BEV preparation + back-projection context
# ------------------------------------------------------------------------------

@dataclass
class BackCtx:
    interp:    int
    map_idx:   cp.ndarray           # shape (3, N) - pre-computed inverse mapping
    roi_shape: Tuple[int, int, int]
    roi_box:   Tuple[int, int, int, int, int, int]


def prepare_bev_inputs(
    patient: str,
    mac_file: str,
    ct_coeff: cp.ndarray,
    ct_shape: Tuple[int, int, int],
    ct_spacing: Tuple[float, float, float],
    ct_origin: Tuple[float, float, float],
    mode: str,
    mac_cache: Optional[Dict[str, Dict[str, List[float]]]],
    NX: int,
    NY: int,
    NZ: int,
    G_lin: cp.ndarray,
    seg_dir: str,
):
    """
    For one beam segment, build the two BEV input volumes (CT and segment
    projection) and pre-compute the inverse mapping needed to back-project
    the predicted dose into patient coordinates.
    """
    times: Dict[str, float] = {}

    ct_spacing = cp.asarray(ct_spacing, cp.float32)
    ct_origin  = cp.asarray(ct_origin,  cp.float32)

    seg_name  = os.path.basename(mac_file)[:-4]
    has_cache = bool(
        mac_cache
        and seg_name in mac_cache
        and isinstance(mac_cache[seg_name].get("U"),   list)
        and isinstance(mac_cache[seg_name].get("src"), list)
    )

    if has_cache:
        rec          = mac_cache[seg_name]
        U            = cp.asarray(np.asarray(rec["U"], dtype=np.float32).reshape(3, 3), cp.float32)
        plane_origin = cp.asarray(np.asarray(rec["src"], dtype=np.float32), cp.float32)
        off          = cp.asarray(
            np.asarray(rec.get("off", [0.0, -(NY // 2) + 0.5, -(NZ // 2) + 0.5]), dtype=np.float32),
            cp.float32,
        )
        ray_origin  = cp.asarray(np.asarray(rec["s"], dtype=np.float32), cp.float32)
        ux, uy, uz  = U[0], U[1], U[2]
    else:
        s_np, dx_np, dy_np = extract_gps(mac_file)
        ray_origin   = cp.asarray(s_np,  cp.float32)
        dx           = cp.asarray(dx_np, cp.float32)
        dy           = cp.asarray(dy_np, cp.float32)
        ux, uy, uz   = _basis(dx, dy)
        U            = cp.stack((ux, uy, uz), 0)
        crop         = cp.float32(1000.0 - float(NX))
        plane_origin = ray_origin + crop * ux
        off          = cp.asarray([0.0, -(NY // 2) + 0.5, -(NZ // 2) + 0.5], cp.float32)

    # World-space coordinates for every BEV voxel centre
    t = time.perf_counter()
    bev_points = G_lin @ U + plane_origin
    cp.cuda.Stream.null.synchronize()
    times["prep.grid"] = time.perf_counter() - t

    # Convert world coordinates to CT voxel indices
    t = time.perf_counter()
    coords = cp.stack(
        (
            (bev_points[:, 0] - ct_origin[0]) / ct_spacing[0],
            (bev_points[:, 1] - ct_origin[1]) / ct_spacing[1],
            (bev_points[:, 2] - ct_origin[2]) / ct_spacing[2],
        ),
        0,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.coords"] = time.perf_counter() - t

    # Resample CT into BEV frame
    interp = 3 if mode == "cubic" else (1 if mode == "linear" else 0)
    t = time.perf_counter()
    bev_ct = cpndi.map_coordinates(
        ct_coeff, coords, order=interp, mode="constant", cval=-1024.0, prefilter=False,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.ct_map"] = time.perf_counter() - t
    bev_ct = cp.clip(bev_ct.reshape(NX, NY, NZ), -1024.0, None).astype(cp.float32, copy=False)

    # Load and resize the binary segment map
    seg_path = os.path.join(seg_dir, f"{seg_name}.bin")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(seg_path)
    t = time.perf_counter()
    seg         = cp.fromfile(seg_path, dtype=cp.int8).reshape(400, 400).astype(cp.float32, copy=False)
    seg_resized = cpndi.zoom(seg, 0.5, order=1).astype(cp.float32, copy=False)
    seg_resized = cp.flip(cp.rot90(seg_resized, 3), axis=0)
    cp.cuda.Stream.null.synchronize()
    times["prep.seg_io_zoom"] = time.perf_counter() - t

    # Project the 2-D segment map onto the 3-D BEV grid via ray casting
    bev_points32 = bev_points.astype(cp.float32, copy=False)
    ray_h   = cp.asnumpy(ray_origin).astype(np.float32)
    plane_h = cp.asnumpy(plane_origin).astype(np.float32)
    ux_h    = cp.asnumpy(ux).astype(np.float32)
    uy_h    = cp.asnumpy(uy).astype(np.float32)
    uz_h    = cp.asnumpy(uz).astype(np.float32)
    off_h   = cp.asnumpy(off).astype(np.float32)

    proj_flat = cp.zeros(bev_points32.shape[0], cp.float32)
    t = time.perf_counter()
    PROJECTION_KERNEL(
        bev_points32, seg_resized,
        *ray_h, np.float32(1000.0),
        *ux_h, *plane_h,
        *uy_h, *uz_h,
        np.float32(2.0), np.float32(2.0),
        off_h[1], off_h[2],
        int(seg_resized.shape[0]), int(seg_resized.shape[1]),
        np.float32(1e-9),
        proj_flat,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.projection"] = time.perf_counter() - t

    seg_proj = proj_flat.reshape(NX, NY, NZ).astype(cp.float32, copy=False)

    # Pre-compute inverse mapping indices for back-projecting dose to patient CT
    t = time.perf_counter()
    dxmm = dymm = dzmm = cp.float32(2.0)

    corners    = cuboid_corners_world(NX, NY, NZ, dxmm, dymm, dzmm, U, plane_origin, off)
    idx        = (corners - ct_origin) / ct_spacing
    lo         = cp.floor(idx.min(0)) - 1.0
    hi         = cp.ceil(idx.max(0))  + 1.0
    lo_i, hi_i = clamp_box(lo, hi, ct_shape)

    xi  = cp.arange(int(lo_i[0].item()), int(hi_i[0].item()) + 1, dtype=cp.int32)
    yi  = cp.arange(int(lo_i[1].item()), int(hi_i[1].item()) + 1, dtype=cp.int32)
    zi  = cp.arange(int(lo_i[2].item()), int(hi_i[2].item()) + 1, dtype=cp.int32)
    nxr, nyr, nzr = xi.size, yi.size, zi.size

    inv_scale = cp.asarray([1.0 / dxmm, 1.0 / dymm, 1.0 / dzmm], cp.float32)
    A  = U.T * inv_scale[:, None]
    C0 = ((ct_origin - plane_origin) @ A) - off
    Cx = ct_spacing[0] * A[0]
    Cy = ct_spacing[1] * A[1]
    Cz = ct_spacing[2] * A[2]

    xi_f, yi_f, zi_f = xi.astype(cp.float32), yi.astype(cp.float32), zi.astype(cp.float32)
    q = (
        C0[:, None, None, None]
        + Cx[:, None, None, None] * xi_f[None, :, None, None]
        + Cy[:, None, None, None] * yi_f[None, None, :, None]
        + Cz[:, None, None, None] * zi_f[None, None, None, :]
    )
    map_idx = q.reshape(3, -1)

    roi_shape = (nxr, nyr, nzr)
    roi_box   = (
        int(lo_i[0]), int(hi_i[0]) + 1,
        int(lo_i[1]), int(hi_i[1]) + 1,
        int(lo_i[2]), int(hi_i[2]) + 1,
    )
    cp.cuda.Stream.null.synchronize()
    times["prep.roi_q"] = time.perf_counter() - t

    ctx = BackCtx(interp=interp, map_idx=map_idx, roi_shape=roi_shape, roi_box=roi_box)
    return bev_ct, seg_proj, ctx, times


def resample_to_ct(pred_bev: cp.ndarray, ctx: BackCtx, dose_sum: cp.ndarray, MU: float) -> float:
    """Back-project a BEV dose prediction into patient space and accumulate MU-weighted dose."""
    t    = time.perf_counter()
    flat = cpndi.map_coordinates(
        pred_bev, ctx.map_idx,
        order=ctx.interp, mode="constant", cval=0.0,
        prefilter=True if ctx.interp > 1 else False,
    )
    roi = cp.maximum(flat.reshape(ctx.roi_shape), 0).astype(cp.float32, copy=False)
    xs, xe, ys, ye, zs, ze = ctx.roi_box
    dose_sum[xs:xe, ys:ye, zs:ze] += roi * cp.float32(MU)
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - t


# ------------------------------------------------------------------------------
# Shared argparser
# ------------------------------------------------------------------------------

def build_argparser(model_name: str) -> argparse.ArgumentParser:
    """Return a pre-configured ArgumentParser for segment-dose inference."""
    parser = argparse.ArgumentParser(
        description=f"Run {model_name} segment-dose inference on a single patient."
    )
    parser.add_argument(
        "--patient", required=True,
        help="Patient ID (e.g. P016_S).",
    )
    parser.add_argument(
        "--sim-root", required=True,
        help="Simulation data root; must contain {patient}/dose_mha/ and {patient}/setup/.",
    )
    parser.add_argument(
        "--seg-dir", required=True,
        help="Directory containing binary segment files ({seg_name}.bin, int8, shape 400x400).",
    )
    parser.add_argument(
        "--ct-root", required=True,
        help="CT root; must contain {patient}/maskedCT_3mm_shifted.mha.",
    )
    parser.add_argument(
        "--model-weights", required=True,
        help="Path to the model weights (.pth file).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for the predicted dose MHA (default: ./output/{patient}).",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="CUDA device string (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of segments per inference batch (default: 8).",
    )
    parser.add_argument(
        "--mode", choices=["linear", "cubic"], default="cubic",
        help="Spatial interpolation mode for CT resampling (default: cubic).",
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Use FP32 instead of FP16 for inference.",
    )
    parser.add_argument(
        "--no-tf32", action="store_true",
        help="Disable TF32 acceleration (default: TF32 enabled).",
    )
    parser.add_argument(
        "--print-details", action="store_true",
        help="Print a per-step timing breakdown after inference.",
    )
    return parser


# ------------------------------------------------------------------------------
# Main inference function
# ------------------------------------------------------------------------------

def run(
    model_cls: Type[nn.Module],
    out_filename: str,
    patient: str,
    sim_root: str,
    seg_dir: str,
    ct_root: str,
    out_dir: str,
    model_weights: str,
    device: str = "cuda:0",
    batch_size: int = 8,
    mode: str = "cubic",
    allow_tf32: bool = True,
    use_fp16: bool = True,
    print_details: bool = False,  # set True to print per-step timing breakdown
) -> Tuple[float, float, float]:
    """
    Run segment-dose inference for one patient using the supplied model class.

    Args:
        model_cls:     PyTorch model class (must accept zero constructor arguments).
        out_filename:  Name of the output MHA file written to out_dir.
        patient:       Patient ID string.
        sim_root:      Root of Monte Carlo simulation data.
        seg_dir:       Directory of binary segment files.
        ct_root:       Root of masked CT volumes.
        out_dir:       Directory where the predicted dose MHA is saved.
        model_weights: Path to the .pth weights file.
        device:        CUDA device string.
        batch_size:    Segments processed per forward pass.
        mode:          Interpolation mode ('cubic' or 'linear').
        allow_tf32:    Enable TF32 for matmul and cuDNN.
        use_fp16:      Run inference in FP16 (AMP + static tensors).
        print_details: Print per-step wall-clock timings.

    Returns:
        (infer_time, preprocess_time, postprocess_time) in seconds.
    """
    NX, NY, NZ = 256, 200, 200
    os.makedirs(out_dir, exist_ok=True)

    dev = torch.device(device)
    torch.cuda.set_device(dev.index or 0)
    cp.cuda.Device(dev.index or 0).use()

    torch.backends.cudnn.benchmark         = True
    torch.backends.cuda.matmul.allow_tf32  = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32        = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    def sync_all():
        torch.cuda.synchronize()
        cp.cuda.Stream.null.synchronize()

    # ===== Initialization (excluded from the three main timing blocks) =====
    init_t0 = time.perf_counter()

    model     = model_cls().to(device).eval()
    amp_dtype = torch.float16 if use_fp16 else None
    state     = torch.load(model_weights, map_location=device)
    model.load_state_dict(state)

    ct_path = os.path.join(ct_root, patient, "maskedCT_3mm_shifted.mha")
    ct_spacing, ct_origin, _, ref_img = read_mha(ct_path, "info")
    ct_arr   = read_mha(ct_path, "array")
    ct_arr   = np.transpose(ct_arr, (2, 1, 0))  # -> (X, Y, Z)
    ct_vol   = cp.asarray(ct_arr, cp.float32)
    ct_coeff = (
        cpndi.spline_filter(ct_vol, order=3, mode="mirror").astype(cp.float32, copy=False)
        if mode == "cubic" else ct_vol
    )

    dose_dir  = os.path.join(sim_root, patient, "dose_mha")
    setup_dir = os.path.join(sim_root, patient, "setup")
    tasks: List[Tuple[str, float]] = []
    for f in sorted(os.listdir(dose_dir)):
        m = re.search(r"MU(\d+\.?\d*)_G", f)
        if not m:
            continue
        mac      = f[5:-4]
        mac_file = os.path.join(setup_dir, f"{mac}.mac")
        if os.path.exists(mac_file):
            tasks.append((mac_file, float(m.group(1))))
    assert tasks, "no segments found"

    mac_cache = load_mac_cache(out_dir)

    def _ok(rec: dict) -> bool:
        return (
            isinstance(rec.get("U"),   list)
            and isinstance(rec.get("src"), list)
            and isinstance(rec.get("off"), list)
        )

    if (mac_cache is None) or (mac_cache and not all(_ok(rec) for rec in mac_cache.values())):
        print("[info] rebuilding mac_cache.json")
        build_mac_cache(patient, sim_root, out_dir, NX=NX, NY=NY, NZ=NZ)
        mac_cache = load_mac_cache(out_dir) or {}

    G_lin = build_bev_index_grid(NX, NY, NZ)

    CT_NORM    = cp.float32(1619.0)
    DOSE_SCALE = cp.float32(453.2444152832031 * 10.0)

    dose_sum = cp.zeros_like(ct_vol, cp.float32)

    dtype_t     = torch.float16 if use_fp16 else torch.float32
    static_ct   = torch.empty((batch_size, NX, NY, NZ), device=device, dtype=dtype_t)
    static_proj = torch.empty((batch_size, NX, NY, NZ), device=device, dtype=dtype_t)
    static_out  = torch.empty((batch_size, NX, NY, NZ), device=device, dtype=dtype_t)

    graph = torch.cuda.CUDAGraph()

    def autocast_if(dtype):
        return torch.amp.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()

    # Warmup + CUDA Graph capture (counted as overhead, not inference time)
    WARMUP = 6
    sync_all()
    with torch.inference_mode():
        for _ in range(WARMUP):
            static_ct.zero_()
            static_proj.zero_()
            with autocast_if(amp_dtype):
                y = model(static_ct, static_proj)
            static_out.copy_(y)
    sync_all()

    with torch.inference_mode():
        sync_all()
        with torch.cuda.graph(graph):
            with autocast_if(amp_dtype):
                _y = model(static_ct, static_proj)
            static_out.copy_(_y)
    sync_all()

    overhead_time = time.perf_counter() - init_t0

    # ===== Accumulators for the three main timing blocks =====
    TOTAL  = defaultdict(float)  # pre / infer / post
    DETAIL = defaultdict(float)  # optional per-step details

    # ===== Main loop (accumulates the three main blocks only) =====
    loop_start = time.perf_counter()
    i = 0
    while i < len(tasks):
        batch = tasks[i : i + batch_size]

        # ---------- Preprocessing: BEV resampling, coord mapping, segment projection, copy to static tensors ----------
        t_pre0 = time.perf_counter()

        bev_list:     List[cp.ndarray] = []
        segproj_list: List[cp.ndarray] = []
        ctx_list:     List[BackCtx]    = []
        mu_list:      List[float]      = []

        t_prep_calls  = 0.0
        t_copy_static = 0.0

        # 1) Generate BEV inputs for each segment in the batch
        for mac_file, MU in batch:
            t1 = time.perf_counter()
            bev_ct, seg_proj, bctx, times = prepare_bev_inputs(
                patient, mac_file, ct_coeff, ct_vol.shape, ct_spacing, ct_origin,
                mode, mac_cache=mac_cache, NX=NX, NY=NY, NZ=NZ, G_lin=G_lin,
                seg_dir=seg_dir,
            )
            t_prep_calls += time.perf_counter() - t1

            bev_norm = (bev_ct * (cp.float32(1.0) / CT_NORM)).astype(cp.float32, copy=False)
            if dtype_t == torch.float16:
                bev_norm = bev_norm.astype(cp.float16, copy=False)
                seg_proj = seg_proj.astype(cp.float16, copy=False)

            bev_list.append(bev_norm)
            segproj_list.append(seg_proj)
            ctx_list.append(bctx)
            mu_list.append(MU)

            if print_details and isinstance(times, dict):
                for k, v in times.items():
                    DETAIL[f"pre.{k}"] += v

        # 2) Copy to static tensors (still counted as preprocessing)
        t2 = time.perf_counter()
        with torch.inference_mode():
            for bi in range(len(batch)):
                static_ct[bi].copy_(torch.from_dlpack(bev_list[bi]),      non_blocking=True)
                static_proj[bi].copy_(torch.from_dlpack(segproj_list[bi]), non_blocking=True)
            for bi in range(len(batch), batch_size):
                static_ct[bi].zero_()
                static_proj[bi].zero_()
        sync_all()
        t_copy_static += time.perf_counter() - t2

        TOTAL["pre"] += time.perf_counter() - t_pre0
        if print_details:
            DETAIL["pre.prepare_inputs"] += t_prep_calls
            DETAIL["pre.copy_to_static"] += t_copy_static

        # ---------- Inference: CUDA graph replay ----------
        t_inf0 = time.perf_counter()
        with torch.inference_mode():
            graph.replay()
        sync_all()
        TOTAL["infer"] += time.perf_counter() - t_inf0

        # ---------- Postprocessing: scale conversion + inverse resampling to CT + MU-weighted accumulation ----------
        t_post0    = time.perf_counter()
        t_scale    = 0.0
        t_back_map = 0.0

        for bi in range(len(batch)):
            pred_bev = cp.from_dlpack(static_out[bi]).astype(cp.float32, copy=False)

            t_s0     = time.perf_counter()
            pred_bev = cp.maximum(pred_bev * DOSE_SCALE, 0)
            cp.cuda.Stream.null.synchronize()
            t_scale += time.perf_counter() - t_s0

            t_b0   = time.perf_counter()
            # MU-weighted accumulation into dose_sum is handled inside resample_to_ct
            t_back = resample_to_ct(pred_bev, ctx_list[bi], dose_sum, mu_list[bi])
            # use the returned wall-clock from resample_to_ct; fall back to outer measurement if needed
            t_back_map += t_back if isinstance(t_back, (float, int)) else (time.perf_counter() - t_b0)

        TOTAL["post"] += time.perf_counter() - t_post0
        if print_details:
            DETAIL["post.convert_scale"] += t_scale
            DETAIL["post.map_to_ct"]     += t_back_map

        i += batch_size

    sync_all()
    loop_time = time.perf_counter() - loop_start

    # ===== Print timing summary =====
    total_core = TOTAL["pre"] + TOTAL["infer"] + TOTAL["post"]

    def pct(x: float, tot: float) -> float:
        return 100.0 * (x / max(tot, 1e-9))

    print("\n=== Segment-dose pipeline timing (s / %) ===")
    print(f"{'Preprocess':12s}: {TOTAL['pre']:8.3f}  ({pct(TOTAL['pre'],   total_core):5.1f}%)")
    print(f"{'Inference':12s}: {TOTAL['infer']:8.3f}  ({pct(TOTAL['infer'], total_core):5.1f}%)")
    print(f"{'Postprocess':12s}: {TOTAL['post']:8.3f}  ({pct(TOTAL['post'],  total_core):5.1f}%)")
    print(f"{'TOTAL(core)':12s}: {total_core:8.3f}  (100.0%)")
    print(f"{'Overhead*':12s}: {overhead_time:8.3f}  (init + warmup + graph capture)")
    print(f"{'Loop only':12s}: {loop_time:8.3f}  (sum of all batches)\n")

    if print_details and DETAIL:
        print("---- details ----")
        for k in sorted(DETAIL.keys()):
            print(f"{k:24s}: {DETAIL[k]:8.3f}")

    # ===== Write accumulated dose volume to disk =====
    out      = np.transpose(cp.asnumpy(dose_sum), (2, 1, 0))
    out_path = os.path.join(out_dir, out_filename)
    write_mha(out_path, out, ref_img)
    print(f"Wrote: {out_path}")
    return TOTAL["infer"], TOTAL["pre"], TOTAL["post"]
