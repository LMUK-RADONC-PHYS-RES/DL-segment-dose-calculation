import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage import zoom
import pymedphys

# CUDA imports
import cupy as cp
import cupyx.scipy.ndimage as cpndi

# ------------------------------------------------------------------------------
# CUDA Kernels
# ------------------------------------------------------------------------------

# Unified projection kernel using bilinear interpolation
projection_kernel = cp.ElementwiseKernel(
    in_params='raw float32 P_coord, raw float32 segment_data, \
               float32 src_x, float32 src_y, float32 src_z, \
               float32 iso_dist_mm, \
               float32 normal_x, float32 normal_y, float32 normal_z, \
               float32 bev_origin_x, float32 bev_origin_y, float32 bev_origin_z, \
               float32 bev_proj_y_x, float32 bev_proj_y_y, float32 bev_proj_y_z, \
               float32 bev_proj_z_x, float32 bev_proj_z_y, float32 bev_proj_z_z, \
               float32 pix_dy, float32 pix_dz, \
               float32 off_y, float32 off_z, \
               int32 seg_N_y, int32 seg_N_z, float32 kernel_eps',
    out_params='float32 projected_val',
    operation='''
        // P_coord represents a single 3D point (x, y, z) flattened
        float px = P_coord[i*3 + 0];
        float py = P_coord[i*3 + 1];
        float pz = P_coord[i*3 + 2];

        // Ray direction vector
        float ray_dir_x = px - src_x;
        float ray_dir_y = py - src_y;
        float ray_dir_z = pz - src_z;

        // Dot product: ray_direction . iso_plane_normal
        float dot_prod = ray_dir_x * normal_x + ray_dir_y * normal_y + ray_dir_z * normal_z;

        // Check for parallel rays
        if (abs(dot_prod) < kernel_eps) {
            projected_val = 0.0f;
            return;
        }

        // Intersection parameter t
        float t_intersect = iso_dist_mm / dot_prod;

        // Check if intersection is behind the source
        if (t_intersect <= 0) {
            projected_val = 0.0f;
            return;
        }

        // Intersection point in world coordinates
        float int_world_x = src_x + t_intersect * ray_dir_x;
        float int_world_y = src_y + t_intersect * ray_dir_y;
        float int_world_z = src_z + t_intersect * ray_dir_z;

        // Vector from BEV grid origin to intersection point
        float delta_bev_x = int_world_x - bev_origin_x;
        float delta_bev_y = int_world_y - bev_origin_y;
        float delta_bev_z = int_world_z - bev_origin_z;

        // Project delta vector onto BEV local Y and Z axes
        float y_local = delta_bev_x * bev_proj_y_x + delta_bev_y * bev_proj_y_y + delta_bev_z * bev_proj_y_z;
        float z_local = delta_bev_x * bev_proj_z_x + delta_bev_y * bev_proj_z_y + delta_bev_z * bev_proj_z_z;

        // Convert local BEV coordinates to floating-point pixel indices
        float y_idx_f = (y_local / pix_dy) - off_y;
        float z_idx_f = (z_local / pix_dz) - off_z;

        // Bilinear Interpolation Boundary Check
        // Ensure the 2x2 neighborhood is within valid bounds
        if (y_idx_f < 0.0f || y_idx_f >= (float)(seg_N_y - 1) ||
            z_idx_f < 0.0f || z_idx_f >= (float)(seg_N_z - 1)) {
            projected_val = 0.0f;
            return;
        }

        // Integer parts (top-left corner of the 2x2 grid)
        int y0 = (int)floorf(y_idx_f);
        int z0 = (int)floorf(z_idx_f);
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        // Fractional parts (weights)
        float wy = y_idx_f - (float)y0; 
        float wz = z_idx_f - (float)z0; 

        // Fetch neighbors
        float v00 = segment_data[y0 * seg_N_z + z0];
        float v10 = segment_data[y1 * seg_N_z + z0];
        float v01 = segment_data[y0 * seg_N_z + z1];
        float v11 = segment_data[y1 * seg_N_z + z1];

        // Interpolate
        float tmp_val_y0 = (1.0f - wz) * v00 + wz * v01;
        float tmp_val_y1 = (1.0f - wz) * v10 + wz * v11;

        projected_val = (1.0f - wy) * tmp_val_y0 + wy * tmp_val_y1;
    ''',
    name='unified_projection_kernel_bilinear'
)


# ------------------------------------------------------------------------------
# I/O and Utility Functions
# ------------------------------------------------------------------------------

def read_mha(mha_path, option='info'):
    """Reads an MHA file and returns image metadata or the array."""
    image = sitk.ReadImage(mha_path)

    if option == 'info':
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        size = image.GetSize()
        return spacing, origin, size

    if option == 'array':
        array = sitk.GetArrayFromImage(image)
        return array


def write_mha(path, image_array, reference_image=None):
    """Writes a numpy array to an MHA file, optionally copying metadata."""
    new_image = sitk.GetImageFromArray(image_array)

    if reference_image is not None:
        new_image.SetOrigin(reference_image.GetOrigin())
        new_image.SetSpacing(reference_image.GetSpacing())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(new_image)


def extract_gps(mac_file):
    """Parses the MAC file to extract source position and rotation vectors."""
    focuspoint_pattern = r"/gps/ang/focuspoint\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+mm"
    direction_pattern = r"/gps/direction\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"
    rot1_pattern = r"/gps/pos/rot1\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"

    source = None
    direction_x = None
    direction_y = None

    with open(mac_file, 'r') as file:
        for line in file:
            if re.search(focuspoint_pattern, line):
                source = tuple(map(float, re.findall(focuspoint_pattern, line)[0]))
            elif re.search(direction_pattern, line):
                direction_x = tuple(map(float, re.findall(direction_pattern, line)[0]))
            elif re.search(rot1_pattern, line):
                direction_y = tuple(map(float, re.findall(rot1_pattern, line)[0]))

    return source, direction_x, direction_y


def unit_vector(vector):
    """Returns the unit vector of the input."""
    return vector / np.linalg.norm(vector)

# ------------------------------------------------------------------------------
# Core Logic: Cuboid Extraction & Projection
# ------------------------------------------------------------------------------

def cuboid_extract(mac_file, dose_array, ct_array, spacing, origin, size, mode,
                   seg_dir, device_id=0):
    """
    Extracts BEV-aligned cuboids for Dose and CT, and projects the segment map.
    Uses GPU acceleration via CuPy.

    Args:
        mac_file:   Path to the Geant4 MAC file for this beam segment.
        dose_array: Monte Carlo dose volume in patient coordinates (X, Y, Z).
        ct_array:   CT Hounsfield-unit volume in patient coordinates (X, Y, Z).
        spacing:    Voxel spacing (mm) as (sx, sy, sz).
        origin:     Volume origin in mm as (ox, oy, oz).
        size:       Volume size in voxels as (nx, ny, nz).
        mode:       Interpolation mode: 'cubic', 'linear', or 'nearest'.
        seg_dir:    Directory containing binary segment files ({seg_name}.bin).
        device_id:  CUDA device index to use (default: 0).
    """
    # 0. Select CUDA device and initialize performance timers
    cp.cuda.Device(device_id).use()
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # 1. Parse GPS and convert geometry to CuPy
    source_ori_np, direction_x_np, direction_y_np = extract_gps(mac_file)
    
    source_ori = cp.asarray(source_ori_np)
    direction_x = cp.asarray(direction_x_np)
    direction_y = cp.asarray(direction_y_np)

    unit_x = direction_x / cp.linalg.norm(direction_x)
    unit_y = direction_y / cp.linalg.norm(direction_y)
    unit_z = cp.array([0, 0, 1])
    unit_matrix = cp.array([unit_x, unit_y, unit_z])

    # Transfer volume data to GPU
    dose_array_cp = cp.asarray(dose_array)
    ct_array_cp = cp.asarray(ct_array)
    spacing_cp = cp.asarray(spacing)
    origin_cp = cp.asarray(origin)

    # 2. Generate BEV Grid Matrix
    crop_distance = 1000 - 256  # 216mm before ISO center
    source = source_ori + crop_distance * unit_x
    
    # Target grid dimensions
    imsize_x, imsize_y, imsize_z = 256 * 2, 400, 400
    d_x, d_y, d_z = 2, 2, 2
    
    N_x = int(imsize_x / d_x)
    N_y = int(imsize_y / d_y)
    N_z = int(imsize_z / d_z)

    # Construct the meshgrid
    x_indices_cuboid = cp.arange(0, N_x, 1)
    y_indices_cuboid = cp.arange(0, N_y, 1)
    z_indices_cuboid = cp.arange(0, N_z, 1)
    x_matrix, y_matrix, z_matrix = cp.meshgrid(x_indices_cuboid, y_indices_cuboid, z_indices_cuboid, indexing='ij')
    
    grid_matrix = cp.stack((x_matrix, y_matrix, z_matrix), axis=-1)
    scale_factor = cp.array([d_x, d_y, d_z])
    offset = cp.array([0, -(N_y // 2) + 0.5, -(N_z // 2) + 0.5])

    # Transform BEV grid points to World Coordinates (P)
    P = source + (grid_matrix + offset) @ cp.diag(scale_factor) @ unit_matrix

    # Convert World Coordinates to Voxel Indices (Float)
    query_points_cp = cp.reshape(P, (N_x * N_y * N_z, 3))
    coords_indices_fwd = cp.zeros_like(query_points_cp)
    
    coords_indices_fwd[:, 0] = (query_points_cp[:, 0] - origin_cp[0]) / spacing_cp[0]
    coords_indices_fwd[:, 1] = (query_points_cp[:, 1] - origin_cp[1]) / spacing_cp[1]
    coords_indices_fwd[:, 2] = (query_points_cp[:, 2] - origin_cp[2]) / spacing_cp[2]
    
    coords_indices_fwd_transposed = coords_indices_fwd.T

    # 2.1 Interpolation (Resampling)
    cp.cuda.Stream.null.synchronize()
    start_event.record()

    # Determine interpolation order
    interp_order = 3 if mode == 'cubic' else (1 if mode == 'linear' else 0)

    # Resample Dose
    dose_cuboid_flat = cpndi.map_coordinates(
        dose_array_cp,
        coords_indices_fwd_transposed,
        order=interp_order,
        mode='constant',
        cval=0.0,
        prefilter=True
    )

    # Resample CT
    ct_cuboid_flat = cpndi.map_coordinates(
        ct_array_cp,
        coords_indices_fwd_transposed,
        order=interp_order,
        mode='constant',
        cval=-1024,
        prefilter=True
    )

    dose_cuboid_cp = cp.reshape(dose_cuboid_flat, (N_x, N_y, N_z))
    ct_cuboid_cp = cp.reshape(ct_cuboid_flat, (N_x, N_y, N_z))

    end_event.record()
    end_event.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
    print(f'GPU resample time ({mode}): {elapsed_time:.2f}ms')

    # Clip values to valid ranges
    dose_cuboid_cp = cp.clip(dose_cuboid_cp, 0, np.max(dose_array))
    ct_cuboid_cp = cp.clip(ct_cuboid_cp, -1024, np.max(ct_array))

    # 2.2 Generate Segment Projection
    segment_name = re.search(r"seg(.*)\.mac", mac_file).group(1)

    # Read binary segment file (int8, shape 400x400)
    segment_path = os.path.join(seg_dir, f'{segment_name}.bin')
    segment = np.fromfile(segment_path, dtype=np.int8)
    segment = segment.reshape(400, 400).astype(np.float32)
    
    # Process segment (Zoom and Rotate)
    segment_2mm = zoom(segment, 0.5, order=1)  # Linear interpolation
    segment_2mm = np.flip(np.rot90(segment_2mm, 3), axis=0)

    # Prepare for projection kernel
    cp.cuda.Stream.null.synchronize()
    start_event.record()

    segment_2mm_cp = cp.asarray(segment_2mm, dtype=cp.float32)
    P_flat = cp.reshape(P, (N_x * N_y * N_z, 3)).astype(np.float32)

    # Projection Kernel Parameters
    projection_source = source_ori
    distance_iso_from_proj = np.float32(1000.0)
    epsilon = np.float32(1e-9)
    bev_grid_origin = source_ori + crop_distance * unit_x
    unit_matrix_T = unit_matrix.T
    bev_project_y_axis_world = unit_matrix_T[:, 1]
    bev_project_z_axis_world = unit_matrix_T[:, 2]
    seg_actual_N_y = segment_2mm_cp.shape[0]
    seg_actual_N_z = segment_2mm_cp.shape[1]

    projected_segment_flat = cp.zeros(N_x * N_y * N_z, dtype=cp.float32)

    # Execute Kernel
    projection_kernel(
        P_flat, segment_2mm_cp,
        projection_source[0].item(), projection_source[1].item(), projection_source[2].item(),
        distance_iso_from_proj,
        unit_x[0].item(), unit_x[1].item(), unit_x[2].item(),
        bev_grid_origin[0].item(), bev_grid_origin[1].item(), bev_grid_origin[2].item(),
        bev_project_y_axis_world[0].item(), bev_project_y_axis_world[1].item(), bev_project_y_axis_world[2].item(),
        bev_project_z_axis_world[0].item(), bev_project_z_axis_world[1].item(), bev_project_z_axis_world[2].item(),
        d_y, d_z,
        offset[1].item(), offset[2].item(),
        seg_actual_N_y, seg_actual_N_z,
        epsilon,
        projected_segment_flat
    )

    projected_segment_cuboid_cp = cp.reshape(projected_segment_flat, (N_x, N_y, N_z))

    end_event.record()
    end_event.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
    print(f'GPU Projection time: {elapsed_time:.2f}ms')

    # 3. Retrieve results from GPU
    ct_cuboid = ct_cuboid_cp.get()
    dose_cuboid = dose_cuboid_cp.get()
    projected_segment_cuboid = projected_segment_cuboid_cp.get()

    return dose_cuboid, ct_cuboid, projected_segment_cuboid


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract BEV-aligned cuboids (dose, CT, segment projection) for all patients."
    )
    parser.add_argument(
        "--sim-root", required=True,
        help="Simulation data root; must contain {patient}/dose_mha/ and {patient}/setup/.",
    )
    parser.add_argument(
        "--ct-root", required=True,
        help="CT root; must contain {patient}/maskedCT_3mm_shifted.mha.",
    )
    parser.add_argument(
        "--seg-dir", required=True,
        help="Directory containing binary segment files ({seg_name}.bin, int8, shape 400x400).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output root directory; sub-folders dose/, ct/, proj/, fig_debug/ are created automatically.",
    )
    parser.add_argument(
        "--mode", choices=["linear", "cubic", "nearest"], default="cubic",
        help="Interpolation mode for GPU resampling (default: cubic).",
    )
    parser.add_argument(
        "--device-id", type=int, default=0,
        help="CUDA device index (default: 0).",
    )
    args = parser.parse_args()

    mode              = args.mode
    base_output_path  = args.out_dir
    simulation_folder = args.sim_root

    # Create output sub-directories
    os.makedirs(base_output_path, exist_ok=True)
    for sub_dir in ['dose', 'ct', 'proj', 'fig_debug']:
        os.makedirs(os.path.join(base_output_path, sub_dir), exist_ok=True)

    for patient in sorted(os.listdir(simulation_folder)):
        ct_path = os.path.join(args.ct_root, patient, 'maskedCT_3mm_shifted.mha')

        # Load CT data and transpose to (X, Y, Z) axis order
        ct_array = read_mha(ct_path, option='array')
        ct_array = np.transpose(ct_array, (2, 1, 0))
        spacing, origin, size = read_mha(ct_path, option='info')

        setup_folder = os.path.join(simulation_folder, patient, 'setup')
        dose_folder  = os.path.join(simulation_folder, patient, 'dose_mha')

        if not os.path.exists(dose_folder):
            print(f"Skipping {patient}: dose folder not found.")
            continue

        for dose_file in os.listdir(dose_folder):
            if not dose_file.endswith('.mha'):
                continue

            dose_path = os.path.join(dose_folder, dose_file)
            mac_name  = dose_file[5:-4]  # strip 'dose_' prefix and '.mha' suffix
            mac_file  = os.path.join(setup_folder, f'{mac_name}.mac')

            dose_array = read_mha(dose_path, option='array')
            dose_array = np.transpose(dose_array, (2, 1, 0))

            try:
                dose_cuboid, ct_cuboid, projected_segment_cuboid = cuboid_extract(
                    mac_file, dose_array, ct_array, spacing, origin, size, mode,
                    seg_dir=args.seg_dir, device_id=args.device_id,
                )

                # Validate shape and save
                if dose_cuboid.shape == (256, 200, 200):
                    dose_cuboid.tofile(os.path.join(base_output_path, 'dose', f'dose_{patient}_{mac_name}.bin'))
                    ct_cuboid.tofile(os.path.join(base_output_path, 'ct', f'ct_{patient}_{mac_name}.bin'))
                    projected_segment_cuboid.tofile(os.path.join(base_output_path, 'proj', f'proj_{patient}_{mac_name}.bin'))
                else:
                    print(f'Warning: Unexpected shape for {patient}_{mac_name}: {dose_cuboid.shape}')
                    
                    # Generate debug plot
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(ct_cuboid[:, :, 100], cmap='gray')
                    plt.title("CT Mid-slice")
                    plt.subplot(1, 2, 2)
                    plt.imshow(dose_cuboid[:, :, 100], cmap='viridis', alpha=0.6)
                    plt.title("Dose Mid-slice")
                    
                    debug_fig_path = os.path.join(base_output_path, 'fig_debug', f'{patient}_{mac_name}_X{dose_cuboid.shape[0]}.png')
                    plt.savefig(debug_fig_path)
                    plt.close()

            except Exception as e:
                print(f"Error processing {patient} - {mac_name}: {e}")