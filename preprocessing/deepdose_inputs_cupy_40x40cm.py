import os
import re
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pymedphys
from scipy.ndimage import zoom

# CUDA / GPU Libraries
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
        if (y_idx_f < 0.0f || y_idx_f >= (float)(seg_N_y - 1) ||
            z_idx_f < 0.0f || z_idx_f >= (float)(seg_N_z - 1)) {
            projected_val = 0.0f;
            return;
        }

        // Integer parts (top-left corner)
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
    name='unified_projection_kernel_bilinear_updated'
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
    """Writes a numpy array to an MHA file."""
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
    """Returns the normalized unit vector."""
    return vector / np.linalg.norm(vector)


def crop_zxy_center(array, spacing, origin, target_shape=(128, 192, 192), is_ct=False):
    """
    Crops or pads a 3D array (Z, X, Y) to the target shape.
    - X and Y are cropped based on the matrix center.
    - Z is cropped based on the iso-center (assumes iso-center at physical Z=0).
    """
    fill_value = -1024 if is_ct else 0
    Z, X, Y = array.shape
    tz, tx, ty = target_shape
    result = np.full((tz, tx, ty), fill_value, dtype=array.dtype)

    # Calculate XY center of input matrix
    cx, cy = X // 2, Y // 2
    # Calculate Z slice index corresponding to physical iso-center (0 mm)
    cz = int(round(-origin[2] / spacing[2]))

    # Determine XY slice ranges
    x_start = max(cx - tx // 2, 0)
    x_end = min(cx + (tx - tx // 2), X)
    y_start = max(cy - ty // 2, 0)
    y_end = min(cy + (ty - ty // 2), Y)
    
    out_x_start = max(tx // 2 - cx, 0)
    out_y_start = max(ty // 2 - cy, 0)
    out_x_end = out_x_start + (x_end - x_start)
    out_y_end = out_y_start + (y_end - y_start)

    # Determine Z slice ranges
    z_start = max(cz - tz // 2, 0)
    z_end = min(cz + (tz - tz // 2), Z)
    out_z_start = max(tz // 2 - cz, 0)
    out_z_end = out_z_start + (z_end - z_start)

    # Copy data into result array
    result[out_z_start:out_z_end, out_x_start:out_x_end, out_y_start:out_y_end] = \
        array[z_start:z_end, x_start:x_end, y_start:y_end]

    return result

# ------------------------------------------------------------------------------
# Core Logic: Cuboid Extraction & Physics Feature Calculation
# ------------------------------------------------------------------------------

def hu_to_density_gpu(hu_values):
    """Converts HU values to density using a lookup table on the GPU."""
    hu_points = cp.array([-1024, -999, -200, -199, -10, -9, 120, 121, 3000, 4000], dtype=cp.float32)
    density_points = cp.array([1.20e-03, 1.21e-03, 8.043754e-01, 8.183035e-01, 1.006579e+00, 
                               9.966749e-01, 1.126553e+00, 1.095097e+00, 3.027294e+00, 3.698428e+00], dtype=cp.float32)
    hu_values_float32 = hu_values.astype(cp.float32)
    return cp.interp(hu_values_float32, hu_points, density_points)


def calculate_radiological_depth(ct_cuboid_cp, mask_cuboid_cp, d_x): 
    """Calculates radiological depth by integrating density along the beam direction."""
    density_grid = hu_to_density_gpu(ct_cuboid_cp)
    is_body = mask_cuboid_cp.astype(bool)
    
    # Identify surface indices along the beam axis
    surface_indices = cp.argmax(is_body, axis=0)
    
    # Cumulative sum for line integral
    cumulative_density = cp.cumsum(density_grid, axis=0)
    
    N_y, N_z = ct_cuboid_cp.shape[1], ct_cuboid_cp.shape[2]
    j_indices, k_indices = cp.meshgrid(
        cp.arange(N_y, dtype=cp.int32), 
        cp.arange(N_z, dtype=cp.int32), 
        indexing='ij'
    )
    
    # Subtract density up to the surface to start depth from the skin
    density_at_surface = cumulative_density[surface_indices, j_indices, k_indices]
    radiological_depth = cumulative_density - density_at_surface
    
    # Apply entry mask and body mask
    N_x = ct_cuboid_cp.shape[0]
    i_indices = cp.arange(N_x, dtype=cp.int32).reshape(N_x, 1, 1)
    entry_mask = (i_indices >= surface_indices) 
    
    radiological_depth *= entry_mask.astype(cp.float32)
    radiological_depth *= mask_cuboid_cp.astype(cp.float32)
    radiological_depth *= cp.float32(d_x / 1000.0)
    
    return radiological_depth.astype(cp.float32)


def cuboid_extract(mac_file, mask_array, ct_array, spacing, origin, size, mode,
                   seg_dir, device_id=0):
    """
    Extracts geometric features and projects segment data in BEV, then maps them back
    to the patient coordinate system.

    Args:
        mac_file:   Path to the Geant4 MAC file for this beam segment.
        mask_array: Body mask volume in patient coordinates (X, Y, Z), binary.
        ct_array:   CT Hounsfield-unit volume in patient coordinates (X, Y, Z).
        spacing:    Voxel spacing (mm) as (sx, sy, sz).
        origin:     Volume origin in mm as (ox, oy, oz).
        size:       Volume size in voxels as (nx, ny, nz).
        mode:       Interpolation mode: 'cubic', 'linear', or 'nearest'.
        seg_dir:    Directory containing binary segment files ({seg_name}.bin).
        device_id:  CUDA device index to use (default: 0).
    """
    # 0. Select CUDA device and initialize timers
    cp.cuda.Device(device_id).use()
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event() 
    
    # 1. Parse MAC file and setup geometry on GPU
    source_ori_np, direction_x_np, direction_y_np = extract_gps(mac_file)
    source_ori = cp.asarray(source_ori_np)
    direction_x = cp.asarray(direction_x_np)
    direction_y = cp.asarray(direction_y_np)

    unit_x = direction_x / cp.linalg.norm(direction_x)
    unit_y = direction_y / cp.linalg.norm(direction_y)
    unit_z = cp.array([0, 0, 1])
    unit_matrix = cp.array([unit_x, unit_y, unit_z]) 

    ct_array_cp = cp.asarray(ct_array)
    mask_array_cp = cp.asarray(mask_array)
    spacing_cp = cp.asarray(spacing) 
    origin_cp = cp.asarray(origin)   

    # 2. Generate Beam's Eye View (BEV) Grid
    crop_distance = 1000 - 256  # 216mm before ISO center
    source = source_ori + crop_distance * unit_x 
    imsize_x, imsize_y, imsize_z = 256 * 2, 400, 400
    d_x, d_y, d_z = 2, 2, 2
    
    N_x = int(imsize_x / d_x)
    N_y = int(imsize_y / d_y)
    N_z = int(imsize_z / d_z)

    # Grid generation
    x_indices_cuboid = cp.arange(0, N_x, 1)
    y_indices_cuboid = cp.arange(0, N_y, 1)
    z_indices_cuboid = cp.arange(0, N_z, 1)
    x_matrix, y_matrix, z_matrix = cp.meshgrid(x_indices_cuboid, y_indices_cuboid, z_indices_cuboid, indexing='ij')
    grid_matrix = cp.stack((x_matrix, y_matrix, z_matrix), axis=-1)

    scale_factor = cp.array([d_x, d_y, d_z])
    offset = cp.array([0, -(N_y // 2) + 0.5, -(N_z // 2) + 0.5])
    
    # Transform BEV grid to World Coordinates (P)
    P = source + (grid_matrix + offset) @ cp.diag(scale_factor) @ unit_matrix 

    # Convert World Coordinates to Patient Voxel Indices
    query_points_cp = cp.reshape(P, (N_x * N_y * N_z, 3))
    coords_indices_fwd = cp.zeros_like(query_points_cp) 
    coords_indices_fwd[:, 0] = (query_points_cp[:, 0] - origin_cp[0]) / spacing_cp[0]
    coords_indices_fwd[:, 1] = (query_points_cp[:, 1] - origin_cp[1]) / spacing_cp[1]
    coords_indices_fwd[:, 2] = (query_points_cp[:, 2] - origin_cp[2]) / spacing_cp[2]
    coords_indices_fwd_transposed = coords_indices_fwd.T 

    # 2.1 Forward Resampling: CT and Mask to BEV
    cp.cuda.Stream.null.synchronize()
    start_event.record()

    interp_order = 3 if mode == 'cubic' else (1 if mode == 'linear' else 0)

    ct_cuboid_flat = cpndi.map_coordinates(
        ct_array_cp, 
        coords_indices_fwd_transposed, 
        order=interp_order, 
        mode='constant',
        cval=-1024, 
        prefilter=True
    )
    
    mask_cuboid_flat = cpndi.map_coordinates(
        mask_array_cp, 
        coords_indices_fwd_transposed, 
        order=0, 
        mode='constant', 
        cval=-1024, 
        prefilter=True
    )
    
    ct_cuboid_cp = cp.reshape(ct_cuboid_flat, (N_x, N_y, N_z))
    mask_cuboid_cp = cp.reshape(mask_cuboid_flat, (N_x, N_y, N_z))

    end_event.record()
    end_event.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
    print(f'GPU resample time ({mode}): {elapsed_time:.2f}ms')

    # Clamp resampled values
    ct_cuboid_cp = cp.clip(ct_cuboid_cp, -1024, np.max(ct_array))
    
    # 2.2 Feature Calculation in BEV Frame
    
    # --- Feature 1: Distance From Source (DFS) ---
    uniform_grid = cp.ones((N_x, N_y, N_z), dtype=cp.float32)
    vectors_to_source = (P - source_ori).astype(cp.float32)
    distances_to_source = cp.linalg.norm(vectors_to_source, axis=-1)
    distances_to_source = distances_to_source / 1000.0 
    dfs = uniform_grid * distances_to_source
    
    # --- Feature 2: Central Beamline Distance (CBD) ---
    central_beam_direction = unit_x.astype(cp.float32) 
    vectors_from_source = (P - source_ori).astype(cp.float32) 
    projection_lengths = cp.sum(vectors_from_source * central_beam_direction, axis=-1)
    projection_vectors = projection_lengths[..., cp.newaxis] * central_beam_direction
    perpendicular_vectors = vectors_from_source - projection_vectors
    central_beamline_distance = cp.linalg.norm(perpendicular_vectors, axis=-1)  
    central_beamline_distance = central_beamline_distance / 1000.0 
    cbd = uniform_grid * central_beamline_distance
    
    # --- Feature 3: Radiological Depth (RD) ---
    rd = calculate_radiological_depth(ct_cuboid_cp, mask_cuboid_cp, d_x)
    
    # --- Feature 4: Segment Projection ---
    segment_name = re.search(r"seg(.*)\.mac", mac_file).group(1)
    segment = np.fromfile(os.path.join(seg_dir, f'{segment_name}.bin'), dtype=np.int8)
    segment = segment.reshape(400, 400).astype(np.float32)
    segment_2mm = zoom(segment, 0.5, order=1)
    segment_2mm = np.flip(np.rot90(segment_2mm, 3), axis=0)

    # Perform Projection
    cp.cuda.Stream.null.synchronize()
    start_event.record()

    segment_2mm_cp = cp.asarray(segment_2mm, dtype=cp.float32) 
    P_flat = cp.reshape(P, (N_x * N_y * N_z, 3)).astype(np.float32) 

    projection_source = source_ori 
    distance_iso_from_projection_source_mm = np.float32(1000.0)
    epsilon = np.float32(1e-9)
    bev_grid_origin = source_ori + crop_distance * unit_x 
    
    unit_matrix_T = unit_matrix.T
    bev_project_y_axis_world = unit_matrix_T[:, 1]
    bev_project_z_axis_world = unit_matrix_T[:, 2]

    seg_actual_N_y = segment_2mm_cp.shape[0]
    seg_actual_N_z = segment_2mm_cp.shape[1]

    projected_segment_flat = cp.zeros(N_x * N_y * N_z, dtype=cp.float32)

    projection_kernel(
        P_flat, segment_2mm_cp,
        projection_source[0].item(), projection_source[1].item(), projection_source[2].item(),
        distance_iso_from_projection_source_mm,
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

    # 3. Inverse Mapping / Back-projection
    # Generate the base coordinates of the patient volume,
    # then express these coordinates relative to the BEV frame for resampling.
    
    x_np = np.arange(origin[0], origin[0] + size[0] * spacing[0], spacing[0])
    y_np = np.arange(origin[1], origin[1] + size[1] * spacing[1], spacing[1])
    z_np = np.arange(origin[2], origin[2] + size[2] * spacing[2], spacing[2])
    
    X_cp, Y_cp, Z_cp = cp.meshgrid(cp.asarray(x_np), cp.asarray(y_np), cp.asarray(z_np), indexing='ij')
    P_back_cp = cp.stack((X_cp, Y_cp, Z_cp), axis=-1)

    inv_scale_factor = 1.0 / scale_factor 
    query_points_back_cp = (P_back_cp - source) @ cp.transpose(unit_matrix) @ cp.diag(inv_scale_factor) - offset
    
    original_shape = ct_array.shape 
    query_points_back_flat = cp.reshape(query_points_back_cp, (original_shape[0] * original_shape[1] * original_shape[2], 3))
    coords_indices_bwd_transposed = query_points_back_flat.T

    cp.cuda.Stream.null.synchronize()
    start_event.record()

    # Resample features back to patient geometry
    segment_array_back_flat = cpndi.map_coordinates(
        projected_segment_cuboid_cp, coords_indices_bwd_transposed,
        order=interp_order, mode='constant', cval=0.0, prefilter=True
    )
    segment_array_back_cp = cp.reshape(segment_array_back_flat, original_shape)

    dfs_back_flat = cpndi.map_coordinates(
        dfs, coords_indices_bwd_transposed,
        order=interp_order, mode='constant', cval=0.0, prefilter=True
    )
    dfs_back_cp = cp.reshape(dfs_back_flat, original_shape)
    
    cbd_back_flat = cpndi.map_coordinates(
        cbd, coords_indices_bwd_transposed,
        order=interp_order, mode='constant', cval=0.0, prefilter=True
    )
    cbd_back_cp = cp.reshape(cbd_back_flat, original_shape)    
    
    rd_back_flat = cpndi.map_coordinates(
        rd, coords_indices_bwd_transposed,
        order=interp_order, mode='constant', cval=0.0, prefilter=True
    )
    rd_back_cp = cp.reshape(rd_back_flat, original_shape)  

    end_event.record()
    end_event.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start_event, end_event)
    print(f'GPU resample back time ({mode}): {elapsed_time:.2f}ms')

    # Clamp negative values caused by cubic spline undershoot
    segment_array_back_cp[segment_array_back_cp < 0] = 0
    
    segment_array_back = cp.asnumpy(segment_array_back_cp)
    dfs_back = cp.asnumpy(dfs_back_cp)
    cbd_back = cp.asnumpy(cbd_back_cp)
    rd_back = cp.asnumpy(rd_back_cp)

    return segment_array_back, dfs_back, cbd_back, rd_back


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract DeepDose physical inputs (segment projection, DFS, CBD, RD) "
            "for all patients and save as binary files."
        )
    )
    parser.add_argument(
        "--sim-root", required=True,
        help="Simulation data root; must contain {patient}/dose_mha/ and {patient}/setup/.",
    )
    parser.add_argument(
        "--ct-root", required=True,
        help=(
            "CT root; must contain {patient}/maskedCT_3mm_shifted.mha "
            "and {patient}/Body_shifted.mha."
        ),
    )
    parser.add_argument(
        "--seg-dir", required=True,
        help="Directory containing binary segment files ({seg_name}.bin, int8, shape 400x400).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help=(
            "Output root directory; sub-folders dose/, proj/, ct/, dfs/, cbd/, rd/ "
            "are created automatically."
        ),
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
    output_path       = args.out_dir
    simulation_folder = args.sim_root

    # Create output sub-directories
    os.makedirs(output_path, exist_ok=True)
    for sub_dir in ['dose', 'proj', 'ct', 'dfs', 'cbd', 'rd']:
        os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)

    for patient in sorted(os.listdir(simulation_folder)):
        ct_path   = os.path.join(args.ct_root, patient, 'maskedCT_3mm_shifted.mha')
        mask_path = os.path.join(args.ct_root, patient, 'Body_shifted.mha')

        # Load patient CT and body mask
        ct_array   = read_mha(ct_path,   option='array')
        spacing, origin, size = read_mha(ct_path, option='info')
        mask_array = read_mha(mask_path, option='array')

        setup_folder = os.path.join(simulation_folder, patient, 'setup')
        dose_folder  = os.path.join(simulation_folder, patient, 'dose_mha')

        if not os.path.exists(dose_folder):
            continue

        for dose_file in os.listdir(dose_folder):
            dose_path = os.path.join(dose_folder, dose_file)
            mac_name  = dose_file[5:-4]
            mac_file  = os.path.join(setup_folder, f'{mac_name}.mac')

            dose_array = read_mha(dose_path, option='array')

            # Transpose to (X, Y, Z) axis order expected by cuboid_extract
            ct_input   = np.transpose(ct_array,   (2, 1, 0))
            mask_input = np.transpose(mask_array, (2, 1, 0))

            # Extract physical features and back-project to patient coordinates
            proj_array_back, dfs_back, cbd_back, rd_back = cuboid_extract(
                mac_file, mask_input, ct_input, spacing, origin, size, mode,
                seg_dir=args.seg_dir, device_id=args.device_id,
            )
            
            # Post-process outputs (transpose back and apply mask)
            proj_array_back = np.transpose(proj_array_back, (2, 1, 0))
            proj_array_back[mask_array == 0] = 0
            
            dfs_back = np.transpose(dfs_back, (2, 1, 0))
            dfs_back[mask_array == 0] = 0
            
            cbd_back = np.transpose(cbd_back, (2, 1, 0))
            cbd_back[mask_array == 0] = 0
            
            rd_back = np.transpose(rd_back, (2, 1, 0))
            rd_back[mask_array == 0] = 0

            # Crop to target shape and save
            crop_shape = (128, 192, 192)
            
            dose_crop = crop_zxy_center(dose_array, spacing, origin, target_shape=crop_shape, is_ct=False)
            dose_crop.tofile(os.path.join(output_path, 'dose', f'dose_{patient}_{mac_name}.bin'))
            
            ct_crop = crop_zxy_center(ct_array, spacing, origin, target_shape=crop_shape, is_ct=True)
            ct_crop.tofile(os.path.join(output_path, 'ct', f'ct_{patient}_{mac_name}.bin'))
            
            proj_array_back_crop = crop_zxy_center(proj_array_back, spacing, origin, target_shape=crop_shape, is_ct=False)
            proj_array_back_crop.tofile(os.path.join(output_path, 'proj', f'proj_{patient}_{mac_name}.bin'))
            
            dfs_back_crop = crop_zxy_center(dfs_back, spacing, origin, target_shape=crop_shape, is_ct=False)
            dfs_back_crop.tofile(os.path.join(output_path, 'dfs', f'dfs_{patient}_{mac_name}.bin'))
            
            cbd_back_crop = crop_zxy_center(cbd_back, spacing, origin, target_shape=crop_shape, is_ct=False)
            cbd_back_crop.tofile(os.path.join(output_path, 'cbd', f'cbd_{patient}_{mac_name}.bin'))
            
            rd_back_crop = crop_zxy_center(rd_back, spacing, origin, target_shape=crop_shape, is_ct=False)
            rd_back_crop.tofile(os.path.join(output_path, 'rd', f'rd_{patient}_{mac_name}.bin'))