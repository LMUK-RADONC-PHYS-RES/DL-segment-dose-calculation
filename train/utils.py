import numpy as np
import torch
import h5py
import pymedphys
from monai.data import Dataset


class MultiModalHDF5Dataset(Dataset):
    def __init__(self, ct_hdf_path: str, proj_hdf_path: str, dose_hdf_path: str,
                 group_name: str = 'train',
                 input_transform=None, target_transform=None):
        """
        Args:
            ct_hdf_path (str): Path to the CT HDF5 file.
            proj_hdf_path (str): Path to the projection HDF5 file.
            dose_hdf_path (str): Path to the dose HDF5 file.
            group_name (str): HDF5 group name to read ('train' or 'validation').
            input_transform (callable, optional): Transform applied to input data.
            target_transform (callable, optional): Transform applied to target data.
        """
        self.ct_hdf_path   = ct_hdf_path
        self.proj_hdf_path = proj_hdf_path
        self.dose_hdf_path = dose_hdf_path
        self.group_name      = group_name
        self.input_transform  = input_transform
        self.target_transform = target_transform

        # File handles, opened independently per DataLoader worker
        self.ct_file   = None
        self.proj_file = None
        self.dose_file = None

        # On init, retrieve all keys and verify consistency across files
        with h5py.File(self.ct_hdf_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.ct_hdf_path}")
            self.ct_keys = sorted(list(f[group_name].keys()))

        with h5py.File(self.proj_hdf_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.proj_hdf_path}")
            self.proj_keys = sorted(list(f[group_name].keys()))

        with h5py.File(self.dose_hdf_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in {self.dose_hdf_path}")
            self.dose_keys = sorted(list(f[group_name].keys()))

        # Verify that all three files have the same number of samples
        if not (len(self.ct_keys) == len(self.proj_keys) == len(self.dose_keys)):
            raise ValueError(
                f"Sample count mismatch in group '{group_name}':\n"
                f"  {self.ct_hdf_path} (CT): {len(self.ct_keys)} samples\n"
                f"  {self.proj_hdf_path} (Proj): {len(self.proj_keys)} samples\n"
                f"  {self.dose_hdf_path} (Dose): {len(self.dose_keys)} samples\n"
                "Ensure each file contains the same number of corresponding samples."
            )

        self.length = len(self.ct_keys)
        if self.length == 0:
            print(f"Warning: no data found in group '{group_name}'.")

    def _open_files(self):
        """Open HDF5 file handles for this worker if not already open."""
        if self.ct_file is None:
            self.ct_file   = h5py.File(self.ct_hdf_path,   'r')
            self.proj_file = h5py.File(self.proj_hdf_path, 'r')
            self.dose_file = h5py.File(self.dose_hdf_path, 'r')

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        self._open_files()

        if not (0 <= idx < self.length):
            raise IndexError(f"Index {idx} out of range (0 to {self.length - 1})")

        ct_key   = self.ct_keys[idx]
        proj_key = self.proj_keys[idx]
        dose_key = self.dose_keys[idx]

        try:
            ct_data   = self.ct_file[self.group_name][ct_key][()]
            proj_data = self.proj_file[self.group_name][proj_key][()]
            dose_data = self.dose_file[self.group_name][dose_key][()]
        except KeyError as e:
            print(f"KeyError at index {idx}:")
            print(f"  CT  : '{ct_key}'   from {self.ct_hdf_path}")
            print(f"  Proj: '{proj_key}' from {self.proj_hdf_path}")
            print(f"  Dose: '{dose_key}' from {self.dose_hdf_path}")
            print(f"  Error: {e}")
            for hdf_path, key in [
                (self.ct_hdf_path,   ct_key),
                (self.proj_hdf_path, proj_key),
                (self.dose_hdf_path, dose_key),
            ]:
                with h5py.File(hdf_path, 'r') as tmp:
                    if key not in tmp[self.group_name]:
                        print(f"  Confirmed missing: '{key}' in '{hdf_path}' group '{self.group_name}'")
                        break
            raise

        ct_tensor   = torch.from_numpy(ct_data.astype(np.float32))
        proj_tensor = torch.from_numpy(proj_data.astype(np.float32))
        dose_tensor = torch.from_numpy(dose_data.astype(np.float32))

        inputs = (ct_tensor, proj_tensor)
        if self.input_transform:
            inputs = self.input_transform(inputs)
        if self.target_transform:
            dose_tensor = self.target_transform(dose_tensor)

        return inputs, dose_tensor

    def close(self):
        """Close all open HDF5 file handles."""
        if self.ct_file:
            self.ct_file.close()
            self.ct_file = None
        if self.proj_file:
            self.proj_file.close()
            self.proj_file = None
        if self.dose_file:
            self.dose_file.close()
            self.dose_file = None

    def __del__(self):
        self.close()

