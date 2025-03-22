import os
import glob
import numpy as np
import h5py

mat_files = sorted(glob.glob("CorrData_SNR_*dB_data.mat"))

for mat_file in mat_files:
    print(f"Processing {mat_file}...")
    
    with h5py.File(mat_file, 'r') as f:
        keys = list(f.keys())
        print("Available keys:", keys)

        if 'dataSet' not in f:
            raise KeyError(f"No 'dataSet' key found in {mat_file}. Keys: {keys}")

        # Load data from v7.3 .mat (HDF5)
        data_np = np.array(f['dataSet'])  # Possibly shape (48,48,4,5000) or something else
        
        # If labels exist
        labels_np = None
        if 'labelSet' in f:
            labels_np = np.array(f['labelSet'])  # Possibly shape (48, 5000) or (5000, 48) etc.

    # --- Fix data shape ---
    old_data_shape = data_np.shape
    # If it exactly matches (48,48,4,5000), transpose to (5000,4,48,48):
    if old_data_shape == (48, 48, 4, 10000):
        data_np = data_np.transpose(3, 2, 0, 1)  # -> (5000,4,48,48)
        print(f"Data transposed from {old_data_shape} to {data_np.shape}")
    else:
        print(f"Data shape {old_data_shape} does not match (48,48,4,5000). No transpose applied.")

    # --- Fix label shape ---
    if labels_np is not None:
        old_label_shape = labels_np.shape
        # If it's (48, 5000), transpose to (5000, 48):
        if old_label_shape == (48, 10000):
            labels_np = labels_np.transpose()  # -> (5000, 48)
            print(f"Labels transposed from {old_label_shape} to {labels_np.shape}")
        else:
            print(f"Label shape {old_label_shape} does not match (48,5000). No transpose applied.")

    # Build output filenames
    npy_file_data = mat_file.replace(".mat", ".npy")  
    npy_file_label = npy_file_data.replace("_data.npy", "_label.npy")

    # Save the data
    np.save(npy_file_data, data_np)
    print(f"Saved {npy_file_data} with shape {data_np.shape}, dtype {data_np.dtype}")

    if labels_np is not None:
        np.save(npy_file_label, labels_np)
        print(f"Saved {npy_file_label} with shape {labels_np.shape}, dtype {labels_np.dtype}")
