import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Get the exact folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define the subfolder and filename
subfolder = "Gw_Strain_Data/CuttingEdge2024-2025/GWTC-4.0"  # <-- Python will look inside here now!
filename = "H-H1_GWOSC_O4a_4KHZ_R1-1376088064-4096.hdf5" 

# 3. Create the bulletproof absolute path
file_path = os.path.join(script_dir, subfolder, filename)

print(f"SYSTEM CHECK: Looking for data file at:\n{file_path}\n")

# 4. Open the HDF5 file
try:
    with h5py.File(file_path, 'r') as f:
        # Read the strain data array
        strain_data = f['strain/Strain'][:]
        
        print(f"SUCCESS! Spacetime data loaded into memory.")
        print(f"Total data points: {len(strain_data)}")
        print(f"First 100 strain values: {strain_data[:100]}")

        # Create a time axis (1 step = 1 / 4096 seconds)
        time = np.arange(len(strain_data)) / 4096.0

    # 5. Plot the raw data
    plt.figure(figsize=(12, 4))
    plt.plot(time, strain_data, color='blue', alpha=0.7)
    plt.title("Raw Gravitational Wave Strain Data")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Strain (dimensionless)")
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("\nERROR: Still can't find it!")
    print(f"Did you actually put the .hdf5 file inside the '{subfolder}' folder?")
    print("Check your File Explorer to make sure it's in there!")