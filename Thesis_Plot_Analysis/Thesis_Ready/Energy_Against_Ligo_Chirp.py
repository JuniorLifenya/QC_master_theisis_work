# import numpy as np
# import matplotlib.pyplot as plt
# from gwpy.timeseries import TimeSeries


# # 1. Get the exact folder where this script lives
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # 2. Define the subfolder and filename
# subfolder = "../data/Gw_Strain_Data/CuttingEdge2024-2025/GWTC-4.0"  # <-- Python will look inside here now!
# filename = "H-H1_GWOSC_O4a_4KHZ_R1-1376088064-4096.hdf5" 

# # 3. Create the bulletproof absolute path
# file_path = os.path.join(script_dir, subfolder, filename)

# print(f"SYSTEM CHECK: Looking for data file at:\n{file_path}\n")

# # 4. Open the HDF5 file
# try:
#     with h5py.File(file_path, 'r') as f:
#         # Read the strain data array
#         strain_data = f['strain/Strain'][:]
        
#         print(f"SUCCESS! Spacetime data loaded into memory.")
#         print(f"Total data points: {len(strain_data)}")
#         print(f"First 100 strain values: {strain_data[:100]}")

#         # Create a time axis (1 step = 1 / 4096 seconds)
#         time = np.arange(len(strain_data)) / 4096.0
        
# # Load GW150914 strain
# strain = TimeSeries.fetch_open_data('H1', 1126259462, 1126259462+32)
# h_plus = strain.value  # dimensionless strain

# # Constants
# kappa = 1.15e-27  # eV^-1 (sqrt(32*pi*G)/c^2 in natural units)
# me = 511e3        # eV
# alpha = 1/137
# p2_2p = (me * alpha)**2 / 4  # <p^2>_{2p} in eV^2
# ang_element = 2/5  # |<1,+1|sin²θcos2φ|1,-1>|

# # Energy shift in eV
# delta_E = (kappa / (2 * me)) * h_plus * p2_2p * ang_element
# # Convert to frequency
# hbar_eV = 6.582e-16  # eV·s
# delta_nu = delta_E / hbar_eV  # Hz

# t = np.linspace(0, len(h_plus)/4096, len(h_plus))
# plt.plot(t, delta_E)
# plt.ylabel('ΔE (eV)')
# plt.xlabel('Time (s)')
# plt.title('GW150914: Kinetic Strain Energy Shift on H 2p State')