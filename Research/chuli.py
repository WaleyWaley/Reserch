import numpy as np
import pandas as pd

print("Starting the data merging process...")

# --- 1. Load Data ---
# This script assumes all required files are in the same folder as the script.
try:
    gt_data = np.load('03-05_16-08_GT.npy')
    imu_data = np.load('03-05_16-08_IMU.npy')
    wifi_data = np.load('03-05_16-08_RssOWP.npy')
    print("Successfully loaded all .npy files.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure all .npy files are in the same folder as this script.")
    exit()

# --- 2. Define Columns ---
gt_cols = ['timestamp', 'x_coord', 'y_coord', 'z_coord',
           'rot_11', 'rot_12', 'rot_13', 'rot_21', 'rot_22', 'rot_23',
           'rot_31', 'rot_32', 'rot_33']
imu_cols = ['timestamp_imu', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
            'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
wifi_cols = ['timestamp_wifi', 'mac_led_1_RSSI', 'mac_led_2_RSSI',
             'mac_led_3_RSSI', 'mac_led_4_RSSI']

gt_df = pd.DataFrame(gt_data, columns=gt_cols)
imu_df = pd.DataFrame(imu_data, columns=imu_cols)
wifi_df = pd.DataFrame(wifi_data, columns=wifi_cols)

print("DataFrames created successfully.")

# --- 3. Time Alignment and Merging ---
print("Aligning data based on timestamps... (This may take a moment)")
imu_df = imu_df.sort_values('timestamp_imu').reset_index(drop=True)
wifi_df = wifi_df.sort_values('timestamp_wifi').reset_index(drop=True)

# Use pandas merge_asof for efficient nearest-neighbor merging
# This is a more robust method than manual index searching
merged_df = pd.merge_asof(
    left=gt_df.sort_values('timestamp'),
    right=imu_df,
    left_on='timestamp',
    right_on='timestamp_imu',
    direction='nearest'
)

merged_df = pd.merge_asof(
    left=merged_df.sort_values('timestamp'),
    right=wifi_df,
    left_on='timestamp',
    right_on='timestamp_wifi',
    direction='nearest'
)
print("Data alignment complete.")

# --- 4. Final Cleanup and Reordering ---
merged_df = merged_df.drop(columns=['timestamp_imu', 'timestamp_wifi'], errors='ignore')

final_cols_order = [
    'timestamp',
    'accelerometer_x', 'accelerometer_y', 'accelerometer_z',
    'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
    'mac_led_1_RSSI', 'mac_led_2_RSSI', 'mac_led_3_RSSI', 'mac_led_4_RSSI',
    'x_coord', 'y_coord', 'z_coord',
    'rot_11', 'rot_12', 'rot_13', 'rot_21', 'rot_22', 'rot_23',
    'rot_31', 'rot_32', 'rot_33'
]
# Make sure to only include columns that actually exist in the merged dataframe
final_cols_order_existing = [col for col in final_cols_order if col in merged_df.columns]
merged_df = merged_df[final_cols_order_existing]

# --- 5. Save the Output ---
output_filename = '0.45_Speed_withoutOB.csv'
merged_df.to_csv(output_filename, index=False)

print("\n--- Process Complete! ---")
print(f"Successfully merged the datasets and saved the result to '{output_filename}'")
print(f"Shape of the final dataset: {merged_df.shape}")
print("You can now find the new CSV file in the same folder.")