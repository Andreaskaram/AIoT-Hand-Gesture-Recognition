from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
import time
import joblib
import numpy as np
from collections import deque
from scipy.signal import butter, lfilter

# --- 1. Load Pre-trained ML Pipeline ---
print("Loading ML models...")
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('best_model.pkl')

# --- 2. Load Config Parameters ---
# Values taken directly from your config.yml
WINDOW_SIZE = 120    # 100Hz = 1.2 seconds of data
OVERLAP = 60         # Slide by 60 samples
FILTER_ORDER = 4
FILTER_CUTOFF = 10   # Hz
SAMPLING_RATE = 100  # Hz

# --- 3. Setup Signal Processing ---
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data, axis=0)
    return y

# Thread-safe buffer for incoming data (Acc X, Y, Z + Gyro X, Y, Z)
data_buffer = deque(maxlen=WINDOW_SIZE)
current_sample = []

def process_and_predict(window_data):
    """Applies your exact ML pipeline to a single live window"""
    # 1. Convert to numpy array (Shape: 120, 6)
    raw_array = np.array(window_data)
    
    # 2. Apply Lowpass Filter
    filtered_array = butter_lowpass_filter(raw_array, FILTER_CUTOFF, SAMPLING_RATE, FILTER_ORDER)
    
    # 3. Flatten the window (Shape: 1, 720) - assuming 6 features * 120 samples
    flattened_window = filtered_array.flatten().reshape(1, -1)
    
    # 4. Scale
    scaled_window = scaler.transform(flattened_window)
    
    # 5. PCA Reduction
    pca_window = pca.transform(scaled_window)
    
    # 6. Predict!
    prediction = model.predict(pca_window)[0]
    print(f"\n>> LIVE PREDICTION: {prediction.upper()} <<\n")

# --- 4. MetaWear Callbacks ---
# We use a simple counter to pair Accel and Gyro readings coming in at 100Hz
sample_sync = {'acc': None, 'gyro': None}

def acc_callback(ctx, data):
    val = parse_value(data)
    sample_sync['acc'] = [val.x, val.y, val.z]
    check_sync()

def gyro_callback(ctx, data):
    val = parse_value(data)
    sample_sync['gyro'] = [val.x, val.y, val.z]
    check_sync()

def check_sync():
    # If we have both readings for this timestamp, append to buffer
    if sample_sync['acc'] is not None and sample_sync['gyro'] is not None:
        combined_features = sample_sync['acc'] + sample_sync['gyro']
        data_buffer.append(combined_features)
        
        # Reset for next sample
        sample_sync['acc'] = None
        sample_sync['gyro'] = None
        
        # Check if window is full and ready for prediction
        if len(data_buffer) == WINDOW_SIZE:
            # Copy buffer to process safely
            window_copy = list(data_buffer)
            process_and_predict(window_copy)
            
            # Pop the oldest samples to create the overlap (Overlap = 60)
            for _ in range(WINDOW_SIZE - OVERLAP):
                data_buffer.popleft()

# --- 5. Connect and Run ---
MAC_ADDRESS = "C0:39:XX:XX:XX:XX" # <-- INSERT YOUR MAC ADDRESS HERE

print(f"Connecting to {MAC_ADDRESS}...")
device = MetaWear(MAC_ADDRESS)
device.connect()
print("Connected! Configuring sensors to 100Hz...")

try:
    # Setup Accelerometer (100Hz, +/- 4g)
    libmetawear.mbl_mw_acc_bmi160_set_odr(device.board, AccBmi160Odr._100Hz)
    libmetawear.mbl_mw_acc_bmi160_set_range(device.board, 4.0)
    libmetawear.mbl_mw_acc_write_acceleration_config(device.board)
    
    # Setup Gyroscope (100Hz, 2000 degrees/s)
    libmetawear.mbl_mw_gyro_bmi160_set_odr(device.board, GyroBmi160Odr._100Hz)
    libmetawear.mbl_mw_gyro_bmi160_set_range(device.board, GyroBmi160Range._2000dps)
    libmetawear.mbl_mw_gyro_write_config(device.board)

    # Subscribe to data streams
    acc_signal = libmetawear.mbl_mw_acc_get_acceleration_data_signal(device.board)
    libmetawear.mbl_mw_datasignal_subscribe(acc_signal, None, FnVoid_VoidP_DataP(acc_callback))
    
    gyro_signal = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(device.board)
    libmetawear.mbl_mw_datasignal_subscribe(gyro_signal, None, FnVoid_VoidP_DataP(gyro_callback))

    # Start sensors
    print("Starting data collection. Perform gestures now! (Press Ctrl+C to stop)")
    libmetawear.mbl_mw_acc_enable_acceleration_sampling(device.board)
    libmetawear.mbl_mw_acc_start(device.board)
    libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(device.board)
    libmetawear.mbl_mw_gyro_bmi160_start(device.board)

    # Keep script running
    while True:
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Safely teardown connection
    libmetawear.mbl_mw_acc_stop(device.board)
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(device.board)
    libmetawear.mbl_mw_gyro_bmi160_stop(device.board)
    libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(device.board)
    
    libmetawear.mbl_mw_datasignal_unsubscribe(acc_signal)
    libmetawear.mbl_mw_datasignal_unsubscribe(gyro_signal)
    
    device.disconnect()
    print("Disconnected safely.")