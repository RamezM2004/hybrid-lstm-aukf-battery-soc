import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# --- Configuration ---
BATTERY_ID = 'B0005'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
FILENAME = os.path.join(DEFAULT_DATA_DIR, f"{BATTERY_ID}.mat")
if not os.path.exists(FILENAME):
    FILENAME = r"C:\Users\USER\OneDrive - GJU\Desktop\4th Year\4.1\Machine Intelligence I\MI Project\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4\B0005.mat"

# Battery Parameters (Thevenin ECM)
R0 = 0.15     # Ohmic Resistance
Rp = 0.1      # Polarization Resistance
Cp = 1000.0   # Polarization Capacitance
Q_capacity = 2.0 * 3600  # Rated Capacity (Amp-seconds)

# --- Physics Models ---
def get_ocv(soc):
    # 6th-order polynomial fit for LCO Li-ion battery (NASA B0005)
    # Updated coefficients to correctly match 4.2V (100%) to 3.2V (0%) range
    soc = np.clip(soc, 0.0, 1.0)
    return (21.72 * soc**6 - 62.65 * soc**5 + 68.39 * soc**4 
            - 35.26 * soc**3 + 8.448 * soc**2 + 0.4475 * soc + 3.2)

def fx(x, dt, u):
    # State Transition: x=[SoC, Vp]
    soc, v_p = x[0], x[1]
    current = u[0]
    
    soc_new = soc - (dt / Q_capacity) * current
    
    if dt > 0:
        tau = Rp * Cp
        exp_val = np.exp(-dt / tau)
        vp_new = exp_val * v_p + Rp * (1 - exp_val) * current
    else:
        vp_new = v_p
        
    return np.array([soc_new, vp_new])

def hx(x, u):
    # Measurement: Terminal Voltage
    soc, v_p = x[0], x[1]
    current = u[0]
    return np.array([get_ocv(soc) - v_p - current * R0])

# --- Data Loader ---

def load_nasa_data(filename):
    if not os.path.exists(filename): 
        print(f"File not found: {filename}")
        return None, None, None
    try:
        data = scipy.io.loadmat(filename, simplify_cells=True)
        cycles = data[BATTERY_ID]['cycle']
        
        for i, cycle in enumerate(cycles):
            if cycle['type'] == 'discharge':
                d = cycle['data']
                time = np.atleast_1d(d['Time'])
                if len(time) < 100: continue # Skip glitches
                
                print(f"Found Data at Cycle {i} (Points: {len(time)})")
                voltage = np.atleast_1d(d['Voltage_measured'])
                current = -np.atleast_1d(d['Current_measured']) # Sign correction
                return time, voltage, current
    except Exception as e: print(e)
    return None, None, None

# --- Main Execution ---

time, meas_voltage, load_current = load_nasa_data(FILENAME)

if time is not None:
    print("Initializing Dual Adaptive UKF...")
    
    points = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2.0, kappa=0)
    kf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=1.0, fx=fx, hx=hx, points=points)
    
    # Init State & Covariance
    kf.x = np.array([1.0, 0.0])
    kf.P *= 0.001
    
    # Baseline Noise
    base_Q = np.diag([1e-5, 1e-4]) 
    kf.Q = base_Q.copy()
    kf.R = np.diag([0.01]) 
    
    # Adaptive Params
    window_alpha = 0.3
    eta_r = 0.5
    
    est_soc, est_voltage, q_history, r_history, nis_history = [], [], [], [], []
    
    print("Running Adaptive Loop...")
    
    for k in range(len(time)):
        # Calculate dynamic dt
        if k == 0: current_dt = 1.0
        else: current_dt = time[k] - time[k-1]
        if current_dt > 20.0: current_dt = 20.0
        
        u = np.array([load_current[k]])
        z = np.array([meas_voltage[k]])
        
        # 1. Predict & 2. Update
        kf.predict(dt=current_dt, u=u)
        kf.update(z, u=u)
        
        # Calculate NIS (Normalized Innovation Squared)
        # NIS = y^T * S^-1 * y
        nis = float(np.dot(kf.y.T, np.linalg.inv(kf.S)).dot(kf.y))
        nis_history.append(nis)

        # 3. Dual Adaptation (IAE)
        innovation = kf.y
        S_theoretical = kf.S
        actual_cov = np.outer(innovation, innovation)
        
        # Adapt R (Measurement Noise)
        S_mismatch = actual_cov - S_theoretical
        r_update = eta_r * np.diag(S_mismatch)
        new_R = np.clip(np.diag(kf.R) + r_update, 0.001, 10.0)
        kf.R = np.diag(new_R)

        # Adapt Q (Process Noise)
        trace_actual = np.trace(actual_cov)
        trace_theory = np.trace(S_theoretical)
        
        if trace_actual > trace_theory:
            scale = min(trace_actual / trace_theory, 1000.0)
            kf.Q = window_alpha * kf.Q + (1 - window_alpha) * (base_Q * scale)
        else:
            kf.Q = window_alpha * kf.Q + (1 - window_alpha) * base_Q
            
        est_soc.append(kf.x[0])
        est_voltage.append(hx(kf.x, u)[0])
        q_history.append(kf.Q[0,0])
        r_history.append(kf.R[0,0])
        
    print("Simulation Complete.")

    # --- Plotting Main Performance ---
    plt.figure(figsize=(12, 14))
    
    plt.subplot(4, 1, 1)
    plt.plot(time, meas_voltage, 'k', label='True Voltage', linewidth=1.5)
    plt.plot(time, est_voltage, 'r--', label='Dual Adaptive Estimate', linewidth=1.5)
    plt.ylabel('Voltage [V]'); plt.title(f'Dual Adaptive UKF Performance')
    plt.legend(); plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time, np.array(est_soc) * 100, 'b', label='Estimated SoC', linewidth=2)
    plt.ylabel('SoC [%]'); plt.title('State of Charge')
    plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time, q_history, 'g', label='Adaptive Q (Process Noise)', linewidth=1.5)
    plt.ylabel('Q Value'); plt.title('Adaptation Response: Q')
    plt.legend(); plt.yscale('log'); plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(time, r_history, 'c', label='Adaptive R (Meas. Noise)', linewidth=1.5)
    plt.ylabel('R Value'); plt.xlabel('Time [s]'); plt.title('Adaptation Response: R')
    plt.legend(); plt.yscale('log'); plt.grid(True)
    
    plt.tight_layout()

    # --- Plotting NIS Separately (Kept as requested previously) ---
    plt.figure(figsize=(12, 6))
    plt.plot(time, nis_history, 'm', label='NIS', linewidth=1.0)
    
    # Add bounds for interpretation
    plt.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, label='Theoretical Mean (1.0)')
    plt.axhline(y=3.84, color='r', linestyle='--', linewidth=1.5, label='95% Confidence Limit (3.84)')
    
    plt.ylabel('NIS Value')
    plt.xlabel('Time [s]')
    plt.title('Normalized Innovation Squared (NIS) Consistency Check')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

else:
    print("Error: Could not load data.")