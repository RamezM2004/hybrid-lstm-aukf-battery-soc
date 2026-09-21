import numpy as np
import matplotlib.pyplot as plt

# --- 1. SCENARIO GENERATION ---
def generate_data(steps=100):
    # State: [x, y, vx, vy]
    dt = 1.0
    gt_data = []
    measurements = []
    
    x, y = 0, 0
    vx, vy = 2, 0 # Initial velocity: Moving East
    
    for t in range(steps):
        # THE MANEUVER: At t=40, sharp turn North
        if 40 < t < 50:
            vx = 0.5
            vy = 2.0
        elif t >= 50:
            vx = 0
            vy = 2
            
        # Update Ground Truth
        x += vx * dt
        y += vy * dt
        gt_data.append([x, y])
        
        # Add Noise to generate Measurements (GPS)
        noise_level = 2.5
        mx = x + np.random.normal(0, noise_level)
        my = y + np.random.normal(0, noise_level)
        measurements.append([mx, my])
        
    return np.array(gt_data), np.array(measurements)

# --- 2. STANDARD KF CLASS ---
class StandardKF:
    def __init__(self):
        self.x = np.zeros((4, 1)) 
        self.P = np.eye(4) * 10
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]) # CV Model
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # GPS measures x, y
        self.Q = np.eye(4) * 0.1 # Low process noise (trusts model)
        self.R = np.eye(2) * 2.5 # Measurement noise
        
    def step(self, z):
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        z = z.reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.flatten()

# --- 3. ADAPTIVE KF CLASS (Akhlaghi 2017) ---
class AdaptiveKF:
    def __init__(self):
        self.x = np.zeros((4, 1)) 
        self.P = np.eye(4) * 10
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Initial Guesses
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 2.5
        
        # Forgetting Factor (Akhlaghi's alpha)
        self.alpha = 0.9 

    def step(self, z):
        # -- Predict --
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # -- Update --
        z = z.reshape(2, 1)
        d_k = z - self.H @ self.x # Innovation (Pre-update error)
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ d_k
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # -- ADAPTATION STEP (Akhlaghi) --
        epsilon = z - self.H @ self.x # Residual (Post-update error)
        
        # Update R (using Residuals)
        R_sample = epsilon @ epsilon.T + self.H @ self.P @ self.H.T
        self.R = (1 - self.alpha) * self.R + self.alpha * R_sample
        
        # Update Q (using Innovation and K)
        Q_sample = K @ (d_k @ d_k.T) @ K.T
        self.Q = (1 - self.alpha) * self.Q + self.alpha * Q_sample
        
        return self.x.flatten(), np.trace(self.Q)

# --- 4. RUN SIMULATION ---
gt, meas = generate_data()
skf = StandardKF()
akf = AdaptiveKF()

skf_path = []
akf_path = []
q_history = []

for z in meas:
    skf_path.append(skf.step(z))
    pos, q_trace = akf.step(z)
    akf_path.append(pos)
    q_history.append(q_trace)

skf_path = np.array(skf_path)
akf_path = np.array(akf_path)

# --- 5. PLOT RESULTS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Trajectory
ax1.plot(gt[:,0], gt[:,1], 'k--', label='Ground Truth', linewidth=2)
ax1.scatter(meas[:,0], meas[:,1], c='gray', s=10, alpha=0.5, label='Noisy GPS')
ax1.plot(skf_path[:,0], skf_path[:,1], 'r-', label='Standard KF')
ax1.plot(akf_path[:,0], akf_path[:,1], 'g-', linewidth=2, label='Adaptive KF (Akhlaghi)')
ax1.set_title("2D Tracking: Sudden Turn Scenario")
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")
ax1.legend()
ax1.grid(True)

# Plot 2: Adaptation of Q
ax2.plot(q_history, 'b-')
ax2.set_title("Adaptive Q (Process Noise) over Time")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Trace of Q Matrix")
ax2.axvline(x=40, color='r', linestyle='--', label='Turn Start')
ax2.axvline(x=50, color='r', linestyle='--', label='Turn End')
ax2.legend()
ax2.grid(True)

plt.show()