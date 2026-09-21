# AI-assisted UKF for Battery SoC Estimation with LSTM-based Knee Detection
# Leakage-fixed version
# By Ramez Al-Masadeh

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

plt.rcParams.update({"font.size": 9})


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"C:\Users\USER\OneDrive - GJU\Desktop\4th Year\4.1\Machine Intelligence I\MI Project\5.+Battery+Data+Set\5. Battery Data Set\1. BatteryAgingARC-FY08Q4"

BATTERY_FILES = {
    "B0005": os.path.join(DATA_DIR, "B0005.mat"),
    "B0006": os.path.join(DATA_DIR, "B0006.mat"),
    "B0007": os.path.join(DATA_DIR, "B0007.mat"),
    "B0018": os.path.join(DATA_DIR, "B0018.mat"),
}

TEST_BATTERY_ID = "B0007"

# Thevenin model parameters
R0 = 0.15
Rp = 0.10
Cp = 1000.0

# LSTM
SEQ_LEN = 25
LSTM_INPUT_SIZE = 3          
BATCH_SIZE = 512
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UKF noise matrices
BASE_Q = np.diag([1e-8, 1e-6, 1e-8])   # states: [SoC, Vp, bias]
MEAS_R = np.array([[1e-4]])            # voltage measurement variance

# Adaptive Q settings
MAX_Q_SCALE = 50.0
MIN_Q_SCALE = 1.0
LOG_SMOOTH = 0.85

# TRUE NIS settings
NIS_TARGET = 1.0
NIS_GAIN = 0.50
NIS_CLIP = 25.0

# Knee probability influence
PK_GAIN = 0.80
PK_ON = 0.55
PK_OFF = 0.40

# Shape Q across states
VP_GAIN = 1.0
SOC_GAIN = 0.25
BIAS_GAIN = 0.35

# Data filtering
MIN_POINTS = max(SEQ_LEN + 10, 60)

# Knee pseudo-label settings
KNEE_Z_THRESH = -6.0
KNEE_PERSIST_W = 7
KNEE_PERSIST_FRAC = 0.6
IGNORE_FIRST_SECONDS = 60.0
PRE_KNEE_SECONDS = 180.0

# Evaluation
EVAL_CYCLES = 10
PLOT_ONE_CYCLE = True

# Debug
PRINT_CAPACITY_NAN_RATE = True
PRINT_Q_STATS_PER_CYCLE = False


# ============================================================
# BATTERY MODEL
# ============================================================

def get_ocv(soc: float) -> float:
    """
    OCV polynomial.
    Later you can replace this with your verified CALCE Samsung OCV curve.
    """
    s = np.clip(soc, 0.0, 1.0)
    return float(
        21.72 * s**6
        - 62.65 * s**5
        + 68.39 * s**4
        - 35.26 * s**3
        + 8.448 * s**2
        + 0.4475 * s
        + 3.20
    )


def fx3(x, dt, u):
    """
    UKF state transition.

    State:
        x = [SoC, Vp, bias]

    Input:
        u = [I]
        I positive = discharge
    """
    soc, vp, b = float(x[0]), float(x[1]), float(x[2])
    I = float(u[0])

    soc_new = soc - (dt / fx3.Q_capacity_C) * I
    soc_new = np.clip(soc_new, 0.0, 1.0)

    tau = Rp * Cp
    a = np.exp(-dt / tau)
    vp_new = a * vp + Rp * (1.0 - a) * I

    return np.array([soc_new, vp_new, b], dtype=float)


fx3.Q_capacity_C = 2.0 * 3600.0


def hx3(x, u):
    """
    UKF measurement model.

    Voltage = OCV(SoC) - Vp - I*R0 + bias
    """
    soc, vp, b = float(x[0]), float(x[1]), float(x[2])
    I = float(u[0])

    v_pred = get_ocv(soc) - vp - I * R0 + b

    return np.array([v_pred], dtype=float)


def init_ukf():
    points = MerweScaledSigmaPoints(
        n=3,
        alpha=0.3,
        beta=2.0,
        kappa=0
    )

    kf = UnscentedKalmanFilter(
        dim_x=3,
        dim_z=1,
        dt=1.0,
        fx=fx3,
        hx=hx3,
        points=points
    )

    kf.x = np.array([1.0, 0.0, 0.0], dtype=float)
    kf.P = np.diag([1e-3, 1e-3, 1e-2])
    kf.Q = BASE_Q.copy()
    kf.R = MEAS_R.copy()

    return kf


def safe_dt(tk, tkm1) -> float:
    dt = float(tk - tkm1)

    if not np.isfinite(dt) or dt <= 1e-6:
        return 1.0

    return dt


# ============================================================
# LSTM KNEE DETECTOR
# ============================================================

class KneeDetectorLSTM(nn.Module):
    """
    Leakage-free LSTM.

    Inputs:
        [Voltage, Current, dV/dt]

    Output:
        knee logit
    """

    def __init__(self, input_size=LSTM_INPUT_SIZE, hidden_size=32, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ============================================================
# DATA IO
# ============================================================

def check_files(files: dict) -> dict:
    ok = {}

    for bid, path in files.items():
        if os.path.exists(path):
            ok[bid] = path
            print(f"[OK]   {bid}: {path}")
        else:
            print(f"[MISS] {bid}: {path}")

    return ok


def _to_float_scalar(x, default=np.nan) -> float:
    try:
        a = np.asarray(x).squeeze()

        if a.size == 1:
            return float(a)

        return float(a[-1])

    except Exception:
        return float(default)


def load_discharge_cycles(mat_path: str, battery_id: str, min_points=60):
    """
    Loads discharge cycles from NASA battery .mat files.

    Current convention:
        current returned here is positive during discharge.
    """
    data = scipy.io.loadmat(mat_path, simplify_cells=True)
    cycles = data[battery_id]["cycle"]

    out = []

    for idx, cyc in enumerate(cycles):
        if cyc.get("type") != "discharge":
            continue

        d = cyc["data"]

        t = np.atleast_1d(d["Time"]).astype(float)
        v = np.atleast_1d(d["Voltage_measured"]).astype(float)

        # NASA current may be negative during discharge, so convert:
        i = -np.atleast_1d(d["Current_measured"]).astype(float)

        cap_ah = np.nan

        if isinstance(d, dict) and ("Capacity" in d):
            cap_ah = _to_float_scalar(d["Capacity"], default=np.nan)

        if not np.isfinite(cap_ah):
            for key in ["Capacity", "capacity", "Qd", "Qdlin", "Ah"]:
                if key in cyc:
                    cap_ah = _to_float_scalar(cyc.get(key), default=np.nan)

                    if np.isfinite(cap_ah):
                        break

        if len(t) >= min_points:
            out.append({
                "battery_id": battery_id,
                "cycle_index": idx,
                "time": t,
                "voltage": v,
                "current": i,
                "capacity_ah": cap_ah
            })

    return out


# ============================================================
# REFERENCE SOC
# ============================================================

def soc_ref_from_cycle_capacity(time, current, capacity_ah):
    """
    Reference SoC from Coulomb counting.

    This is used for evaluation and labels,
    but NOT used as LSTM input anymore.
    """
    t = np.asarray(time, float)
    I = np.asarray(current, float)

    dt = np.diff(t, prepend=t[0])
    good = np.isfinite(dt) & (dt > 1e-6)

    if np.any(good):
        dt[~good] = np.median(dt[good])
    else:
        dt[~good] = 1.0

    qd_ah = np.cumsum(I * dt) / 3600.0

    if np.isfinite(capacity_ah) and capacity_ah > 1e-6:
        q_cycle = float(capacity_ah)
    else:
        q_cycle = float(qd_ah[-1])

    if not np.isfinite(q_cycle) or q_cycle <= 1e-6:
        q_cycle = 1.0

    soc = 1.0 - (qd_ah / q_cycle)
    soc = np.clip(soc, 0.0, 1.0)

    return soc, q_cycle


# ============================================================
# KNEE PSEUDO LABELS
# ============================================================

def knee_labels_from_voltage(time, voltage):
    """
    Creates pseudo-labels for voltage knee/collapse region.

    This is not true supervised labeling, but a rule-based approximation.
    """
    t = np.asarray(time, float)
    v = np.asarray(voltage, float)

    dt = np.diff(t, prepend=t[0])
    good = np.isfinite(dt) & (dt > 1e-6)

    if np.any(good):
        dt[~good] = np.median(dt[good])
    else:
        dt[~good] = 1.0

    dVdt = np.diff(v, prepend=v[0]) / dt

    n = len(v)
    base = dVdt[:max(10, int(0.3 * n))]

    med = np.median(base)
    mad = np.median(np.abs(base - med)) + 1e-9

    z = (dVdt - med) / (1.4826 * mad)

    knee = (z < KNEE_Z_THRESH).astype(np.float32)

    if KNEE_PERSIST_W > 1:
        k = np.convolve(knee, np.ones(KNEE_PERSIST_W), mode="same")
        knee = (k >= (KNEE_PERSIST_FRAC * KNEE_PERSIST_W)).astype(np.float32)

    knee[t < (t[0] + IGNORE_FIRST_SECONDS)] = 0.0

    if PRE_KNEE_SECONDS > 0 and np.any(knee > 0):
        start = int(np.where(knee > 0)[0][0])
        t0 = t[start]

        knee[(t >= (t0 - PRE_KNEE_SECONDS)) & (t <= t0)] = 1.0

    return knee


# ============================================================
# SEQUENCE CREATION
# ============================================================

def make_sequences(time, voltage, current, knee, seq_len):
    """
    Create LSTM sequences without leakage.

    Inputs:
        Voltage
        Current
        dV/dt

    Target:
        knee label
    """
    t = np.asarray(time, float)
    v = np.asarray(voltage, float)
    i = np.asarray(current, float)
    y = np.asarray(knee, np.float32)

    dt = np.diff(t, prepend=t[0])
    good = np.isfinite(dt) & (dt > 1e-6)

    if np.any(good):
        dt[~good] = np.median(dt[good])
    else:
        dt[~good] = 1.0

    dVdt = np.diff(v, prepend=v[0]) / dt

    # FIXED: no soc_ref here
    feats = np.column_stack([
        v,
        i,
        dVdt
    ]).astype(np.float32)

    X, Y = [], []

    for k in range(seq_len, len(feats)):
        X.append(feats[k - seq_len:k])
        Y.append(y[k])

    if not X:
        return (
            np.zeros((0, seq_len, LSTM_INPUT_SIZE), np.float32),
            np.zeros((0, 1), np.float32)
        )

    return (
        np.array(X, np.float32),
        np.array(Y, np.float32).reshape(-1, 1)
    )


def build_train_test(valid_files: dict, test_battery: str):
    train_cycles = []
    test_cycles = []

    for bid, path in valid_files.items():
        cycles = load_discharge_cycles(path, bid, min_points=MIN_POINTS)

        print(f"{bid}: {len(cycles)} discharge cycles loaded")

        if bid == test_battery:
            test_cycles.extend(cycles)
        else:
            train_cycles.extend(cycles)

    if PRINT_CAPACITY_NAN_RATE:
        caps = [c["capacity_ah"] for c in train_cycles + test_cycles]
        nan_rate = np.mean([not np.isfinite(x) for x in caps]) if caps else 1.0
        print(f"[Capacity] NaN rate: {nan_rate * 100:.1f}%")

    Xtr_l, Ytr_l = [], []

    for cyc in train_cycles:
        t = cyc["time"]
        v = cyc["voltage"]
        i = cyc["current"]

        knee = knee_labels_from_voltage(t, v)

        # FIXED: no soc_ref passed into make_sequences
        X, Y = make_sequences(t, v, i, knee, SEQ_LEN)

        if len(X):
            Xtr_l.append(X)
            Ytr_l.append(Y)

    Xte_l, Yte_l = [], []

    for cyc in test_cycles:
        t = cyc["time"]
        v = cyc["voltage"]
        i = cyc["current"]

        knee = knee_labels_from_voltage(t, v)

        # FIXED: no soc_ref passed into make_sequences
        X, Y = make_sequences(t, v, i, knee, SEQ_LEN)

        if len(X):
            Xte_l.append(X)
            Yte_l.append(Y)

    if not Xtr_l:
        raise ValueError("No training sequences created.")

    if not Xte_l:
        raise ValueError("No testing sequences created.")

    Xtr = np.concatenate(Xtr_l, axis=0)
    Ytr = np.concatenate(Ytr_l, axis=0)

    Xte = np.concatenate(Xte_l, axis=0)
    Yte = np.concatenate(Yte_l, axis=0)

    return (Xtr, Ytr), (Xte, Yte), test_cycles


# ============================================================
# TRAIN / EVALUATE CLASSIFIER
# ============================================================

def train_classifier(Xtr, Ytr):
    scaler = StandardScaler()

    Xtr_n = scaler.fit_transform(
        Xtr.reshape(-1, LSTM_INPUT_SIZE)
    ).reshape(Xtr.shape)

    model = KneeDetectorLSTM(input_size=LSTM_INPUT_SIZE).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    y = Ytr.reshape(-1)

    pos = max(int(np.sum(y == 1)), 1)
    neg = max(int(np.sum(y == 0)), 1)

    pos_weight = torch.tensor(
        [neg / pos],
        dtype=torch.float32,
        device=DEVICE
    )

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(
        torch.tensor(Xtr_n, dtype=torch.float32),
        torch.tensor(Ytr, dtype=torch.float32)
    )

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for e in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()

            logits = model(xb)
            loss = crit(logits, yb)

            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        if e % 5 == 0 or e == EPOCHS - 1:
            print(f"Epoch {e:02d}: loss={total_loss / len(dl):.4f}")

    return model, scaler


@torch.no_grad()
def eval_classifier(model, scaler, Xte, Yte, threshold=0.5):
    Xte_n = scaler.transform(
        Xte.reshape(-1, LSTM_INPUT_SIZE)
    ).reshape(Xte.shape)

    xb = torch.tensor(Xte_n, dtype=torch.float32).to(DEVICE)

    logits = model(xb).cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))

    y_true = Yte.reshape(-1).astype(int)
    y_pred = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    return float(precision), float(recall), float(f1)


# ============================================================
# METRICS
# ============================================================

def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    return float(np.mean(np.abs(a - b)))


def detect_voltage_knee_time(time, voltage, smooth_window=21):
    """
    Knee detection using voltage curvature.

    Returns:
        knee_time, knee_idx, smoothed_voltage, dvdt, d2vdt2
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)

    w = int(smooth_window)

    if w < 5:
        w = 5

    if w % 2 == 0:
        w += 1

    if len(v) < w:
        w = max(5, (len(v) // 2) * 2 + 1)

    kernel = np.ones(w) / w
    v_smooth = np.convolve(v, kernel, mode="same")

    dvdt = np.gradient(v_smooth, t)
    d2vdt2 = np.gradient(dvdt, t)

    knee_idx = int(np.argmin(d2vdt2))

    return (
        float(t[knee_idx]),
        knee_idx,
        v_smooth,
        dvdt,
        d2vdt2
    )


# ============================================================
# UKF + LEAKAGE-FREE LSTM ADAPTIVE Q
# ============================================================

@torch.no_grad()
def run_ukf_adaptiveQ_trueNIS(time, voltage, current, soc_ref, q_cycle_ah, model, scaler):
    """
    UKF with adaptive Q.

    Leakage fix:
        LSTM input during runtime is [V, I, dV/dt]
        It does NOT use soc_ref.
    """
    t = np.asarray(time, float)
    v = np.asarray(voltage, float)
    i = np.asarray(current, float)
    soc_ref = np.asarray(soc_ref, float)

    fx3.Q_capacity_C = float(q_cycle_ah) * 3600.0

    kf = init_ukf()

    feat_buf = []
    prev_v = float(v[0])

    knee_state = 0
    p_smooth = 0.0

    log_q = 0.0
    nis_prev = 1.0

    est_v = []
    est_soc = []

    q_scales = []
    p_hist = []
    nis_hist = []

    for k in range(len(t)):
        V = float(v[k])
        I = float(i[k])

        z = np.array([V], dtype=float)
        u = np.array([I], dtype=float)

        if k == 0:
            dt = 1.0
        else:
            dt = safe_dt(t[k], t[k - 1])

        # Predict
        kf.predict(dt=dt, u=u)

        # LSTM feature, leakage-free
        dvdt = (V - prev_v) / dt
        prev_v = V

        feat = np.array([V, I, dvdt], np.float32)
        feat_n = scaler.transform(feat.reshape(1, -1))[0]
        feat_buf.append(feat_n)

        # Predict knee probability
        p_knee = 0.0

        if len(feat_buf) >= SEQ_LEN:
            seq = np.array(feat_buf[-SEQ_LEN:], np.float32)
            seq_t = torch.tensor(seq).unsqueeze(0).to(DEVICE)

            logit = float(model(seq_t).cpu().numpy().reshape(-1)[0])
            p_knee = float(1.0 / (1.0 + np.exp(-logit)))

        # Smooth knee probability
        p_smooth = 0.85 * p_smooth + 0.15 * p_knee

        # Hysteresis
        if knee_state == 0 and p_smooth > PK_ON:
            knee_state = 1
        elif knee_state == 1 and p_smooth < PK_OFF:
            knee_state = 0

        p_eff = p_smooth if knee_state else 0.0

        # NIS-based adaptive Q
        nis_ratio = float(np.clip(nis_prev / NIS_TARGET, 0.0, NIS_CLIP))
        nis_term = max(0.0, nis_ratio - 1.0)

        q_des = 1.0
        q_des *= (1.0 + NIS_GAIN * nis_term)
        q_des *= (1.0 + PK_GAIN * p_eff)
        q_des = float(np.clip(q_des, MIN_Q_SCALE, MAX_Q_SCALE))

        # Smooth Q scale in log space
        log_q_des = float(np.log(q_des + 1e-12))
        log_q = LOG_SMOOTH * log_q + (1.0 - LOG_SMOOTH) * log_q_des

        q_scale = float(np.clip(np.exp(log_q), MIN_Q_SCALE, MAX_Q_SCALE))

        # Shape Q across states
        Qk = BASE_Q.copy()
        Qk[1, 1] *= (1.0 + VP_GAIN * (q_scale - 1.0))
        Qk[0, 0] *= (1.0 + SOC_GAIN * (q_scale - 1.0))
        Qk[2, 2] *= (1.0 + BIAS_GAIN * (q_scale - 1.0))

        kf.Q = Qk

        # Update
        kf.update(z, u=u)

        # TRUE NIS
        try:
            y = float(np.asarray(kf.y).reshape(-1)[0])
            S = float(np.asarray(kf.S).reshape(-1)[0])
            nis = (y * y) / (S + 1e-12)
        except Exception:
            nis = np.nan

        if not np.isfinite(nis):
            nis = nis_prev

        nis_prev = float(nis)

        nis_hist.append(float(nis))

        est_v.append(float(hx3(kf.x, u)[0]))
        est_soc.append(float(np.clip(kf.x[0], 0.0, 1.0)))

        q_scales.append(q_scale)
        p_hist.append(p_smooth)

    if PRINT_Q_STATS_PER_CYCLE:
        q_arr = np.array(q_scales, dtype=float)

        print(
            f"[Q] mean={q_arr.mean():.2f} "
            f"max={q_arr.max():.2f} "
            f"pct>2={100 * np.mean(q_arr > 2):.1f}%"
        )

    return (
        np.array(est_v, float),
        np.array(est_soc, float),
        np.array(q_scales, float),
        np.array(p_hist, float),
        np.array(nis_hist, float)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print("Leakage-fixed LSTM input: [Voltage, Current, dV/dt]")
    print("Reference SoC is used only for evaluation, not as LSTM input.")

    valid = check_files(BATTERY_FILES)

    if not valid or TEST_BATTERY_ID not in valid:
        print("Missing files or test battery not found.")
        return

    (Xtr, Ytr), (Xte, Yte), test_cycles = build_train_test(
        valid,
        TEST_BATTERY_ID
    )

    print(f"Train sequences: {len(Xtr):,}")
    print(f"Test sequences:  {len(Xte):,}")
    print(f"Input shape:     {Xtr.shape}")

    model, scaler = train_classifier(Xtr, Ytr)

    precision, recall, f1 = eval_classifier(model, scaler, Xte, Yte)

    print(
        f"[Classifier] "
        f"Precision={precision:.3f} "
        f"Recall={recall:.3f} "
        f"F1={f1:.3f}"
    )

    n = min(EVAL_CYCLES, len(test_cycles))

    if n == 0:
        print("No test cycles.")
        return

    soc_rmse_all = []
    soc_mae_all = []

    v_rmse_all = []
    v_mae_all = []
    v_max_all = []

    knee_time_err_all = []

    nis_mean_all = []
    nis_med_all = []
    nis_pct95_all = []

    for j in range(n):
        cyc = test_cycles[j]

        t = cyc["time"]
        v = cyc["voltage"]
        i = cyc["current"]

        soc_ref, q_cycle_ah = soc_ref_from_cycle_capacity(
            t,
            i,
            cyc.get("capacity_ah", np.nan)
        )

        va, sa, q_sc, p_sc, nis_sc = run_ukf_adaptiveQ_trueNIS(
            t,
            v,
            i,
            soc_ref,
            q_cycle_ah,
            model,
            scaler
        )

        soc_rmse_all.append(rmse(sa, soc_ref))
        soc_mae_all.append(mae(sa, soc_ref))

        v_rmse_all.append(rmse(va, v))
        v_mae_all.append(mae(va, v))
        v_max_all.append(float(np.max(np.abs(va - v))))

        t_knee_meas, _, _, _, _ = detect_voltage_knee_time(t, v)
        t_knee_est, _, _, _, _ = detect_voltage_knee_time(t, va)

        knee_time_err_all.append(float(t_knee_est - t_knee_meas))

        nis_sc = np.asarray(nis_sc, float)
        nis_sc = nis_sc[np.isfinite(nis_sc)]

        if len(nis_sc) == 0:
            nis_mean_all.append(np.nan)
            nis_med_all.append(np.nan)
            nis_pct95_all.append(np.nan)
        else:
            nis_mean_all.append(float(np.mean(nis_sc)))
            nis_med_all.append(float(np.median(nis_sc)))
            nis_pct95_all.append(float(100.0 * np.mean(nis_sc <= 3.84)))

        print(
            f"[Cycle {cyc['cycle_index']:03d}] "
            f"SoC_RMSE={soc_rmse_all[-1]:.5f} "
            f"SoC_MAE={soc_mae_all[-1]:.5f} | "
            f"V_RMSE={v_rmse_all[-1]:.4f} V "
            f"V_MAE={v_mae_all[-1]:.4f} V "
            f"V_MAX={v_max_all[-1]:.4f} V | "
            f"Knee_dt={knee_time_err_all[-1]:+.2f} s | "
            f"NIS_mean={nis_mean_all[-1]:.3f} "
            f"pct<=3.84={nis_pct95_all[-1]:.1f}%"
        )

    def _nanmean(x):
        return float(np.nanmean(np.asarray(x, float)))

    def _nanmedian(x):
        return float(np.nanmedian(np.asarray(x, float)))

    print("\n==================== Evaluation Metrics Averaged ====================")

    print("\n1) SoC estimation error")
    print(f"   SoC RMSE : {_nanmean(soc_rmse_all):.6f} fraction | {_nanmean(soc_rmse_all) * 100:.3f}%")
    print(f"   SoC MAE  : {_nanmean(soc_mae_all):.6f} fraction | {_nanmean(soc_mae_all) * 100:.3f}%")

    print("\n2) Terminal voltage prediction error")
    print(f"   Voltage RMSE : {_nanmean(v_rmse_all):.6f} V")
    print(f"   Voltage MAE  : {_nanmean(v_mae_all):.6f} V")
    print(f"   Voltage MAX  : {_nanmean(v_max_all):.6f} V")

    print("\n3) Voltage knee timing")
    print(f"   Mean knee time error   : {_nanmean(knee_time_err_all):+.2f} s")
    print(f"   Median knee time error : {_nanmedian(knee_time_err_all):+.2f} s")

    print("\n4) NIS consistency")
    print(f"   Mean NIS: {_nanmean(nis_mean_all):.4f}")
    print(f"   Median NIS: {_nanmean(nis_med_all):.4f}")
    print(f"   % NIS <= 3.84: {_nanmean(nis_pct95_all):.2f}%")

    print("=====================================================================\n")

    # Plot one test cycle
    if PLOT_ONE_CYCLE and len(test_cycles) > 0:
        cyc = test_cycles[0]

        t = cyc["time"]
        v = cyc["voltage"]
        i = cyc["current"]

        soc_ref, q_cycle_ah = soc_ref_from_cycle_capacity(
            t,
            i,
            cyc.get("capacity_ah", np.nan)
        )

        knee = knee_labels_from_voltage(t, v).astype(bool)

        va, sa, q_sc, p_sc, nis_sc = run_ukf_adaptiveQ_trueNIS(
            t,
            v,
            i,
            soc_ref,
            q_cycle_ah,
            model,
            scaler
        )

        t_knee_meas, _, _, _, _ = detect_voltage_knee_time(t, v)
        t_knee_est, _, _, _, _ = detect_voltage_knee_time(t, va)

        fig = plt.figure(figsize=(14, 10))

        fig.suptitle(
            f"Test {TEST_BATTERY_ID} | Cycle {cyc['cycle_index']} | "
            f"Q_cycle={q_cycle_ah:.3f} Ah | Leakage-fixed LSTM",
            fontsize=10
        )

        ax1 = plt.subplot(5, 1, 1)
        ax1.plot(t, v, label="Measured V", linewidth=1.4)
        ax1.plot(t, va, "--", label="UKF Adaptive-Q Vhat", linewidth=1.1)
        ax1.axvline(t_knee_meas, linestyle=":", linewidth=1.2, label=f"Measured knee @ {t_knee_meas:.1f}s")
        ax1.axvline(t_knee_est, linestyle=":", linewidth=1.2, label=f"Estimated knee @ {t_knee_est:.1f}s")
        ax1.grid(True)
        ax1.set_ylabel("Voltage (V)")
        ax1.legend(fontsize=8)

        ax2 = plt.subplot(5, 1, 2)
        ax2.plot(t, soc_ref * 100, label="SoC_ref", linewidth=1.2)
        ax2.plot(t, sa * 100, "--", label="SoC UKF", linewidth=1.2)
        ax2.grid(True)
        ax2.set_ylabel("SoC (%)")
        ax2.legend(fontsize=8)

        ax3 = plt.subplot(5, 1, 3)
        ax3.plot(t, p_sc, label="p_knee smoothed", linewidth=1.2)
        ax3.axhline(0.5, linestyle="--", linewidth=1.0)

        if np.any(knee):
            ax3.fill_between(
                t,
                0,
                1,
                where=knee,
                alpha=0.10,
                transform=ax3.get_xaxis_transform(),
                label="Knee label region"
            )

        ax3.grid(True)
        ax3.set_ylabel("p")
        ax3.legend(fontsize=8)

        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(t, nis_sc, label="TRUE NIS", linewidth=1.2)
        ax4.axhline(1.0, linestyle="--", linewidth=1.0, label="Target ≈ 1")
        ax4.axhline(3.84, linestyle="--", linewidth=1.0, label="95% bound")
        ax4.grid(True)
        ax4.set_ylabel("NIS")
        ax4.legend(fontsize=8)

        ax5 = plt.subplot(5, 1, 5)
        ax5.plot(t, q_sc, label="Q scale", linewidth=1.2)

        if np.any(knee):
            ax5.fill_between(
                t,
                0,
                1,
                where=knee,
                alpha=0.10,
                transform=ax5.get_xaxis_transform()
            )

        ax5.set_yscale("log")
        ax5.grid(True)
        ax5.set_ylabel("Q scale")
        ax5.set_xlabel("Time (s)")
        ax5.legend(fontsize=8)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()