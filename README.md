# Hybrid Physics-AI State-of-Charge (SoC) Estimation for Lithium-Ion Batteries
### Two-Dataset LSTM and Adaptive Unscented Kalman Filtering Study

**Author:** Ramez Al-Masadeh  
**Advisor:** Dr.-Ing. Sahar Qaadan  
**Institution:** German Jordanian University (GJU) — School of Applied Technical Sciences  
**Paper Manuscript:** `paper/Samsung_Hybrid_LSTM_AUKF_IEEE_Paper.tex`

---

## Project Overview

Precise State-of-Charge (SoC) estimation is important for battery management systems (BMS) in electric vehicles and energy storage systems. This repository documents two related but different research tracks. The first track uses NASA battery aging data to test whether an LSTM can help an Adaptive Unscented Kalman Filter respond to the end-of-discharge voltage knee. The second track uses CALCE Samsung INR18650-20R dynamic drive-cycle data to fuse an LSTM SoC estimate with a physics-based AUKF.

The project is intentionally reported as two datasets and two estimator roles:

1. **Project I - NASA PCoE aging dataset:** B0005, B0006, B0007, and B0018 discharge cycles are used with a battery-level split. The LSTM is a pseudo-knee-region classifier, not a direct SoC regressor. It acts as a trigger for process-noise adaptation in an AUKF. This track is useful as a leakage-controlled prototype and diagnostic study; it does not represent the strongest final estimator.
2. **Project II - CALCE Samsung INR18650-20R dataset:** Dynamic profiles are split by drive-cycle family. DST and BJDST are used for training, FUDS for validation, and US06 is held out for final testing at 0 C, 25 C, and 45 C. Here, the LSTM directly predicts SoC and the AUKF fuses that value with terminal-voltage measurements using a first-order Thevenin ECM.

---

## Key Scientific Innovations

- **Leakage-controlled NASA evaluation:** Uses a battery-level split and avoids same-cycle test capacity in UKF propagation. The NASA result is reported honestly as a prototype/negative-result audit because model mismatch and pseudo-OCV limitations affected consistency.
- **Profile-separated Samsung evaluation:** Uses a drive-cycle split to test generalization on a completely held-out US06 profile family.
- **Physics + AI fusion:** The Samsung estimator uses LSTM SoC as a learned measurement while voltage is constrained by a Thevenin ECM inside the AUKF.
- **Innovation monitoring:** Reports Normalized Innovation Squared (NIS) to check whether the filter is statistically consistent, rather than reporting only SoC error.

---

## Validation Results (US06 Driving Cycle)

| Estimator Architecture | SoC RMSE (%) | SoC MAE (%) | Max Error (%) | Mean NIS | 95% Chi-Square Bound Conformity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standalone LSTM | 1.239% | 0.937% | 6.563% | - | - |
| **Hybrid LSTM-AUKF** | **1.114%** | **0.812%** | **5.344%** | **1.671** | **96.01%** |

The Samsung hybrid estimator reduces maximum error by **18.6%** and keeps the 2-D NIS close to its expected range. The NASA track remains in the repository because it explains the method development, the leakage controls, and the limitations that led to the stronger Samsung fusion design.

---

## Repository Structure

```text
├── paper/
│   └── Samsung_Hybrid_LSTM_AUKF_IEEE_Paper.tex # IEEE Transactions manuscript
├── src/
│   ├── BatterySoC_LSTM.py                     # PyTorch sequence model
│   ├── AdpNasaDataSet.py                      # Data preprocessing & loader
│   ├── AKF_Test.py                            # Adaptive Kalman Filter testing
│   └── KFTest1.m                              # MATLAB validation scripts
├── presentation/
│   └── Kalman_Filter_State_Estimation_ASILA.pptx # ASILA Capstone defense deck
├── docs/
│   ├── Adaptive_KalmanFilter_Report.pdf       # Initial project technical report
│   └── AI_Models_Description.pdf              # Machine intelligence architecture breakdown
└── README.md
```
---

## Quickstart & Reproduction

The NASA battery aging datasets (`B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`) are included directly in the `data/` directory. The Samsung pipeline is documented in the paper and result summaries; large raw Samsung files may need to be obtained separately depending on distribution limits.

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/RamezM2004/hybrid-lstm-aukf-battery-soc.git
cd hybrid-lstm-aukf-battery-soc
pip install -r requirements.txt
```

### 2. Run Adaptive UKF State Estimation
```bash
python src/AdpNasaDataSet.py
```

### 3. Run Hybrid LSTM-AUKF Pipeline
```bash
python src/BatterySoC_LSTM.py
```
