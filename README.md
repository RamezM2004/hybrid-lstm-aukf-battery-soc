# Hybrid Physics–AI State-of-Charge (SoC) Estimation for Lithium-Ion Batteries
### Integrating Adaptive Unscented Kalman Filtering (AUKF) with Deep LSTM Sequence Networks

**Author:** Ramez Al-Masadeh  
**Advisor:** Dr.-Ing. Sahar Qaadan  
**Institution:** German Jordanian University (GJU) — School of Applied Technical Sciences  
**Paper Manuscript:** `paper/Samsung_Hybrid_LSTM_AUKF_IEEE_Paper.tex` (IEEE Transactions format)

---

## Project Overview

Precise State-of-Charge (SoC) estimation is paramount for battery management systems (BMS) in electric vehicles and energy storage systems. Conventional Equivalent Circuit Models (ECMs) struggle in non-linear regions (such as the end-of-discharge "voltage knee"), while purely data-driven black-box neural networks lack physical guarantees and fail under unseen thermal dynamics.

This project delivers a **hybrid physics-informed framework** that fuses:
1. **Long Short-Term Memory (LSTM) Neural Network:** Predicts nominal SoC and detects the non-linear voltage collapse from causal measurements (voltage, current, voltage derivative, temperature).
2. **Adaptive Unscented Kalman Filter (AUKF):** Enforces a 1st-order Thevenin ECM, continuously correcting predictions through innovation monitoring and dynamic process noise ($Q$) adaptation.

---

## Key Scientific Innovations

- **The Voltage Knee Detection Mechanism:** An LSTM sequence classifier tracks temporal voltage derivatives to anticipate the sudden exponential voltage drop. Upon detection, the filter dynamically scales $Q$, making the state estimator aggressively responsive when model assumptions degrade.
- **Innovation Consistency & Huber Weighting:** Incorporates online innovation covariance matching and Normalized Innovation Squared (NIS) monitoring to bound sensor divergence under high noise.
- **Leakage-Safe Benchmarking (CALCE Samsung INR18650-20R):** Built a strict profile-separated evaluation pipeline (DST and BJDST for training, FUDS for validation, and US06 reserved for testing) across **0°C, 25°C, and 45°C**.

---

## Validation Results (US06 Driving Cycle)

| Estimator Architecture | SoC RMSE (%) | Max Error (%) | Mean NIS | 95% Chi-Square Bound Conformity |
| :--- | :---: | :---: | :---: | :---: |
| Standalone LSTM | 1.239% | 6.563% | — | — |
| **Hybrid LSTM–AUKF (Ours)** | **1.114%** | **5.344%** | **1.671** | **96.01%** |

The hybrid estimator reduces maximum error by **18.6%** and achieves **96.01% statistical filter consistency**, maintaining robustness across broad thermal variations.

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