# NEPHROTWIN — In Silico Hemodynamic Digital Twin for Early Warning of Acute Kidney Injury

> **Track:** Happiest Health — Reimagining & Reforming Healthcare in India Summit 2026  
> **Core Architecture:** Lumbar hemodynamic state estimation + LightGBM GBDT rolling early warning + Isotonic probability calibration + TreeSHAP explainability[cite: 1, 2, 3].

---

## 1. Executive Summary & Clinical Premise

India faces a large chronic kidney disease (CKD) burden (pooled community prevalence ~13.2%, roughly 9–17% across studies)[cite: 3]. In patients with reduced renal reserve, baseline hyperfiltration and blunted autoregulation make the glomerulus vulnerable to acute hypoperfusion, heat/dehydration stress, or nephrotoxic medication exposure (e.g., NSAIDs, ACEi/ARBs)[cite: 3].

Traditional detection depends on **serum creatinine**, which is a lagging marker that rises 24–48 hours after hemodynamic compromise has already occurred[cite: 3]. 

**NephroTwin** couples continuous/periodic vital signs with baseline renal reserve to flag acute-on-chronic decompensation before scheduled lab draws, offering a **counterfactual simulation sandbox** to evaluate hemodynamic interventions[cite: 3].

EHR Baseline Labs + Continuous Telemetry (MAP, HR, HRV)│▼┌───────────────────────────┐│   ml/features/api.py      │ ◄── Zero temporal data leakage (t <= T_pred)│   Rolling stats, dip %,   ││   Hypoperfusion burden    │└─────────────┬─────────────┘│▼┌───────────────────────────┐│      LightGBM GBDT        │ ◄── Patient-level GroupKFold CV│   + Isotonic Calibration  │ ◄── Calibrated 24h/48h AKI probability└─────────────┬─────────────┘│┌─────────────┴─────────────┐▼                           ▼┌───────────────┐           ┌───────────────────┐│ TreeSHAP      │           │ Counterfactual    ││ Driver Cards  │           │ Trajectory Engine │└───────────────┘           └───────────────────┘
---

## 2. Claim Discipline & Ethics Guardrails

To maintain clinical and regulatory rigor, NephroTwin observes strict claim limits[cite: 3]:

| Metric / Feature | What We Claim | What We DO NOT Claim |
|---|---|---|
| **Lead Time** | Flags risk prior to scheduled creatinine draws; measured as an empirical distribution on ICU hemodynamics[cite: 3]. | Does not guarantee fixed 24–48h advance warning on commercial consumer wearables[cite: 3]. |
| **Glomerular Dynamics** | Model-estimated physiology ($P_{\text{gc}}$, $R_a$, $R_e$, reserve index). | Not directly measured human micropuncture pressure[cite: 3]. |
| **Counterfactuals** | Mechanistic hypothesis exploration under documented effect sizes[cite: 3]. | Not certified autonomous medical decisions or causal clinical guarantees[cite: 3]. |
| **Blood Pressure** | Primary BP relies on validated cuff readings; wearable HR/HRV/SpO2/sleep are complementary[cite: 3]. | Cuffless wrist-optical BP is not assumed to be diagnostic-grade[cite: 3]. |

---

## 3. Repository Structure

nephrotwin/├── contracts/│   ├── interfaces.py          # Frozen shared types and Forecaster protocol (Day 1 freeze)│   └── variables.yaml         # Plausible physical limits and risk thresholds (<0.30, 0.30-0.69, >=0.70)├── ml/│   ├── artefacts/             # Exported model bundles (model.joblib, meta.json)│   ├── data/                  # raw (git-ignored), interim, processed│   ├── etl/│   │   ├── cohort.py          # Inclusion/exclusion criteria & baseline creatinine│   │   ├── labels.py          # KDIGO Stage 1 labeling (abs rise >=0.3 mg/dL, rel rise >=1.5x)│   │   ├── loaders.py         # VitalDB streaming & Apollo Hospitals (UCI CKD) loaders│   │   └── synthetic.py       # Indian-phenotype synthetic cohort generator│   ├── features/│   │   ├── api.py             # Shared extract_features implementation (pure function)│   │   ├── dynamic.py         # Rolling stats (6/12/24/48h), nocturnal dip %, hypotension burden│   │   └── static.py          # Race-free CKD-EPI 2021 eGFR equation│   ├── explain/│   │   ├── driver_names.yaml  # Feature-to-clinician translation mapping│   │   └── explainer.py       # TreeSHAP factor attributions + Isotonic Calibration│   ├── artefacts.py           # MLForecaster implementing the Forecaster protocol│   └── run_pipeline.py        # End-to-end training and artifact generation├── tests/│   └── ml/│       └── test_contracts.py  # Zero-leakage & boundary condition test suite├── .gitignore├── requirements.txt└── README.md
---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.10+
- Virtual environment

```bash
git clone [https://github.com/](https://github.com/)<your-username>/nephrotwin.git
cd nephrotwin

# Create and activate virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Reproduce Model Training & Artifact GenerationRun the pipeline to fetch clinical time-series, compute leakage-free features, train LightGBM via GroupKFold cross-validation, calibrate probabilities, and export the model bundle:   Bashpython -m ml.run_pipeline
Outputs:ml/artefacts/v1/model.joblib   ml/artefacts/v1/meta.json   Run Inference & Counterfactual TestBashpython -m ml.test_inference
Run Unit TestsVerify contract compliance, missingness handling, and absence of temporal leakage:   Bashpytest tests/ml/test_contracts.py
5. Machine Learning MethodologyTask Formulation: Predict KDIGO Stage 1 AKI onset within 24h and 48h horizons ($y_{24\text{h}}, y_{48\text{h}}$) evaluated over discrete 6-hour prediction grids.   Leakage Prevention: Pure function signature extract_features guarantees strictly no access to observations where $\text{timestamp} > T_{\text{pred}}$.   Validation Scheme: Patient-level GroupKFold split ensures no patient window overlaps between train and validation sets.   Probability Calibration: Isotonic regression aligns raw tree scores to empirical event rates, preventing probability clustering.   Explainability: TreeSHAP decomposes log-odds movements into individual clinical Driver objects (increases_risk / decreases_risk) with readable rationale.   6. Regulatory & Data Governance ComplianceIndian Data Protection: Designed around India's Digital Personal Data Protection (DPDP) Act 2023 principles: pseudonymized patient IDs (PATIENT_XXXX), strict separation of identity from telemetry, and edge-aggregation friendly[cite: 1, 3].Ethics Alignment: Conforms to the ICMR Guidelines for AI in Healthcare (2023) regarding human-in-the-loop decision support (CDSS) without autonomous clinical ordering[cite: 3].Credentialed Datasets: Compliant with PhysioNet and hospital DUAs; raw clinical telemetry is git-ignored and distributed via deterministic reproduction scripts.   
---

### 3. Git Commands to Commit and Push

Run these commands in PowerShell from the project root (`nephrotwin/`):

```powershell
# 1. Initialize git (if not already done)
git init

# 2. Check status to ensure NO csv, parquet, or venv files are staged
git status

# 3. Stage the code, contracts, tests, requirements, and docs
git add contracts/
git add ml/
git add tests/
git add .gitignore
git add requirements.txt
git add README.md

# 4. Confirm again that only source files and metadata are tracked
git status

# 5. Commit the milestone
git commit -m "feat(ml): complete BE1 MLForecaster pipeline, contracts, and evaluation suite"

# 6. Set your main branch and connect to your remote repository
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 7. Push to GitHub
git push -u origin main