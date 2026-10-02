from ml.features.static import egfr_ckd_epi_2021

# Patient 1: 60yo Male, Creatinine 1.8 mg/dL (Suspected Stage 3 CKD)
p1 = egfr_ckd_epi_2021(creatinine_mg_dl=1.8, age_years=60.0, sex="M")

# Patient 2: 22yo Female, Creatinine 0.8 mg/dL (Healthy baseline)
p2 = egfr_ckd_epi_2021(creatinine_mg_dl=0.8, age_years=22.0, sex="F")

print("Patient 1 eGFR:", p1)
print("Patient 2 eGFR:", p2)