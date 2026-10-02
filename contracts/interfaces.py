from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional, Protocol, Sequence
from pydantic import BaseModel

Provenance = Literal["measured", "derived", "model-estimated"]
RiskBand = Literal["green", "amber", "red"]


class StaticProfile(BaseModel):
    patient_id: str
    age_years: float
    sex: Literal["F", "M", "O"]
    baseline_creatinine_mg_dl: float
    egfr_ml_min_1_73m2: float
    uacr_category: Literal["A1", "A2", "A3"]
    bun_mg_dl: Optional[float] = None
    diabetes: bool = False
    hypertension: bool = False
    on_acei_arb: bool = False
    nsaid_exposure: bool = False
    on_diuretic: bool = False


class TelemetryFrame(BaseModel):
    timestamp: datetime
    map_mmhg: Optional[float] = None
    sbp_mmhg: Optional[float] = None
    dbp_mmhg: Optional[float] = None
    hr_bpm: Optional[float] = None
    hrv_rmssd_ms: Optional[float] = None
    spo2_pct: Optional[float] = None
    resp_rate_bpm: Optional[float] = None
    temp_c: Optional[float] = None
    sleep_stage: Optional[Literal["awake", "light", "deep", "rem"]] = None


class FeatureResult(BaseModel):
    vector: list[float]
    names: list[str]
    missingness: dict[str, float]
    schema_version: str


class TwinValue(BaseModel):
    value: float
    unit: str
    sd: float
    provenance: Provenance


class TwinState(BaseModel):
    timestamp: datetime
    ra: TwinValue
    re: TwinValue
    rbf: TwinValue
    pgc: TwinValue
    puf: TwinValue
    reserve_index: TwinValue


class Driver(BaseModel):
    feature: str
    display_name: str
    direction: Literal["increases_risk", "decreases_risk"]
    magnitude: float
    rationale: str


class ForecastResult(BaseModel):
    patient_id: str
    prediction_time: datetime
    risk_24h: float
    risk_48h: float
    risk_24h_interval: tuple[float, float]
    risk_48h_interval: tuple[float, float]
    risk_band: RiskBand
    low_confidence: bool
    low_confidence_reasons: list[str]
    drivers: list[Driver]
    model_version: str


class TrajectoryPerturbation(BaseModel):
    feature_overrides: dict[str, float] = {}
    feature_deltas: dict[str, float] = {}
    twin_state_override: Optional[TwinState] = None
    scenario: Literal["low", "central", "high"] = "central"


class ForecasterMeta(BaseModel):
    model_version: str
    feature_schema_version: str
    feature_names: list[str]
    trained_on: str
    is_stub: bool


class Forecaster(Protocol):
    meta: ForecasterMeta

    def predict(
        self, features: FeatureResult, patient_id: str, prediction_time: datetime
    ) -> ForecastResult: ...

    def predict_batch(
        self,
        features: Sequence[FeatureResult],
        patient_ids: Sequence[str],
        prediction_time: datetime,
    ) -> list[ForecastResult]: ...

    def predict_trajectory(
        self,
        base_features: FeatureResult,
        modified_telemetry_delta: TrajectoryPerturbation,
        patient_id: str,
        prediction_time: datetime,
    ) -> ForecastResult: ...


def risk_band(p: float) -> RiskBand:
    if p < 0.30:
        return "green"
    elif p < 0.70:
        return "amber"
    return "red"