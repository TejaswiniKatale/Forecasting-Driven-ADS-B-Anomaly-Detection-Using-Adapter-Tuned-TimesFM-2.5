"""
Colab-ready TimesFM 2.5 + CNN classifier for ADS-B attack tables.

Modes:
- finetuned: try TimesFM-side trainable path with fallback adapters
- frozen_residual: old frozen residual feature pipeline
"""

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception:
    print("Not running inside Colab, or Drive is already mounted.")

import hashlib
import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

geojson_path = "D:/Flights_Project/us_flights_2022-06-27_top10000_clean.csv"

SEED = 42
OUTPUT_DIR = Path("D:/Flights_Project/timesfm25_cnn_outputs_toplevel_api_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESFM_REPO_ID = "google/timesfm-2.5-200m-pytorch"
TIMESFM_FREQ = 0
TIMESFM_BACKBONE_PARAMS = 231_289_280
MODEL_NAME = "TimesFM 2.5 + CNN"
MODEL_NAME_ADAPTER_TUNED = "TimesFM 2.5 adapter-tuned + CNN"
MODEL_MODE = "finetuned"  # options: "finetuned", "frozen_residual"
FORCE_REBUILD_CACHE = False

FINE_TUNE_TIMESFM = True
FREEZE_TIMESFM_BACKBONE = False
FINE_TUNE_LAST_N_LAYERS = 2
USE_PARAMETER_EFFICIENT_ADAPTER = True
USE_RESIDUAL_FORECAST_AUX_LOSS = True
LAMBDA_FORECAST = 0.1

RAW_FEATURES = ["lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]
MODEL_FEATURES = [
    "x_local_m",
    "y_local_m",
    "baroaltitude",
    "velocity",
    "heading_sin",
    "heading_cos",
    "vertrate",
]
WINDOW_BY_TASK = {"spoofing": 128, "saturation": 32, "interpolation": 128, "replay": 32}
DILATION_BY_TASK = {"spoofing": 2, "saturation": 1, "interpolation": 1, "replay": 1}
HORIZON_BY_TASK = {"spoofing": 1, "saturation": 5, "interpolation": 1, "replay": 1}

MAX_CLEANED_SEGMENTS = 10_000
REQUIRED_TRAIN_FLIGHTS = 8_280
REQUIRED_EVAL_FLIGHTS = 800
REQUIRED_TOTAL_FLIGHTS = REQUIRED_TRAIN_FLIGHTS + REQUIRED_EVAL_FLIGHTS
MAX_WINDOWS_PER_FLIGHT = 2
MAX_WINDOWS_PER_TASK = 2_000
FROZEN_BATCH_SIZE = 128
FINETUNE_BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-4
WEIGHT_DECAY = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESFM_FORECAST_METHOD = None
TIMESFM_FORECAST_WORKING = False
FORECAST_FALLBACK_WARNED = False
FORECAST_API_METHOD_PRINTED = False

@dataclass
class Example:
    task: str
    flight_id: str
    x: np.ndarray
    y: int
    future: np.ndarray
    raw_error_m: float = 0.0
    case_id: str = ""
    role: str = ""

class TimesFmFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

class WindowFutureDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return (
            torch.from_numpy(ex.x.astype(np.float32)),
            torch.from_numpy(ex.future.astype(np.float32)),
            torch.tensor(float(ex.y), dtype=torch.float32),
            idx,
        )

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """After alias-renaming, multiple raw columns can map to the same name
    (for example `time`, `timestamp`, and `lastcontact`). This helper merges
    duplicate columns by taking the first non-null value across duplicates.
    It prevents errors like: AttributeError: 'DataFrame' object has no attribute 'dtype'
    when selecting df["time"] returns multiple columns instead of one Series.
    """
    duplicate_names = [c for c in df.columns if list(df.columns).count(c) > 1]
    if not duplicate_names:
        return df

    out = pd.DataFrame(index=df.index)
    for name in df.columns:
        if name in out.columns:
            continue
        cols = df.loc[:, df.columns == name]
        if isinstance(cols, pd.DataFrame) and cols.shape[1] > 1:
            out[name] = cols.bfill(axis=1).iloc[:, 0]
        else:
            out[name] = cols.iloc[:, 0] if isinstance(cols, pd.DataFrame) else cols
    return out

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "latitude": "lat",
        "longitude": "lon",
        "geoaltitude": "baroaltitude",
        "altitude": "baroaltitude",
        "alt": "baroaltitude",
        "speed": "velocity",
        "groundspeed": "velocity",
        "track": "heading",
        "true_track": "heading",
        "vertical_rate": "vertrate",
        "verticalrate": "vertrate",
        "timestamp": "time",
        "lastcontact": "time",
        "icao": "icao24",
    }
    rename = {c: aliases.get(str(c).lower().strip(), str(c).lower().strip()) for c in df.columns}
    df = df.rename(columns=rename)
    df = _coalesce_duplicate_columns(df)
    return df

def to_seconds(series_or_df) -> pd.Series:
    """Convert a time column to seconds, robust to duplicate time columns.
    If a DataFrame is accidentally passed, coalesce it first.
    """
    if isinstance(series_or_df, pd.DataFrame):
        s = series_or_df.bfill(axis=1).iloc[:, 0]
    else:
        s = series_or_df

    if np.issubdtype(s.dtype, np.number):
        return pd.to_numeric(s, errors="coerce")

    ts = pd.to_datetime(s, errors="coerce")
    # Convert datetime to Unix seconds while preserving NaT as NaN.
    return ts.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)

def load_and_clean_segments(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if path_obj.suffix.lower() in {".geojson", ".json"}:
        import geopandas as gpd

        df = pd.DataFrame(gpd.read_file(path))
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = standardize_columns(df)
    missing = sorted(set(RAW_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "flight_id" not in df.columns:
        flight_col = next((c for c in ["icao24", "callsign", "track_id", "flight", "flightid"] if c in df.columns), None)
        df["flight_id"] = df[flight_col].astype(str) if flight_col else "flight_" + (np.arange(len(df)) // 256).astype(str)
    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=np.int64)
    df["time_s"] = to_seconds(df["time"])
    for c in RAW_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["flight_id", "time_s"]).copy()
    df = df[(df["lat"].between(24.0, 49.5)) & (df["lon"].between(-125.0, -66.0)) & (df["baroaltitude"] < 10000.0)].copy()
    df = df.sort_values(["flight_id", "time_s"]).reset_index(drop=True)

    segments = []
    for fid, g in df.groupby("flight_id", sort=False):
        dt = g["time_s"].diff().fillna(0.0)
        for seg_idx, seg in g.groupby((dt > 300.0).cumsum(), sort=False):
            seg = seg.copy()
            if len(seg) < 2:
                continue
            if float(seg["time_s"].iloc[-1] - seg["time_s"].iloc[0]) < 900.0:
                continue
            if float(seg[RAW_FEATURES].isna().mean().max()) > 0.10:
                continue
            seg[RAW_FEATURES] = seg[RAW_FEATURES].interpolate(limit_direction="both").ffill().bfill()
            seg["flight_id"] = f"{fid}__seg{int(seg_idx)}"
            segments.append(seg)
            if len(segments) >= MAX_CLEANED_SEGMENTS:
                break
        if len(segments) >= MAX_CLEANED_SEGMENTS:
            break
    if not segments:
        raise ValueError("No cleaned flight segments after filtering.")
    return pd.concat(segments, ignore_index=True)

def split_flight_ids(flight_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(sorted(set(map(str, flight_ids))))
    if len(ids) < REQUIRED_TOTAL_FLIGHTS:
        raise ValueError(
            f"Insufficient cleaned usable flights: found {len(ids)}, required at least {REQUIRED_TOTAL_FLIGHTS} "
            f"({REQUIRED_TRAIN_FLIGHTS} train + {REQUIRED_EVAL_FLIGHTS} final eval/test)."
        )
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    train_ids = ids[:REQUIRED_TRAIN_FLIGHTS]
    eval_ids = ids[REQUIRED_TRAIN_FLIGHTS : REQUIRED_TRAIN_FLIGHTS + REQUIRED_EVAL_FLIGHTS]
    assert len(set(train_ids).intersection(set(eval_ids))) == 0
    return train_ids, eval_ids

def build_scaler(train_df: pd.DataFrame) -> tuple[dict, dict]:
    tmp = train_df.copy()
    lat0 = tmp.groupby("flight_id")["lat"].transform("first")
    lon0 = tmp.groupby("flight_id")["lon"].transform("first")
    tmp["x_local_m"] = (tmp["lon"] - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    tmp["y_local_m"] = (tmp["lat"] - lat0) * 111320.0
    rad = np.deg2rad(tmp["heading"].to_numpy(np.float32))
    tmp["heading_sin"], tmp["heading_cos"] = np.sin(rad), np.cos(rad)
    means, stds = {}, {}
    for c in MODEL_FEATURES:
        means[c] = float(np.nanmean(tmp[c]))
        s = float(np.nanstd(tmp[c]))
        stds[c] = s if np.isfinite(s) and s > 1e-6 else 1.0
    return means, stds

def local_feature_transform(raw_seq: np.ndarray, ref_lat: float, ref_lon: float, means: dict, stds: dict) -> np.ndarray:
    lat, lon, alt, vel, hdg, vrt = [raw_seq[:, i] for i in range(6)]
    x_local = (lon - ref_lon) * 111320.0 * float(np.cos(np.deg2rad(ref_lat)))
    y_local = (lat - ref_lat) * 111320.0
    feats = np.stack([x_local, y_local, alt, vel, np.sin(np.deg2rad(hdg)), np.cos(np.deg2rad(hdg)), vrt], axis=1).astype(np.float32)
    for i, c in enumerate(MODEL_FEATURES):
        feats[:, i] = (feats[:, i] - means[c]) / stds[c]
    return feats

def add_position_offset(lat: np.ndarray, lon: np.ndarray, meters: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(angle_deg)
    dlat = (meters * np.cos(angle)) / 111320.0
    dlon = (meters * np.sin(angle)) / (111320.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.2))
    return lat + dlat, lon + dlon

def spoof_sequence(raw_seq: np.ndarray) -> np.ndarray:
    out = raw_seq.copy()
    start = max(2, len(out) // 3)
    ramp = np.linspace(0.0, 1.0, len(out) - start, dtype=np.float32)
    lat, lon = add_position_offset(out[start:, 0], out[start:, 1], 1500.0 * (ramp**1.3), 35.0)
    out[start:, 0], out[start:, 1] = lat, lon
    out[start:, 4] = (out[start:, 4] + 25.0 * ramp) % 360.0
    return out

def interpolation_sequence(raw_seq: np.ndarray) -> np.ndarray:
    out = raw_seq.copy()
    anchors = np.linspace(0, len(out) - 1, max(4, len(out) // 8)).astype(int)
    x = np.arange(len(out))
    for c in range(out.shape[1]):
        out[:, c] = np.interp(x, anchors, out[anchors, c])
    return out

def replay_sequence(raw_seq: np.ndarray, replay_len: int = 32) -> np.ndarray:
    out = raw_seq.copy()
    seg = min(replay_len, max(8, len(out) // 2))
    src_start = max(0, len(out) // 4 - seg // 2)
    out[-seg:] = out[src_start : src_start + seg]
    return out

def saturation_candidates(context_future_raw: np.ndarray, horizon: int) -> list[tuple[np.ndarray, int, float, str]]:
    out = [(context_future_raw.copy(), 1, 0.0, "real")]
    offsets = [-25, -16, -11, -7, -4, -2, -1, 1, 2, 4, 7, 11, 16, 25]
    ctx_len = len(context_future_raw) - horizon
    f_idx = np.arange(ctx_len, len(context_future_raw))
    for deg in offsets:
        cand = context_future_raw.copy()
        meters = 80.0 + 420.0 * np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        lat, lon = add_position_offset(cand[f_idx, 0], cand[f_idx, 1], meters, float(deg))
        cand[f_idx, 0], cand[f_idx, 1] = lat, lon
        out.append((cand, 0, float(meters[-1]), f"ghost_{deg:+d}"))
    return out

def collect_examples_for_task(df: pd.DataFrame, task: str, flight_ids: set[str], means: dict, stds: dict) -> list[Example]:
    w, d, h = WINDOW_BY_TASK[task], DILATION_BY_TASK[task], HORIZON_BY_TASK[task]
    examples: list[Example] = []
    stride = max(1, w // 2)
    for fid, g in df.groupby("flight_id", sort=False):
        fid = str(fid)
        if fid not in flight_ids:
            continue
        arr = g.sort_values("time_s")[RAW_FEATURES].to_numpy(np.float32)
        need = (w - 1) * d + 1 + h
        if len(arr) < need:
            continue
        starts = list(range(0, len(arr) - need + 1, stride))
        random.shuffle(starts)
        for start in starts[:MAX_WINDOWS_PER_FLIGHT]:
            ctx_idx = start + np.arange(w) * d
            fut_idx = ctx_idx[-1] + np.arange(1, h + 1)
            seq = arr[np.concatenate([ctx_idx, fut_idx])]
            context_raw, future_raw = seq[:w], seq[w:]
            ref_lat, ref_lon = float(context_raw[-1, 0]), float(context_raw[-1, 1])
            sat_case_id = f"{fid}_s{start}"
            if task == "saturation":
                for cand_seq, label, err_m, role in saturation_candidates(seq, h):
                    examples.append(
                        Example(
                            task=task,
                            flight_id=fid,
                            x=local_feature_transform(cand_seq[:w], ref_lat, ref_lon, means, stds),
                            y=label,
                            future=local_feature_transform(cand_seq[w:], ref_lat, ref_lon, means, stds),
                            raw_error_m=err_m,
                            case_id=sat_case_id,
                            role=role,
                        )
                    )
            else:
                examples.append(
                    Example(
                        task=task,
                        flight_id=fid,
                        x=local_feature_transform(context_raw, ref_lat, ref_lon, means, stds),
                        y=0,
                        future=local_feature_transform(future_raw, ref_lat, ref_lon, means, stds),
                        case_id=f"{fid}_s{start}_clean",
                        role="clean",
                    )
                )
                attacked = spoof_sequence(seq) if task == "spoofing" else interpolation_sequence(seq) if task == "interpolation" else replay_sequence(seq, 32)
                examples.append(
                    Example(
                        task=task,
                        flight_id=fid,
                        x=local_feature_transform(attacked[:w], ref_lat, ref_lon, means, stds),
                        y=1,
                        future=local_feature_transform(attacked[w:], ref_lat, ref_lon, means, stds),
                        case_id=f"{fid}_s{start}_{task}_attack",
                        role="attack",
                    )
                )
            if len(examples) >= MAX_WINDOWS_PER_TASK:
                break
        if len(examples) >= MAX_WINDOWS_PER_TASK:
            break
    random.shuffle(examples)
    return examples

def _build_timesfm_forecast_config(max_context: int = 512, max_horizon: int = 128):
    """Build a ForecastConfig across different timesfm package versions.

    TimesFM 2.5 torch commonly exposes forecast(horizon, inputs) and requires
    model.compile(ForecastConfig(...)) before forecasting. Constructor fields
    vary by package version, so this tries several safe variants.
    """
    candidate_classes = []
    try:
        import timesfm as _timesfm_pkg
        fc_cls = getattr(_timesfm_pkg, "ForecastConfig", None)
        if fc_cls is not None:
            candidate_classes.append(fc_cls)
    except Exception:
        pass
    try:
        from timesfm.timesfm_base import ForecastConfig as fc_cls
        candidate_classes.append(fc_cls)
    except Exception:
        pass

    seen = set()
    unique_classes = []
    for cls in candidate_classes:
        if id(cls) not in seen:
            unique_classes.append(cls)
            seen.add(id(cls))

    candidate_kw_sets = [
        {"max_context": max_context, "max_horizon": max_horizon, "normalize_inputs": True},
        {"max_context": max_context, "max_horizon": max_horizon},
        {"context_len": max_context, "horizon_len": max_horizon},
        {"context_length": max_context, "prediction_length": max_horizon},
        {"horizon": max_horizon},
        {},
    ]

    last_errors = []
    for cls in unique_classes:
        for kw in candidate_kw_sets:
            try:
                try:
                    sig = inspect.signature(cls)
                    allowed = set(sig.parameters.keys())
                    filtered = {k: v for k, v in kw.items() if k in allowed}
                except Exception:
                    filtered = dict(kw)
                return cls(**filtered)
            except Exception as exc:
                last_errors.append(f"{getattr(cls, '__name__', str(cls))}{kw} failed: {repr(exc)}")
    raise RuntimeError("Could not build TimesFM ForecastConfig. " + " | ".join(last_errors[-4:]))


def ensure_timesfm_compiled(timesfm_model, error_sink: Optional[list[str]] = None) -> bool:
    """Best-effort compile for TimesFM versions that require it before forecast()."""
    compile_fn = getattr(timesfm_model, "compile", None)
    if not callable(compile_fn):
        return True

    try:
        fc = _build_timesfm_forecast_config(max_context=512, max_horizon=128)
        compile_fn(fc)
        return True
    except Exception as exc_fc:
        if error_sink is not None:
            error_sink.append(f"compile(ForecastConfig) failed: {repr(exc_fc)}")

    try:
        fc = getattr(timesfm_model, "forecast_config", None)
        if callable(fc):
            fc = fc()
        if fc is not None:
            compile_fn(fc)
            return True
    except Exception as exc1:
        if error_sink is not None:
            error_sink.append(f"compile(model.forecast_config) failed: {repr(exc1)}")

    try:
        compile_fn()
        return True
    except Exception as exc0:
        if error_sink is not None:
            error_sink.append(f"compile() failed: {repr(exc0)}")

    return False


def load_timesfm_model():
    """Load TimesFM with support for both old internal API and new top-level API.

    Old API path:
        timesfm.timesfm_2p5.timesfm_2p5_torch.TimesFM_2p5_200M_torch
    New/top-level API:
        timesfm.TimesFm, timesfm.TimesFmHparams, timesfm.TimesFmCheckpoint
    """
    print("Loading TimesFM 2.5 backbone:", TIMESFM_REPO_ID)
    load_errors: list[str] = []

    # ------------------------------------------------------------------
    # 1) Try the old PyTorch TimesFM 2.5 internal import path.
    # ------------------------------------------------------------------
    try:
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

        for label, fn in [
            (
                "old API keyword arguments",
                lambda: TimesFM_2p5_200M_torch.from_pretrained(
                    pretrained_model_name_or_path=TIMESFM_REPO_ID,
                    torch_compile=False,
                ),
            ),
            (
                "old API positional repo id",
                lambda: TimesFM_2p5_200M_torch.from_pretrained(
                    TIMESFM_REPO_ID,
                    torch_compile=False,
                ),
            ),
            (
                "old API repo id only",
                lambda: TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO_ID),
            ),
        ]:
            try:
                model = fn()
                print(f"TimesFM loaded using {label}.")
                compile_errors: list[str] = []
                if ensure_timesfm_compiled(model, compile_errors):
                    print("TimesFM compile step completed or not required.")
                elif compile_errors:
                    print("TimesFM compile warnings:", " | ".join(compile_errors))
                return model
            except Exception as exc:
                load_errors.append(f"{label} failed: {repr(exc)}")
    except Exception as exc:
        load_errors.append(f"old internal import failed: {repr(exc)}")

    # ------------------------------------------------------------------
    # 2) Try the newer top-level TimesFM API.
    # ------------------------------------------------------------------
    try:
        import timesfm

        print("Using top-level timesfm API.")
        backend = "gpu" if torch.cuda.is_available() else "cpu"

        checkpoint_repos = [
            TIMESFM_REPO_ID,
            "google/timesfm-2.0-500m-pytorch",
            "google/timesfm-1.0-200m-pytorch",
        ]

        hparams_attempts = []
        if hasattr(timesfm, "TimesFmHparams"):
            # Newer versions may accept context_len/horizon_len.
            hparams_attempts.append(
                lambda: timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=16,
                    context_len=512,
                    horizon_len=5,
                )
            )
            # Older top-level API may accept only backend/per_core_batch_size.
            hparams_attempts.append(
                lambda: timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=16,
                )
            )
            # Some releases use positional/no-arg hparams; try no-arg too.
            hparams_attempts.append(lambda: timesfm.TimesFmHparams())
        else:
            raise RuntimeError("timesfm.TimesFmHparams not found in top-level API")

        if not hasattr(timesfm, "TimesFm") or not hasattr(timesfm, "TimesFmCheckpoint"):
            raise RuntimeError("Top-level timesfm API missing TimesFm or TimesFmCheckpoint")

        for repo in checkpoint_repos:
            for make_hparams in hparams_attempts:
                try:
                    hparams = make_hparams()
                    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=repo)
                    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
                    print(f"TimesFM loaded using top-level TimesFm API from {repo}.")
                    compile_errors: list[str] = []
                    if ensure_timesfm_compiled(model, compile_errors):
                        print("TimesFM compile step completed or not required.")
                    elif compile_errors:
                        print("TimesFM compile warnings:", " | ".join(compile_errors))
                    return model
                except Exception as exc:
                    load_errors.append(f"top-level API repo={repo} failed: {repr(exc)}")

    except Exception as exc:
        load_errors.append(f"top-level timesfm API failed: {repr(exc)}")

    raise RuntimeError(
        "Could not load TimesFM with either old internal API or new top-level API.\n"
        "Install/check TimesFM in Python 3.10, then rerun.\n"
        "Errors:\n" + "\n".join(load_errors[-20:])
    )


def extract_point_forecast(raw_output) -> np.ndarray:
    """Convert TimesFM forecast output into a 1-D numpy point forecast."""
    out = raw_output

    # Common TimesFM output: (point_forecast, quantile_forecast)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, list):
        if len(out) == 0:
            raise RuntimeError("Empty list forecast output.")
        out = out[0]
    if isinstance(out, dict):
        for k in ["point_forecast", "mean", "forecast", "forecasts", "prediction", "predictions", "outputs"]:
            if k in out:
                out = out[k]
                break

    # After dict extraction, output may still be tuple/list.
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, list):
        if len(out) == 0:
            raise RuntimeError("Empty list forecast output after extraction.")
        out = out[0]

    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()

    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        return arr[0, :, 0]
    raise RuntimeError(f"Unsupported TimesFM forecast output shape: {arr.shape}")


def debug_timesfm_forecast_api(timesfm_model):
    """Detect which forecast() signature works in the installed TimesFM version."""
    print("Debugging TimesFM forecast API...")
    try:
        sig = inspect.signature(timesfm_model.forecast)
        print("forecast signature:", sig)
    except Exception as exc:
        print("Could not inspect forecast signature:", repr(exc))

    doc = getattr(timesfm_model.forecast, "__doc__", None)
    if isinstance(doc, str) and doc.strip():
        print("forecast doc first line:", doc.strip().splitlines()[0][:300])
    else:
        print("forecast doc: <not available>")

    context = np.sin(np.linspace(0, 6.28, 128)).astype(np.float32)
    attempts = [
        ("horizon_inputs_kw", "forecast(horizon=1, inputs=[context])", lambda: timesfm_model.forecast(horizon=1, inputs=[context])),
        ("horizon_inputs_pos", "forecast(1, [context])", lambda: timesfm_model.forecast(1, [context])),
        ("list_context", "forecast([context])", lambda: timesfm_model.forecast([context])),
        ("numpy_batch", "forecast(np.asarray([context]))", lambda: timesfm_model.forecast(np.asarray([context], dtype=np.float32))),
        ("raw_context", "forecast(context)", lambda: timesfm_model.forecast(context)),
        ("inputs_kw", "forecast(inputs=[context])", lambda: timesfm_model.forecast(inputs=[context])),
        ("context_kw", "forecast(context=[context])", lambda: timesfm_model.forecast(context=[context])),
        ("forecast_context_kw", "forecast(forecast_context=[context])", lambda: timesfm_model.forecast(forecast_context=[context])),
        # Last-resort old-style attempts:
        ("top_level_list_freq", "forecast([context], freq=[0])", lambda: timesfm_model.forecast([context], freq=[TIMESFM_FREQ])),
        ("top_level_inputs_freq", "forecast(inputs=[context], freq=[0])", lambda: timesfm_model.forecast(inputs=[context], freq=[TIMESFM_FREQ])),
        ("freq_positional", "forecast([context], [freq])", lambda: timesfm_model.forecast([context], [TIMESFM_FREQ])),
        ("freq_inputs_kw", "forecast(inputs=[context], freq=[freq])", lambda: timesfm_model.forecast(inputs=[context], freq=[TIMESFM_FREQ])),
        ("freq_forecast_inputs_kw", "forecast(forecast_inputs=[context], freq=[freq])", lambda: timesfm_model.forecast(forecast_inputs=[context], freq=[TIMESFM_FREQ])),
    ]

    for method, display, fn in attempts:
        try:
            out = fn()
            arr = extract_point_forecast(out)
            print(f"[SUCCESS] {display} -> output={type(out).__name__}, parsed_shape={arr.shape}")
            return method
        except Exception as exc:
            print(f"[FAIL] {display} -> {repr(exc)}")

    return None


def _call_timesfm_forecast_by_method(timesfm_model, context: np.ndarray, method: Optional[str], horizon: int = 1):
    if method == "horizon_inputs_kw":
        return timesfm_model.forecast(horizon=horizon, inputs=[context])
    if method == "horizon_inputs_pos":
        return timesfm_model.forecast(horizon, [context])
    if method == "list_context":
        return timesfm_model.forecast([context])
    if method == "numpy_batch":
        return timesfm_model.forecast(np.asarray([context], dtype=np.float32))
    if method == "raw_context":
        return timesfm_model.forecast(context)
    if method == "inputs_kw":
        return timesfm_model.forecast(inputs=[context])
    if method == "context_kw":
        return timesfm_model.forecast(context=[context])
    if method == "forecast_context_kw":
        return timesfm_model.forecast(forecast_context=[context])
    if method == "top_level_list_freq":
        return timesfm_model.forecast([context], freq=[TIMESFM_FREQ])
    if method == "top_level_inputs_freq":
        return timesfm_model.forecast(inputs=[context], freq=[TIMESFM_FREQ])
    if method == "freq_positional":
        return timesfm_model.forecast([context], [TIMESFM_FREQ])
    if method == "freq_inputs_kw":
        return timesfm_model.forecast(inputs=[context], freq=[TIMESFM_FREQ])
    if method == "freq_forecast_inputs_kw":
        return timesfm_model.forecast(forecast_inputs=[context], freq=[TIMESFM_FREQ])
    raise ValueError(f"Unknown TimesFM forecast method: {method}")


def forecast_single_series(timesfm_model, series: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast one normalized ADS-B feature sequence.

    If TimesFM forecast() is incompatible in the current Colab runtime, this
    returns a persistence fallback instead of crashing. The CSV records whether
    real TimesFM forecasting worked.
    """
    global TIMESFM_FORECAST_METHOD, TIMESFM_FORECAST_WORKING, FORECAST_FALLBACK_WARNED, FORECAST_API_METHOD_PRINTED

    context = np.nan_to_num(np.asarray(series, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

    # Use detected forecast API first.
    if TIMESFM_FORECAST_METHOD is not None:
        try:
            pred = extract_point_forecast(_call_timesfm_forecast_by_method(timesfm_model, context, TIMESFM_FORECAST_METHOD, horizon=horizon))
            pred = np.asarray(pred, dtype=np.float32).reshape(-1)
            TIMESFM_FORECAST_WORKING = True
            if not FORECAST_API_METHOD_PRINTED:
                print(f"TimesFM forecast API working using: {TIMESFM_FORECAST_METHOD}")
                FORECAST_API_METHOD_PRINTED = True
            if len(pred) < horizon:
                pred = np.pad(pred, (0, horizon - len(pred)), mode="edge")
            return pred[:horizon]
        except Exception as exc:
            # Try all methods below before falling back.
            pass

    # If debug was not run or selected method failed, try all known methods.
    methods = [
        "horizon_inputs_kw", "horizon_inputs_pos",
        "top_level_list_freq", "top_level_inputs_freq",
        "list_context", "numpy_batch", "raw_context", "inputs_kw", "context_kw",
        "forecast_context_kw", "freq_positional", "freq_inputs_kw", "freq_forecast_inputs_kw"
    ]
    for method in methods:
        try:
            pred = extract_point_forecast(_call_timesfm_forecast_by_method(timesfm_model, context, method, horizon=horizon))
            pred = np.asarray(pred, dtype=np.float32).reshape(-1)
            TIMESFM_FORECAST_METHOD = method
            TIMESFM_FORECAST_WORKING = True
            if not FORECAST_API_METHOD_PRINTED:
                print(f"TimesFM forecast API working using: {method}")
                FORECAST_API_METHOD_PRINTED = True
            if len(pred) < horizon:
                pred = np.pad(pred, (0, horizon - len(pred)), mode="edge")
            return pred[:horizon]
        except Exception:
            continue

    # Fallback keeps the script from crashing, but results are not valid TimesFM results.
    TIMESFM_FORECAST_WORKING = False
    TIMESFM_FORECAST_METHOD = None
    if not FORECAST_FALLBACK_WARNED:
        print("ERROR: TimesFM forecast API still failed. Using persistence fallback. These results are NOT valid TimesFM results.")
        FORECAST_FALLBACK_WARNED = True

    last_val = float(context[-1]) if len(context) > 0 else 0.0
    return np.full((horizon,), last_val, dtype=np.float32)


def make_timesfm_feature_tensor(timesfm_model, window: np.ndarray, actual_future: np.ndarray, horizon: int) -> np.ndarray:
    residuals = []
    for i in range(window.shape[1]):
        pred = forecast_single_series(timesfm_model, window[:, i], horizon=horizon)
        actual = float(actual_future[min(horizon - 1, len(actual_future) - 1), i])
        residuals.append(actual - float(pred[min(horizon - 1, len(pred) - 1)]))
    residuals = np.asarray(residuals, dtype=np.float32)
    rep_signed = np.repeat(residuals.reshape(1, -1), window.shape[0], axis=0)
    rep_abs = np.repeat(np.abs(residuals).reshape(1, -1), window.shape[0], axis=0)
    return np.concatenate([window.T, rep_signed.T, rep_abs.T], axis=0).astype(np.float32)

def split_hash(train_ids: np.ndarray, eval_ids: np.ndarray) -> str:
    payload = "|".join(sorted(map(str, train_ids))) + "||" + "|".join(sorted(map(str, eval_ids)))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

def materialize_frozen_features(timesfm_model, examples: list[Example], task: str, cache_tag: str, split_sig: str) -> tuple[np.ndarray, np.ndarray, float]:
    cache_path = OUTPUT_DIR / f"{cache_tag}_timesfm25_features.npz"
    meta = {"task": task, "window": WINDOW_BY_TASK[task], "dilation": DILATION_BY_TASK[task], "horizon": HORIZON_BY_TASK[task], "feature_list": MODEL_FEATURES, "split_hash": split_sig, "num_examples": len(examples)}
    if cache_path.exists() and not FORCE_REBUILD_CACHE:
        cached = np.load(cache_path, allow_pickle=True)
        if json.loads(str(cached["metadata"].item())) == meta:
            return cached["x"], cached["y"], float(cached["timesfm_time_per_message_ms"])
    h = HORIZON_BY_TASK[task]
    t0 = time.perf_counter()
    tensors = [make_timesfm_feature_tensor(timesfm_model, ex.x, ex.future, h) for ex in tqdm(examples, desc=f"TimesFM features {task}")]
    elapsed = max(1e-9, time.perf_counter() - t0)
    x = np.stack(tensors).astype(np.float32)
    y = np.asarray([ex.y for ex in examples], dtype=np.int64)
    tfm_ms = 1000.0 * elapsed / max(1, len(examples))
    np.savez_compressed(cache_path, x=x, y=y, metadata=json.dumps(meta), timesfm_time_per_message_ms=np.float32(tfm_ms))
    return x, y, tfm_ms


def safe_timesfm_parameters(timesfm_model):
    """Safely access parameters for TimesFM wrappers that may not be nn.Module."""
    if hasattr(timesfm_model, "parameters") and callable(timesfm_model.parameters):
        try:
            return list(timesfm_model.parameters())
        except Exception:
            return []
    return []

def safe_timesfm_named_parameters(timesfm_model):
    """Safely access named parameters for TimesFM wrappers that may not be nn.Module."""
    if hasattr(timesfm_model, "named_parameters") and callable(timesfm_model.named_parameters):
        try:
            return list(timesfm_model.named_parameters())
        except Exception:
            return []
    return []

def get_timesfm_blocks(timesfm_model) -> list[nn.Module]:
    blocks = []
    for name in ["decoder", "transformer", "model", "backbone"]:
        mod = getattr(timesfm_model, name, None)
        if mod is None:
            continue
        for seq_name in ["layers", "blocks", "h", "block"]:
            seq = getattr(mod, seq_name, None)
            if isinstance(seq, (nn.ModuleList, list, tuple)):
                blocks = list(seq)
                if blocks:
                    return blocks
    return blocks

def configure_timesfm_trainability(timesfm_model, fine_tune_last_n_layers: int) -> dict:
    params = safe_timesfm_parameters(timesfm_model)
    if not params:
        print("TimesFM object does not expose .parameters(); true TimesFM fine-tuning is not available.")
        return {"timesfm_trainable": 0, "timesfm_total": TIMESFM_BACKBONE_PARAMS}

    for p in params:
        p.requires_grad = False

    blocks = get_timesfm_blocks(timesfm_model)
    if blocks:
        for block in blocks[-fine_tune_last_n_layers:]:
            for p in block.parameters():
                p.requires_grad = True
    else:
        for name, p in safe_timesfm_named_parameters(timesfm_model):
            if any(k in name.lower() for k in ["final", "norm", "head", "out", "proj"]):
                p.requires_grad = True

    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    print(f"TimesFM trainable parameters after configure_timesfm_trainability: {trainable:,} / {total:,}")
    return {"timesfm_trainable": int(trainable), "timesfm_total": int(total)}

class ConvClassifierHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Conv1d(96, 160, kernel_size=5, padding=2),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(160, 160, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

class FineTunedTimesFmCnnClassifier(nn.Module):
    def __init__(self, timesfm_model, feature_dim: int, horizon: int):
        super().__init__()
        self.timesfm_model = timesfm_model
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.input_adapter = nn.Linear(feature_dim, feature_dim)
        self.residual_projection = nn.Linear(feature_dim * 3, feature_dim)
        self.forecast_head = nn.Linear(feature_dim, horizon * feature_dim)
        self.cnn_head = ConvClassifierHead(feature_dim)
        self.has_diff_hidden = False
        self.parameter_efficient_mode = True
        self._probe_differentiable_path()

    def _probe_differentiable_path(self) -> None:
        if not FINE_TUNE_TIMESFM or FREEZE_TIMESFM_BACKBONE:
            return
        try:
            dummy = torch.randn(1, 16, requires_grad=True)
            raw = self.timesfm_model.forecast([dummy.detach().cpu().numpy().reshape(-1).astype(np.float32)])
            out = raw[0] if isinstance(raw, tuple) else raw
            if isinstance(out, dict):
                out = out.get("point_forecast", out.get("forecast", out))
            if torch.is_tensor(out) and out.requires_grad:
                self.has_diff_hidden = True
                self.parameter_efficient_mode = False
        except Exception:
            self.has_diff_hidden = False
            self.parameter_efficient_mode = True

    def _timesfm_forecast_numpy(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        preds = np.zeros((x.shape[0], self.horizon, x.shape[2]), dtype=np.float32)
        for b in range(x_np.shape[0]):
            for f in range(x_np.shape[2]):
                preds[b, :, f] = forecast_single_series(self.timesfm_model, x_np[b, :, f], self.horizon)
        return torch.from_numpy(preds).to(x.device)

    def _timesfm_forecast_differentiable(self, x: torch.Tensor) -> torch.Tensor:
        preds = []
        for b in range(x.shape[0]):
            per_feat = []
            for f in range(x.shape[2]):
                series = x[b, :, f]
                raw = self.timesfm_model.forecast([series.detach().cpu().numpy().reshape(-1).astype(np.float32)])
                out = raw[0] if isinstance(raw, tuple) else raw
                if isinstance(out, dict):
                    out = out.get("point_forecast", out.get("forecast", out))
                if not torch.is_tensor(out):
                    raise RuntimeError("TimesFM forecast did not return tensor for differentiable path.")
                if out.ndim == 1:
                    p = out
                elif out.ndim == 2:
                    p = out[0]
                elif out.ndim == 3:
                    p = out[0, :, 0]
                else:
                    raise RuntimeError(f"Unexpected differentiable forecast shape: {tuple(out.shape)}")
                if p.shape[0] < self.horizon:
                    p = torch.nn.functional.pad(p, (0, self.horizon - p.shape[0]), mode="replicate")
                per_feat.append(p[: self.horizon])
            preds.append(torch.stack(per_feat, dim=-1))
        return torch.stack(preds, dim=0)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> dict[str, torch.Tensor]:
        x_adapt = self.input_adapter(x)
        if self.has_diff_hidden:
            try:
                forecast = self._timesfm_forecast_differentiable(x_adapt)
                self.parameter_efficient_mode = False
            except Exception:
                forecast = self.forecast_head(x_adapt.mean(dim=1)).view(x.size(0), self.horizon, self.feature_dim)
                self.parameter_efficient_mode = True
        else:
            with torch.no_grad():
                timesfm_forecast = self._timesfm_forecast_numpy(x_adapt)
            proj_in = torch.cat([x_adapt[:, -self.horizon :, :], timesfm_forecast, timesfm_forecast - future], dim=-1)
            forecast = self.forecast_head(self.residual_projection(proj_in).mean(dim=1)).view(x.size(0), self.horizon, self.feature_dim)
            self.parameter_efficient_mode = True
        residual = forecast - future
        seq = x_adapt + self.residual_projection(torch.cat([x_adapt, x_adapt, x_adapt * 0.0 + residual.mean(dim=1, keepdim=True)], dim=-1))
        logits = self.cnn_head(seq.transpose(1, 2))
        return {"logits": logits, "forecast": forecast}

class FrozenResidualClassifier(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.cnn_head = ConvClassifierHead(in_channels)

    def forward(self, x: torch.Tensor, future: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        return {"logits": self.cnn_head(x), "forecast": torch.empty(0, device=x.device)}

def model_param_breakdown(model: nn.Module) -> dict[str, int]:
    timesfm_trainable = 0
    adapter_trainable = 0
    cnn_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if "timesfm_model" in name:
            timesfm_trainable += n
        elif "cnn_head" in name:
            cnn_trainable += n
        else:
            adapter_trainable += n
    total_trainable = timesfm_trainable + adapter_trainable + cnn_trainable
    return {
        "timesfm_trainable": int(timesfm_trainable),
        "adapter_trainable": int(adapter_trainable),
        "cnn_trainable": int(cnn_trainable),
        "total_trainable": int(total_trainable),
        "total_reported": int(TIMESFM_BACKBONE_PARAMS + adapter_trainable + cnn_trainable),
    }

def count_timesfm_grad_flow(model: nn.Module) -> int:
    grad_count = 0
    for name, p in model.named_parameters():
        if "timesfm_model" in name and p.requires_grad and p.grad is not None:
            if torch.is_tensor(p.grad) and torch.any(p.grad != 0):
                grad_count += p.numel()
    return int(grad_count)

def evaluate_loss(model: nn.Module, loader: DataLoader, aux_loss: bool) -> float:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, fb, yb, _ in loader:
            xb, fb, yb = xb.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
            out = model(xb, fb)
            loss = bce(out["logits"], yb)
            if aux_loss and out["forecast"].numel() > 0:
                loss = loss + LAMBDA_FORECAST * mse(out["forecast"], fb)
            total += float(loss.item()) * len(xb)
            n += len(xb)
    return total / max(1, n)

def predict_scores_finetuned(model: nn.Module, examples: list[Example], batch_size: int) -> tuple[np.ndarray, float]:
    ds = WindowFutureDataset(examples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, fb, _, _ in loader:
            prob = torch.sigmoid(model(xb.to(DEVICE), fb.to(DEVICE))["logits"]).detach().cpu().numpy()
            scores.append(prob)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - t0)
    arr = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)
    return arr, 1000.0 * elapsed / max(1, len(ds))

def predict_scores_frozen(model: nn.Module, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    ds = TimesFmFeatureDataset(x, np.zeros((len(x),), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            prob = torch.sigmoid(model(xb.to(DEVICE), None)["logits"]).detach().cpu().numpy()
            scores.append(prob)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - t0)
    arr = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)
    return arr, 1000.0 * elapsed / max(1, len(ds))

def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "specificity": float(tn / max(1, tn + fp)),
        "fpr": float(fp / max(1, fp + tn)),
        "fnr": float(fn / max(1, fn + tp)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_t, best_f1, best_bal = 0.5, -1.0, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        m = metrics_at_threshold(y_true, scores, float(t))
        if (m["f1"] > best_f1) or (abs(m["f1"] - best_f1) < 1e-9 and m["balanced_accuracy"] > best_bal):
            best_t, best_f1, best_bal = float(t), m["f1"], m["balanced_accuracy"]
    return best_t

def train_classifier_finetuned(train_examples: list[Example], task: str, timesfm_model) -> tuple[nn.Module, float, dict]:
    idx = np.arange(len(train_examples))
    y_all = np.asarray([ex.y for ex in train_examples], dtype=np.int64)
    tr_idx, va_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y_all if len(np.unique(y_all)) > 1 else None)
    tr_examples = [train_examples[i] for i in tr_idx]
    va_examples = [train_examples[i] for i in va_idx]
    tr_loader = DataLoader(WindowFutureDataset(tr_examples), batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(WindowFutureDataset(va_examples), batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=0)

    model = FineTunedTimesFmCnnClassifier(timesfm_model, feature_dim=len(MODEL_FEATURES), horizon=HORIZON_BY_TASK[task]).to(DEVICE)
    if FINE_TUNE_TIMESFM and not FREEZE_TIMESFM_BACKBONE and not model.parameter_efficient_mode:
        configure_timesfm_trainability(model.timesfm_model, FINE_TUNE_LAST_N_LAYERS)
    else:
        for p in safe_timesfm_parameters(model.timesfm_model):
            p.requires_grad = False

    params = model_param_breakdown(model)
    print(f"TimesFM trainable params: {params['timesfm_trainable']:,}")
    print(f"Adapter trainable params: {params['adapter_trainable']:,}")
    print(f"CNN trainable params: {params['cnn_trainable']:,}")
    print(f"Total trainable params: {params['total_trainable']:,}")
    print(f"Total reported params: {params['total_reported']:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
    best_state, best_val = None, math.inf
    printed_actual_mode = False
    actual_mode = "parameter_efficient_adaptation"
    model_name_actual = MODEL_NAME_ADAPTER_TUNED
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, fb, yb, _ in tr_loader:
            xb, fb, yb = xb.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
            out = model(xb, fb)
            loss = bce(out["logits"], yb)
            if USE_RESIDUAL_FORECAST_AUX_LOSS and out["forecast"].numel() > 0:
                loss = loss + LAMBDA_FORECAST * mse(out["forecast"], fb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if not printed_actual_mode:
                timesfm_grad_params = count_timesfm_grad_flow(model)
                timesfm_trainable = params.get("timesfm_trainable", 0)
                if timesfm_trainable > 0 and timesfm_grad_params > 0:
                    print("Running TRUE fine-tuning: TimesFM parameters + adapters + CNN head are trainable.")
                    actual_mode = "true_finetuning"
                    model_name_actual = MODEL_NAME
                else:
                    print("Running parameter-efficient adaptation: TimesFM backbone frozen; adapters + CNN head are trainable.")
                    actual_mode = "parameter_efficient_adaptation"
                    model_name_actual = MODEL_NAME_ADAPTER_TUNED
                printed_actual_mode = True
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        val_loss = evaluate_loss(model, va_loader, aux_loss=USE_RESIDUAL_FORECAST_AUX_LOSS)
        print(f"{task} epoch {epoch:02d}: train_loss={running / max(1, n):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores, _ = predict_scores_finetuned(model, va_examples, FINETUNE_BATCH_SIZE)
    threshold = select_threshold(np.asarray([ex.y for ex in va_examples], dtype=np.int64), val_scores)
    print(f"{task} selected_threshold={threshold:.3f}")
    torch.save(model.state_dict(), OUTPUT_DIR / f"timesfm25_cnn_{task}.pt")
    params["training_mode_actual"] = actual_mode
    params["model_name_actual"] = model_name_actual
    params["timesfm_trainable_parameters"] = params.get("timesfm_trainable", 0)
    params["adapter_trainable_parameters"] = params.get("adapter_trainable", 0)
    params["cnn_trainable_parameters"] = params.get("cnn_trainable", 0)
    params["total_trainable_parameters"] = params.get("total_trainable", 0)
    return model, threshold, params

def train_classifier_frozen(train_x: np.ndarray, train_y: np.ndarray, task: str) -> tuple[nn.Module, float, dict]:
    idx = np.arange(len(train_y))
    tr_idx, va_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=train_y if len(np.unique(train_y)) > 1 else None)
    tr_ds = TimesFmFeatureDataset(train_x[tr_idx], train_y[tr_idx])
    va_ds = TimesFmFeatureDataset(train_x[va_idx], train_y[va_idx])
    tr_loader = DataLoader(tr_ds, batch_size=FROZEN_BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=FROZEN_BATCH_SIZE, shuffle=False, num_workers=0)
    model = FrozenResidualClassifier(in_channels=train_x.shape[1]).to(DEVICE)
    print("Running frozen residual mode: TimesFM backbone frozen; CNN trained on precomputed residual features.")
    params = model_param_breakdown(model)
    params["total_reported"] = TIMESFM_BACKBONE_PARAMS + params["cnn_trainable"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss()
    best_state, best_val = None, math.inf
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = bce(model(xb, None)["logits"], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        vloss = 0.0
        vn = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vloss += float(bce(model(xb, None)["logits"], yb).item()) * len(xb)
                vn += len(xb)
        val_loss = vloss / max(1, vn)
        print(f"{task} epoch {epoch:02d}: train_loss={running / max(1, n):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores, _ = predict_scores_frozen(model, train_x[va_idx], FROZEN_BATCH_SIZE)
    threshold = select_threshold(train_y[va_idx], val_scores)
    print(f"{task} selected_threshold={threshold:.3f}")
    torch.save(model.state_dict(), OUTPUT_DIR / f"timesfm25_cnn_{task}.pt")
    params["training_mode_actual"] = "frozen_residual"
    params["model_name_actual"] = MODEL_NAME
    params["timesfm_trainable_parameters"] = 0
    params["adapter_trainable_parameters"] = 0
    params["cnn_trainable_parameters"] = params.get("cnn_trainable", 0)
    params["total_trainable_parameters"] = params.get("total_trainable", params.get("cnn_trainable", 0))
    return model, threshold, params

def aggregate_accuracy(examples: list[Example], scores: np.ndarray, threshold: float, key_mode: str) -> float:
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        k = ex.flight_id if key_mode == "flight" else ex.case_id
        groups.setdefault(k, []).append(i)
    y_true, y_pred = [], []
    for _, idxs in groups.items():
        y_true.append(int(max(examples[i].y for i in idxs)))
        y_pred.append(int(max(scores[i] for i in idxs) >= threshold))
    return float(accuracy_score(y_true, y_pred)) if y_true else float("nan")

def evaluate_binary_task(task: str, model: nn.Module, threshold: float, eval_examples: list[Example], mode: str, eval_x: Optional[np.ndarray], tfm_ms: float) -> dict:
    y = np.asarray([ex.y for ex in eval_examples], dtype=np.int64)
    if mode == "finetuned":
        scores, total_ms = predict_scores_finetuned(model, eval_examples, FINETUNE_BATCH_SIZE)
        cnn_ms = float("nan")
        timesfm_ms = total_ms
    else:
        assert eval_x is not None
        scores, cnn_ms = predict_scores_frozen(model, eval_x, FROZEN_BATCH_SIZE)
        timesfm_ms = tfm_ms
        total_ms = cnn_ms + timesfm_ms
    base = metrics_at_threshold(y, scores, threshold)
    case_acc = aggregate_accuracy(eval_examples, scores, threshold, "case")
    return {
        **base,
        "selected_threshold": threshold,
        "message_accuracy": base["accuracy"],
        "trajectory_accuracy": case_acc,
        "case_accuracy": case_acc,
        "time_per_message_ms": total_ms,
        "cnn_time_per_message_ms": cnn_ms,
        "timesfm_feature_time_per_message_ms": timesfm_ms,
        "time_per_flight_ms": total_ms * WINDOW_BY_TASK[task],
        "localization_error_m": float("nan"),
        "perfect_percent": float("nan"),
    }

def evaluate_saturation(task: str, model: nn.Module, threshold: float, eval_examples: list[Example], mode: str, eval_x: Optional[np.ndarray], tfm_ms: float) -> dict:
    y = np.asarray([ex.y for ex in eval_examples], dtype=np.int64)
    if mode == "finetuned":
        scores, total_ms = predict_scores_finetuned(model, eval_examples, FINETUNE_BATCH_SIZE)
        cnn_ms = float("nan")
        timesfm_ms = total_ms
    else:
        assert eval_x is not None
        scores, cnn_ms = predict_scores_frozen(model, eval_x, FROZEN_BATCH_SIZE)
        timesfm_ms = tfm_ms
        total_ms = cnn_ms + timesfm_ms
    base = metrics_at_threshold(y, scores, threshold)
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(eval_examples):
        groups.setdefault(ex.case_id, []).append(i)
    top_hits, errs = [], []
    for idxs in groups.values():
        best_i = max(idxs, key=lambda i: scores[i])
        true_i = next((i for i in idxs if eval_examples[i].y == 1), None)
        if true_i is None:
            continue
        top_hits.append(1 if best_i == true_i else 0)
        errs.append(float(eval_examples[best_i].raw_error_m))
    return {
        **base,
        "selected_threshold": threshold,
        "binary_accuracy": base["accuracy"],
        "message_accuracy": base["accuracy"],
        "trajectory_accuracy": aggregate_accuracy(eval_examples, scores, threshold, "case"),
        "time_per_message_ms": total_ms,
        "cnn_time_per_message_ms": cnn_ms,
        "timesfm_feature_time_per_message_ms": timesfm_ms,
        "time_per_flight_ms": total_ms * WINDOW_BY_TASK[task],
        "localization_error_m": float(np.mean(errs)) if errs else float("nan"),
        "perfect_percent": float(np.mean(top_hits)) if top_hits else float("nan"),
    }

def pct(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{100.0 * v:.2f}%"

def ms(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.2f} ms"

def print_final_tables(results: dict, total_params: int) -> None:
    spoof, sat, interp, replay = results["spoofing"], results["saturation"], results["interpolation"], results["replay"]
    print("\n" + "=" * 78)
    print("FINAL TIMESFM 2.5 + CNN TABLE VALUES ONLY")
    print("=" * 78)
    spoof_df = pd.DataFrame([{"Model": MODEL_NAME, "Accuracy": pct(spoof["trajectory_accuracy"]), "Time/msg.": ms(spoof["time_per_message_ms"]), "Parameters": f"{total_params:,}"}])
    sat_df = pd.DataFrame([{"Model": MODEL_NAME, "Binary Acc.": pct(sat["binary_accuracy"]), "Error": f"{sat['localization_error_m']:.2f} m", "Perfect": pct(sat["perfect_percent"]), "Time/msg.": ms(sat["time_per_message_ms"])}])
    interp_df = pd.DataFrame([{"Model": MODEL_NAME, "Acc.": pct(interp["trajectory_accuracy"]), "Time/msg.": ms(interp["time_per_message_ms"]), "Msg. Acc.": pct(interp["message_accuracy"]), "Precision": pct(interp["precision"]), "Recall": pct(interp["recall"]), "F1": pct(interp["f1"]), "ROC-AUC": pct(interp["roc_auc"]), "FPR": pct(interp["fpr"])}])
    replay_df = pd.DataFrame([{"Model": MODEL_NAME, "Acc.": pct(replay["trajectory_accuracy"]), "Recall": pct(replay["recall"]), "FPR": pct(replay["fpr"]), "Precision": pct(replay["precision"]), "F1": pct(replay["f1"]), "ROC-AUC": pct(replay["roc_auc"]), "Time/msg.": ms(replay["time_per_message_ms"])}])
    print("\nSpoofing Attack Detection")
    print(spoof_df.to_string(index=False))
    print("\nSaturation Attack Detection and Localization")
    print(sat_df.to_string(index=False))
    print("\nInterpolated Attack Detection")
    print(interp_df.to_string(index=False))
    print("\nReplay Attack Detection")
    print(replay_df.to_string(index=False))

def save_all_tables_metrics(
    results: dict,
    eval_windows: dict,
    param_stats_by_attack: dict,
    cleaned_usable_flights: int,
    total_params: int,
) -> None:
    rows: list[dict[str, Any]] = []

    def add_row(
        table_id: str,
        table_name: str,
        attack: str,
        model: str,
        metric_name: str,
        metric_value: Any,
        metric_unit: str = "",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "table_id": table_id,
                "table_name": table_name,
                "attack": attack,
                "model": model,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_unit": metric_unit,
                "notes": notes,
            }
        )

    # Table 1
    t1_id = "Table 1"
    t1_name = "Dataset and Experimental Configuration"
    add_row(t1_id, t1_name, "configuration", "", "Cleaned usable flights", cleaned_usable_flights, "flights")
    add_row(t1_id, t1_name, "configuration", "", "Train flights", REQUIRED_TRAIN_FLIGHTS, "flights")
    add_row(t1_id, t1_name, "configuration", "", "Final Eval/Test flights", REQUIRED_EVAL_FLIGHTS, "flights")
    add_row(t1_id, t1_name, "configuration", "", "Max altitude", 10000, "ft")
    add_row(t1_id, t1_name, "configuration", "", "Saturation candidates", 15, "candidates")
    add_row(t1_id, t1_name, "configuration", "", "TimesFM backbone parameters", TIMESFM_BACKBONE_PARAMS, "parameters")
    add_row(t1_id, t1_name, "configuration", "", "Total reported parameters", total_params, "parameters")

    attack_meta_names = [
        ("num_train_flights", "count"),
        ("num_eval_flights", "count"),
        ("num_eval_windows", "count"),
        ("cnn_time_per_message_ms", "ms"),
        ("timesfm_feature_time_per_message_ms", "ms"),
        ("time_per_flight_ms", "ms"),
        ("timesfm_trainable_parameters", "parameters"),
        ("adapter_trainable_parameters", "parameters"),
        ("cnn_trainable_parameters", "parameters"),
        ("total_trainable_parameters", "parameters"),
        ("parameters", "parameters"),
    ]

    def add_attack_common(table_id: str, table_name: str, attack: str, model: str, res: dict, ps: dict) -> None:
        common_values = {
            "num_train_flights": REQUIRED_TRAIN_FLIGHTS,
            "num_eval_flights": REQUIRED_EVAL_FLIGHTS,
            "num_eval_windows": eval_windows.get(attack, 0),
            "cnn_time_per_message_ms": res.get("cnn_time_per_message_ms", np.nan),
            "timesfm_feature_time_per_message_ms": res.get("timesfm_feature_time_per_message_ms", np.nan),
            "time_per_flight_ms": res.get("time_per_flight_ms", np.nan),
            "timesfm_trainable_parameters": ps.get("timesfm_trainable_parameters", 0),
            "adapter_trainable_parameters": ps.get("adapter_trainable_parameters", 0),
            "cnn_trainable_parameters": ps.get("cnn_trainable_parameters", 0),
            "total_trainable_parameters": ps.get("total_trainable_parameters", 0),
            "parameters": ps.get("total_reported", TIMESFM_BACKBONE_PARAMS),
            "timesfm_forecast_method": TIMESFM_FORECAST_METHOD if TIMESFM_FORECAST_METHOD is not None else "persistence_fallback",
            "timesfm_forecast_working": bool(TIMESFM_FORECAST_WORKING),
        }
        for name, unit in attack_meta_names:
            add_row(table_id, table_name, attack, model, name, common_values[name], unit)
        add_row(table_id, table_name, attack, model, "timesfm_forecast_method", common_values["timesfm_forecast_method"], "")
        add_row(table_id, table_name, attack, model, "timesfm_forecast_working", common_values["timesfm_forecast_working"], "")

    # Table 2: Spoofing
    spoof = results["spoofing"]
    spoof_ps = param_stats_by_attack.get("spoofing", {})
    spoof_model = MODEL_NAME
    t2_id, t2_name = "Table 2", "Spoofing"
    add_row(t2_id, t2_name, "spoofing", spoof_model, "Accuracy", spoof.get("case_accuracy", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "Time/msg.", spoof.get("time_per_message_ms", np.nan), "ms")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "Parameters", spoof_ps.get("total_reported", TIMESFM_BACKBONE_PARAMS), "parameters")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "selected_threshold", spoof.get("selected_threshold", np.nan), "")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "message_accuracy", spoof.get("message_accuracy", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "case_accuracy", spoof.get("case_accuracy", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "precision", spoof.get("precision", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "recall", spoof.get("recall", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "f1", spoof.get("f1", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "roc_auc", spoof.get("roc_auc", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "average_precision", spoof.get("average_precision", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "specificity", spoof.get("specificity", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "fpr", spoof.get("fpr", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "fnr", spoof.get("fnr", np.nan), "ratio")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "tp", spoof.get("tp", np.nan), "count")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "tn", spoof.get("tn", np.nan), "count")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "fp", spoof.get("fp", np.nan), "count")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "fn", spoof.get("fn", np.nan), "count")
    add_row(t2_id, t2_name, "spoofing", spoof_model, "training_mode_actual", spoof_ps.get("training_mode_actual", "unknown"), "")
    add_attack_common(t2_id, t2_name, "spoofing", spoof_model, spoof, spoof_ps)

    # Table 3: Saturation
    sat = results["saturation"]
    sat_ps = param_stats_by_attack.get("saturation", {})
    sat_model = MODEL_NAME
    t3_id, t3_name = "Table 3", "Saturation"
    add_row(t3_id, t3_name, "saturation", sat_model, "Binary Acc.", sat.get("binary_accuracy", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "Error", sat.get("localization_error_m", np.nan), "m")
    add_row(t3_id, t3_name, "saturation", sat_model, "Perfect", sat.get("perfect_percent", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "Time/msg.", sat.get("time_per_message_ms", np.nan), "ms")
    add_row(t3_id, t3_name, "saturation", sat_model, "selected_threshold", sat.get("selected_threshold", np.nan), "")
    add_row(t3_id, t3_name, "saturation", sat_model, "precision", sat.get("precision", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "recall", sat.get("recall", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "f1", sat.get("f1", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "roc_auc", sat.get("roc_auc", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "average_precision", sat.get("average_precision", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "specificity", sat.get("specificity", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "fpr", sat.get("fpr", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "fnr", sat.get("fnr", np.nan), "ratio")
    add_row(t3_id, t3_name, "saturation", sat_model, "tp", sat.get("tp", np.nan), "count")
    add_row(t3_id, t3_name, "saturation", sat_model, "tn", sat.get("tn", np.nan), "count")
    add_row(t3_id, t3_name, "saturation", sat_model, "fp", sat.get("fp", np.nan), "count")
    add_row(t3_id, t3_name, "saturation", sat_model, "fn", sat.get("fn", np.nan), "count")
    add_row(t3_id, t3_name, "saturation", sat_model, "training_mode_actual", sat_ps.get("training_mode_actual", "unknown"), "")
    add_attack_common(t3_id, t3_name, "saturation", sat_model, sat, sat_ps)

    # Table 4: Interpolated
    interp = results["interpolation"]
    interp_ps = param_stats_by_attack.get("interpolation", {})
    interp_model = MODEL_NAME
    t4_id, t4_name = "Table 4", "Interpolated"
    add_row(t4_id, t4_name, "interpolation", interp_model, "Acc.", interp.get("case_accuracy", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "Time/msg.", interp.get("time_per_message_ms", np.nan), "ms")
    add_row(t4_id, t4_name, "interpolation", interp_model, "Msg. Acc.", interp.get("message_accuracy", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "Precision", interp.get("precision", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "Recall", interp.get("recall", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "F1", interp.get("f1", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "ROC-AUC", interp.get("roc_auc", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "FPR", interp.get("fpr", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "selected_threshold", interp.get("selected_threshold", np.nan), "")
    add_row(t4_id, t4_name, "interpolation", interp_model, "average_precision", interp.get("average_precision", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "specificity", interp.get("specificity", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "fnr", interp.get("fnr", np.nan), "ratio")
    add_row(t4_id, t4_name, "interpolation", interp_model, "tp", interp.get("tp", np.nan), "count")
    add_row(t4_id, t4_name, "interpolation", interp_model, "tn", interp.get("tn", np.nan), "count")
    add_row(t4_id, t4_name, "interpolation", interp_model, "fp", interp.get("fp", np.nan), "count")
    add_row(t4_id, t4_name, "interpolation", interp_model, "fn", interp.get("fn", np.nan), "count")
    add_row(t4_id, t4_name, "interpolation", interp_model, "training_mode_actual", interp_ps.get("training_mode_actual", "unknown"), "")
    add_attack_common(t4_id, t4_name, "interpolation", interp_model, interp, interp_ps)

    # Table 5: Replay
    replay = results["replay"]
    replay_ps = param_stats_by_attack.get("replay", {})
    replay_model = MODEL_NAME
    t5_id, t5_name = "Table 5", "Replay"
    add_row(t5_id, t5_name, "replay", replay_model, "Acc.", replay.get("case_accuracy", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "Recall", replay.get("recall", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "FPR", replay.get("fpr", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "Precision", replay.get("precision", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "F1", replay.get("f1", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "ROC-AUC", replay.get("roc_auc", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "Time/msg.", replay.get("time_per_message_ms", np.nan), "ms")
    add_row(t5_id, t5_name, "replay", replay_model, "selected_threshold", replay.get("selected_threshold", np.nan), "")
    add_row(t5_id, t5_name, "replay", replay_model, "average_precision", replay.get("average_precision", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "specificity", replay.get("specificity", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "fnr", replay.get("fnr", np.nan), "ratio")
    add_row(t5_id, t5_name, "replay", replay_model, "tp", replay.get("tp", np.nan), "count")
    add_row(t5_id, t5_name, "replay", replay_model, "tn", replay.get("tn", np.nan), "count")
    add_row(t5_id, t5_name, "replay", replay_model, "fp", replay.get("fp", np.nan), "count")
    add_row(t5_id, t5_name, "replay", replay_model, "fn", replay.get("fn", np.nan), "count")
    add_row(t5_id, t5_name, "replay", replay_model, "training_mode_actual", replay_ps.get("training_mode_actual", "unknown"), "")
    add_attack_common(t5_id, t5_name, "replay", replay_model, replay, replay_ps)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "timesfm25_cnn_all_tables_metrics.csv", index=False)

def print_example_sanity(task: str, train_examples: list[Example], eval_examples: list[Example]) -> None:
    train_labels = pd.Series([ex.y for ex in train_examples]).value_counts().to_dict()
    eval_labels = pd.Series([ex.y for ex in eval_examples]).value_counts().to_dict()
    print(f"{task} train_examples={len(train_examples)} eval_examples={len(eval_examples)}")
    print(f"{task} train label counts={train_labels}")
    print(f"{task} eval label counts={eval_labels}")
    print(f"{task} unique train case_ids={len(set(ex.case_id for ex in train_examples))}")
    print(f"{task} unique eval case_ids={len(set(ex.case_id for ex in eval_examples))}")

    if task in {"spoofing", "interpolation", "replay"}:
        tr_clean = {ex.case_id for ex in train_examples if ex.role == "clean"}
        tr_attack = {ex.case_id for ex in train_examples if ex.role == "attack"}
        ev_clean = {ex.case_id for ex in eval_examples if ex.role == "clean"}
        ev_attack = {ex.case_id for ex in eval_examples if ex.role == "attack"}
        assert len(tr_clean.intersection(tr_attack)) == 0, f"{task}: train clean/attack case_id overlap detected"
        assert len(ev_clean.intersection(ev_attack)) == 0, f"{task}: eval clean/attack case_id overlap detected"
    if task == "saturation":
        groups: dict[str, list[Example]] = {}
        for ex in eval_examples:
            groups.setdefault(ex.case_id, []).append(ex)
        for case_id, rows in groups.items():
            if len(rows) != 15:
                raise AssertionError(f"saturation eval case {case_id} has {len(rows)} candidates; expected 15")
            real_count = sum(1 for ex in rows if ex.y == 1)
            if real_count != 1:
                raise AssertionError(f"saturation eval case {case_id} has {real_count} real candidates; expected 1")

def main() -> None:
    seed_everything(SEED)
    df = load_and_clean_segments(geojson_path)
    all_ids = df["flight_id"].drop_duplicates().to_numpy()
    train_ids, eval_ids = split_flight_ids(all_ids)
    print(f"Cleaned usable flights: {len(all_ids)}")
    print(f"Train flights: {REQUIRED_TRAIN_FLIGHTS}")
    print(f"Final Eval/Test flights: {REQUIRED_EVAL_FLIGHTS}")
    print(f"Overlap between train and eval/test: {len(set(train_ids).intersection(set(eval_ids)))}")

    means, stds = build_scaler(df[df["flight_id"].isin(set(train_ids))].copy())
    split_sig = split_hash(train_ids, eval_ids)
    timesfm_model = load_timesfm_model()
    global TIMESFM_FORECAST_METHOD, TIMESFM_FORECAST_WORKING
    TIMESFM_FORECAST_METHOD = debug_timesfm_forecast_api(timesfm_model)
    TIMESFM_FORECAST_WORKING = TIMESFM_FORECAST_METHOD is not None
    print(f"Selected TimesFM forecast method: {TIMESFM_FORECAST_METHOD}")
    results, eval_windows = {}, {}
    final_param_stats = {"total_reported": TIMESFM_BACKBONE_PARAMS, "cnn_trainable": 0}
    param_stats_by_attack = {}

    for task in ["spoofing", "interpolation", "replay", "saturation"]:
        print("\n" + "-" * 78)
        print("Task:", task, "| window", WINDOW_BY_TASK[task], "| dilation", DILATION_BY_TASK[task], "| horizon", HORIZON_BY_TASK[task])
        tr_ex = collect_examples_for_task(df, task, set(train_ids), means, stds)
        ev_ex = collect_examples_for_task(df, task, set(eval_ids), means, stds)
        if len(tr_ex) < 8 or len(ev_ex) < 8:
            raise ValueError(f"Not enough examples for task={task}")
        print_example_sanity(task, tr_ex, ev_ex)

        if MODEL_MODE == "frozen_residual":
            tr_x, tr_y, _ = materialize_frozen_features(timesfm_model, tr_ex, task, f"{task}_train", split_sig)
            ev_x, _, ev_tfm_ms = materialize_frozen_features(timesfm_model, ev_ex, task, f"{task}_eval", split_sig)
            model, threshold, stats = train_classifier_frozen(tr_x, tr_y, task)
            res = evaluate_saturation(task, model, threshold, ev_ex, MODEL_MODE, ev_x, ev_tfm_ms) if task == "saturation" else evaluate_binary_task(task, model, threshold, ev_ex, MODEL_MODE, ev_x, ev_tfm_ms)
        else:
            model, threshold, stats = train_classifier_finetuned(tr_ex, task, timesfm_model)
            print(f"{task} internal run name: {stats.get('model_name_actual', MODEL_NAME)}")
            res = evaluate_saturation(task, model, threshold, ev_ex, MODEL_MODE, None, 0.0) if task == "saturation" else evaluate_binary_task(task, model, threshold, ev_ex, MODEL_MODE, None, 0.0)
        print(f"{task} runtime: cnn={res['cnn_time_per_message_ms']} ms/msg, timesfm_feature={res['timesfm_feature_time_per_message_ms']:.3f} ms/msg, total={res['time_per_message_ms']:.3f} ms/msg")
        results[task], eval_windows[task], final_param_stats = res, len(ev_ex), stats
        param_stats_by_attack[task] = stats

    print(f"TimesFM backbone params: {TIMESFM_BACKBONE_PARAMS:,}")
    print(f"CNN head params: {final_param_stats.get('cnn_trainable', 0):,}")
    print(f"Total params (reported): {final_param_stats['total_reported']:,}")
    print("Note: Spoofing/Interpolated/Replay table accuracy is case-level aggregated accuracy.")
    print_final_tables(results, final_param_stats["total_reported"])
    save_all_tables_metrics(
        results,
        eval_windows,
        param_stats_by_attack,
        cleaned_usable_flights=len(all_ids),
        total_params=final_param_stats["total_reported"],
    )
    print("\nSaved CSV outputs to:", OUTPUT_DIR)
    print("Saved all tables metrics CSV:", OUTPUT_DIR / "timesfm25_cnn_all_tables_metrics.csv")

if __name__ == "__main__":
    main()
