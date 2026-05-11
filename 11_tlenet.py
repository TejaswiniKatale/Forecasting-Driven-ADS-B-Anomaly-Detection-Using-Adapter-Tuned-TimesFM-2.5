"""
Colab-ready pure T-LeNet baseline for ADS-B anomaly detection.
Standalone script: no TimesFM dependency.
"""

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception:
    print("Not running inside Colab, or Drive is already mounted.")

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


geojson_path = "/content/drive/MyDrive/Opensky_Flight_Project/flights.csv"
SEED = 42
OUTPUT_DIR = Path("/content/drive/MyDrive/Opensky_Flight_Project/tlenet_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "T-LeNet"
TIMESFM_BACKBONE_PARAMS = 0
RAW_FEATURES = ["lat", "lon", "baroaltitude", "velocity", "heading", "vertrate"]
MODEL_FEATURES = ["x_local_m", "y_local_m", "baroaltitude", "velocity", "heading_sin", "heading_cos", "vertrate"]
WINDOW_BY_TASK = {"spoofing": 128, "saturation": 32, "interpolation": 128, "replay": 32}
DILATION_BY_TASK = {"spoofing": 2, "saturation": 1, "interpolation": 1, "replay": 1}
HORIZON_BY_TASK = {"spoofing": 1, "saturation": 5, "interpolation": 1, "replay": 1}
MAX_CLEANED_SEGMENTS = 10_000
REQUIRED_TRAIN_FLIGHTS = 8_280
REQUIRED_EVAL_FLIGHTS = 800
REQUIRED_TOTAL_FLIGHTS = REQUIRED_TRAIN_FLIGHTS + REQUIRED_EVAL_FLIGHTS
MAX_WINDOWS_PER_TASK = 2_000
MAX_WINDOWS_PER_FLIGHT = 2
EPOCHS = 3
BATCH_SIZE = 128
LR = 2e-4
WEIGHT_DECAY = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class WindowDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = torch.from_numpy(ex.x.astype(np.float32))
        y = torch.tensor(float(ex.y), dtype=torch.float32)
        return x, y, idx


class TLeNetClassifier(nn.Module):
    def __init__(self, num_features: int, c1: int = 32, c2: int = 64, k: int = 5, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        pad = k // 2
        self.features = nn.Sequential(
            nn.Conv1d(num_features, c1, kernel_size=k, padding=pad),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k, padding=pad),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x).squeeze(1)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    df = df.rename(columns={c: aliases.get(c.lower().strip(), c.lower().strip()) for c in df.columns})
    if df.columns.duplicated().any():
        out = pd.DataFrame(index=df.index)
        for c in pd.unique(df.columns):
            block = df.loc[:, df.columns == c]
            out[c] = block.bfill(axis=1).iloc[:, 0] if block.shape[1] > 1 else block.iloc[:, 0]
        df = out
    return df


def to_seconds(s):
    if isinstance(s, pd.DataFrame):
        s = s.bfill(axis=1).iloc[:, 0]
    if np.issubdtype(s.dtype, np.number):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(s, errors="coerce").astype("int64") / 1e9


def load_and_clean_segments(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".geojson", ".json"}:
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
        c = next((x for x in ["icao24", "callsign", "track_id", "flight", "flightid"] if x in df.columns), None)
        df["flight_id"] = df[c].astype(str) if c else "flight_" + (np.arange(len(df)) // 256).astype(str)
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


def split_flight_ids(flight_ids: np.ndarray):
    ids = np.array(sorted(set(map(str, flight_ids))))
    if len(ids) < REQUIRED_TOTAL_FLIGHTS:
        raise ValueError(f"Insufficient cleaned usable flights: found {len(ids)}, required at least {REQUIRED_TOTAL_FLIGHTS}")
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    tr = ids[:REQUIRED_TRAIN_FLIGHTS]
    ev = ids[REQUIRED_TRAIN_FLIGHTS : REQUIRED_TRAIN_FLIGHTS + REQUIRED_EVAL_FLIGHTS]
    assert len(set(tr).intersection(set(ev))) == 0
    return tr, ev


def build_scaler(train_df: pd.DataFrame):
    t = train_df.copy()
    lat0 = t.groupby("flight_id")["lat"].transform("first")
    lon0 = t.groupby("flight_id")["lon"].transform("first")
    t["x_local_m"] = (t["lon"] - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    t["y_local_m"] = (t["lat"] - lat0) * 111320.0
    rad = np.deg2rad(t["heading"].to_numpy(np.float32))
    t["heading_sin"], t["heading_cos"] = np.sin(rad), np.cos(rad)
    means, stds = {}, {}
    for c in MODEL_FEATURES:
        means[c] = float(np.nanmean(t[c]))
        s = float(np.nanstd(t[c]))
        stds[c] = s if np.isfinite(s) and s > 1e-6 else 1.0
    return means, stds


def local_feature_transform(raw_seq: np.ndarray, ref_lat: float, ref_lon: float, means: dict, stds: dict):
    lat, lon, alt, vel, hdg, vrt = [raw_seq[:, i] for i in range(6)]
    x_local = (lon - ref_lon) * 111320.0 * float(np.cos(np.deg2rad(ref_lat)))
    y_local = (lat - ref_lat) * 111320.0
    feats = np.stack([x_local, y_local, alt, vel, np.sin(np.deg2rad(hdg)), np.cos(np.deg2rad(hdg)), vrt], axis=1).astype(np.float32)
    for i, c in enumerate(MODEL_FEATURES):
        feats[:, i] = (feats[:, i] - means[c]) / stds[c]
    return feats


def add_position_offset(lat, lon, meters, angle_deg):
    a = np.deg2rad(angle_deg)
    dlat = (meters * np.cos(a)) / 111320.0
    dlon = (meters * np.sin(a)) / (111320.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.2))
    return lat + dlat, lon + dlon


def spoof_sequence(raw):
    o = raw.copy()
    start = max(2, len(o) // 3)
    ramp = np.linspace(0.0, 1.0, len(o) - start, dtype=np.float32)
    lat, lon = add_position_offset(o[start:, 0], o[start:, 1], 1500.0 * (ramp**1.3), 35.0)
    o[start:, 0], o[start:, 1] = lat, lon
    o[start:, 4] = (o[start:, 4] + 25.0 * ramp) % 360.0
    return o


def interpolation_sequence(raw):
    o = raw.copy()
    anchors = np.linspace(0, len(o) - 1, max(4, len(o) // 8)).astype(int)
    x = np.arange(len(o))
    for c in range(o.shape[1]):
        o[:, c] = np.interp(x, anchors, o[anchors, c])
    return o


def replay_sequence(raw, replay_len=32):
    o = raw.copy()
    seg = min(replay_len, max(8, len(o) // 2))
    src_start = max(0, len(o) // 4 - seg // 2)
    o[-seg:] = o[src_start : src_start + seg]
    return o


def saturation_candidates(raw, horizon):
    out = [(raw.copy(), 1, 0.0, "real")]
    offsets = [-25, -16, -11, -7, -4, -2, -1, 1, 2, 4, 7, 11, 16, 25]
    ctx_len = len(raw) - horizon
    f_idx = np.arange(ctx_len, len(raw))
    for deg in offsets:
        cand = raw.copy()
        meters = 80.0 + 420.0 * np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        lat, lon = add_position_offset(cand[f_idx, 0], cand[f_idx, 1], meters, float(deg))
        cand[f_idx, 0], cand[f_idx, 1] = lat, lon
        out.append((cand, 0, float(meters[-1]), f"ghost_{deg:+d}"))
    return out


def collect_examples_for_task(df, task, flight_ids, means, stds):
    w, d, h = WINDOW_BY_TASK[task], DILATION_BY_TASK[task], HORIZON_BY_TASK[task]
    out = []
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
            ctx, fut = seq[:w], seq[w:]
            ref_lat, ref_lon = float(ctx[-1, 0]), float(ctx[-1, 1])
            sat_case = f"{fid}_s{start}"
            if task == "saturation":
                for cand, label, err, role in saturation_candidates(seq, h):
                    out.append(Example(task, fid, local_feature_transform(cand[:w], ref_lat, ref_lon, means, stds), label, local_feature_transform(cand[w:], ref_lat, ref_lon, means, stds), err, sat_case, role))
            else:
                out.append(Example(task, fid, local_feature_transform(ctx, ref_lat, ref_lon, means, stds), 0, local_feature_transform(fut, ref_lat, ref_lon, means, stds), 0.0, f"{fid}_s{start}_clean", "clean"))
                attacked = spoof_sequence(seq) if task == "spoofing" else interpolation_sequence(seq) if task == "interpolation" else replay_sequence(seq, 32)
                out.append(Example(task, fid, local_feature_transform(attacked[:w], ref_lat, ref_lon, means, stds), 1, local_feature_transform(attacked[w:], ref_lat, ref_lon, means, stds), 0.0, f"{fid}_s{start}_{task}_attack", "attack"))
            if len(out) >= MAX_WINDOWS_PER_TASK:
                break
        if len(out) >= MAX_WINDOWS_PER_TASK:
            break
    random.shuffle(out)
    return out


def print_example_sanity(task, tr, ev):
    print(f"{task} train_examples={len(tr)} eval_examples={len(ev)}")
    print(f"{task} train label counts={pd.Series([e.y for e in tr]).value_counts().to_dict()}")
    print(f"{task} eval label counts={pd.Series([e.y for e in ev]).value_counts().to_dict()}")
    if task in {"spoofing", "interpolation", "replay"}:
        tr_clean = {e.case_id for e in tr if e.role == "clean"}
        tr_attack = {e.case_id for e in tr if e.role == "attack"}
        ev_clean = {e.case_id for e in ev if e.role == "clean"}
        ev_attack = {e.case_id for e in ev if e.role == "attack"}
        assert len(tr_clean.intersection(tr_attack)) == 0
        assert len(ev_clean.intersection(ev_attack)) == 0
    if task == "saturation":
        groups = {}
        for ex in ev:
            groups.setdefault(ex.case_id, []).append(ex)
        for rows in groups.values():
            assert len(rows) == 15
            assert sum(1 for ex in rows if ex.y == 1) == 1


def metrics_at_threshold(y_true, scores, t):
    y_pred = (scores >= t).astype(int)
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


def select_threshold(y_true, scores):
    best_t, best_f1, best_bal = 0.5, -1.0, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        m = metrics_at_threshold(y_true, scores, float(t))
        if (m["f1"] > best_f1) or (abs(m["f1"] - best_f1) < 1e-9 and m["balanced_accuracy"] > best_bal):
            best_t, best_f1, best_bal = float(t), m["f1"], m["balanced_accuracy"]
    return best_t


def predict_scores(model, examples):
    ds = WindowDataset(examples)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    scores = []
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _, _ in loader:
            prob = torch.sigmoid(model(xb.to(DEVICE))).detach().cpu().numpy()
            scores.append(prob)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-9, time.perf_counter() - t0)
    return (np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32), 1000.0 * elapsed / max(1, len(ds)))


def train_classifier(train_examples, task):
    idx = np.arange(len(train_examples))
    y_all = np.asarray([e.y for e in train_examples], dtype=np.int64)
    tr_idx, va_idx = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y_all if len(np.unique(y_all)) > 1 else None)
    tr_ex = [train_examples[i] for i in tr_idx]
    va_ex = [train_examples[i] for i in va_idx]
    tr_loader = DataLoader(WindowDataset(tr_ex), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(WindowDataset(va_ex), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = TLeNetClassifier(num_features=len(MODEL_FEATURES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss()
    best_state, best_val = None, float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb, _ in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = bce(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        model.eval()
        vv, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb, _ in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vv += float(bce(model(xb), yb).item()) * len(xb)
                vn += len(xb)
        val_loss = vv / max(1, vn)
        print(f"{task} epoch {epoch:02d}: train_loss={running/max(1,n):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    va_scores, _ = predict_scores(model, va_ex)
    threshold = select_threshold(np.asarray([e.y for e in va_ex], dtype=np.int64), va_scores)
    print(f"{task} selected_threshold={threshold:.3f}")
    torch.save(model.state_dict(), OUTPUT_DIR / f"tlenet_{task}.pt")
    params = sum(p.numel() for p in model.parameters())
    return model, threshold, {
        "training_mode_actual": "pure_tlenet",
        "timesfm_trainable_parameters": 0,
        "adapter_trainable_parameters": 0,
        "cnn_trainable_parameters": int(params),
        "total_trainable_parameters": int(params),
        "total_reported": int(params),
        "cnn_trainable": int(params),
        "model_name_actual": MODEL_NAME,
    }


def aggregate_accuracy(examples, scores, threshold, by="case"):
    groups = {}
    for i, ex in enumerate(examples):
        key = ex.case_id if by == "case" else ex.flight_id
        groups.setdefault(key, []).append(i)
    yt, yp = [], []
    for idxs in groups.values():
        yt.append(int(max(examples[i].y for i in idxs)))
        yp.append(int(max(scores[i] for i in idxs) >= threshold))
    return float(accuracy_score(yt, yp)) if yt else float("nan")


def evaluate_binary(task, model, threshold, eval_examples):
    y = np.asarray([e.y for e in eval_examples], dtype=np.int64)
    scores, ms_msg = predict_scores(model, eval_examples)
    base = metrics_at_threshold(y, scores, threshold)
    case_acc = aggregate_accuracy(eval_examples, scores, threshold, by="case")
    return {**base, "selected_threshold": threshold, "message_accuracy": base["accuracy"], "case_accuracy": case_acc, "trajectory_accuracy": case_acc, "time_per_message_ms": ms_msg, "cnn_time_per_message_ms": ms_msg, "timesfm_feature_time_per_message_ms": 0.0, "time_per_flight_ms": ms_msg * WINDOW_BY_TASK[task], "localization_error_m": float("nan"), "perfect_percent": float("nan")}


def evaluate_saturation(task, model, threshold, eval_examples):
    y = np.asarray([e.y for e in eval_examples], dtype=np.int64)
    scores, ms_msg = predict_scores(model, eval_examples)
    base = metrics_at_threshold(y, scores, threshold)
    groups = {}
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
    return {**base, "selected_threshold": threshold, "binary_accuracy": base["accuracy"], "message_accuracy": base["accuracy"], "trajectory_accuracy": aggregate_accuracy(eval_examples, scores, threshold, by="case"), "time_per_message_ms": ms_msg, "cnn_time_per_message_ms": ms_msg, "timesfm_feature_time_per_message_ms": 0.0, "time_per_flight_ms": ms_msg * WINDOW_BY_TASK[task], "localization_error_m": float(np.mean(errs)) if errs else float("nan"), "perfect_percent": float(np.mean(top_hits)) if top_hits else float("nan")}


def pct(v):
    return "nan" if not np.isfinite(v) else f"{100.0 * v:.2f}%"


def ms(v):
    return "nan" if not np.isfinite(v) else f"{v:.2f} ms"


def print_final_tables(results, total_params):
    spoof, sat, interp, replay = results["spoofing"], results["saturation"], results["interpolation"], results["replay"]
    print("\n" + "=" * 78)
    print("FINAL PURE T-LENET TABLE VALUES ONLY")
    print("=" * 78)
    print("\nSpoofing Attack Detection")
    print(pd.DataFrame([{"Model": MODEL_NAME, "Accuracy": pct(spoof["case_accuracy"]), "Time/msg.": ms(spoof["time_per_message_ms"]), "Parameters": f"{total_params:,}"}]).to_string(index=False))
    print("\nSaturation Attack Detection and Localization")
    print(pd.DataFrame([{"Model": MODEL_NAME, "Binary Acc.": pct(sat["binary_accuracy"]), "Error": f"{sat['localization_error_m']:.2f} m", "Perfect": pct(sat["perfect_percent"]), "Time/msg.": ms(sat["time_per_message_ms"])}]).to_string(index=False))
    print("\nInterpolated Attack Detection")
    print(pd.DataFrame([{"Model": MODEL_NAME, "Acc.": pct(interp["case_accuracy"]), "Time/msg.": ms(interp["time_per_message_ms"]), "Msg. Acc.": pct(interp["message_accuracy"]), "Precision": pct(interp["precision"]), "Recall": pct(interp["recall"]), "F1": pct(interp["f1"]), "ROC-AUC": pct(interp["roc_auc"]), "FPR": pct(interp["fpr"])}]).to_string(index=False))
    print("\nReplay Attack Detection")
    print(pd.DataFrame([{"Model": MODEL_NAME, "Acc.": pct(replay["case_accuracy"]), "Recall": pct(replay["recall"]), "FPR": pct(replay["fpr"]), "Precision": pct(replay["precision"]), "F1": pct(replay["f1"]), "ROC-AUC": pct(replay["roc_auc"]), "Time/msg.": ms(replay["time_per_message_ms"])}]).to_string(index=False))


def save_all_tables_metrics(results, eval_windows, param_stats_by_attack, cleaned_usable_flights, total_params):
    rows = []

    def add(table_id, table_name, attack, model, metric_name, metric_value, metric_unit="", notes=""):
        rows.append({"table_id": table_id, "table_name": table_name, "attack": attack, "model": model, "metric_name": metric_name, "metric_value": metric_value, "metric_unit": metric_unit, "notes": notes})

    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "Cleaned usable flights", cleaned_usable_flights, "flights")
    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "Train flights", REQUIRED_TRAIN_FLIGHTS, "flights")
    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "Final Eval/Test flights", REQUIRED_EVAL_FLIGHTS, "flights")
    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "Max altitude", 10000, "ft")
    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "Saturation candidates", 15, "candidates")
    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "TimesFM backbone parameters", 0, "parameters")
    add("Table 1", "Dataset and Experimental Configuration", "configuration", "", "Total reported parameters", total_params, "parameters")

    for attack, table_id, table_name in [("spoofing", "Table 2", "Spoofing"), ("saturation", "Table 3", "Saturation"), ("interpolation", "Table 4", "Interpolated"), ("replay", "Table 5", "Replay")]:
        r = results[attack]
        ps = param_stats_by_attack.get(attack, {})
        add(table_id, table_name, attack, MODEL_NAME, "training_mode_actual", ps.get("training_mode_actual", "pure_tlenet"))
        if attack == "spoofing":
            for n, k, u in [("Accuracy", "case_accuracy", "ratio"), ("Time/msg.", "time_per_message_ms", "ms"), ("Parameters", "dummy", "parameters")]:
                add(table_id, table_name, attack, MODEL_NAME, n, ps.get("total_reported", 0) if n == "Parameters" else r.get(k, np.nan), u)
            for k in ["selected_threshold", "message_accuracy", "case_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "specificity", "fpr", "fnr", "tp", "tn", "fp", "fn"]:
                add(table_id, table_name, attack, MODEL_NAME, k, r.get(k, np.nan))
        elif attack == "saturation":
            for n, k, u in [("Binary Acc.", "binary_accuracy", "ratio"), ("Error", "localization_error_m", "m"), ("Perfect", "perfect_percent", "ratio"), ("Time/msg.", "time_per_message_ms", "ms")]:
                add(table_id, table_name, attack, MODEL_NAME, n, r.get(k, np.nan), u)
            for k in ["selected_threshold", "precision", "recall", "f1", "roc_auc", "average_precision", "specificity", "fpr", "fnr", "tp", "tn", "fp", "fn"]:
                add(table_id, table_name, attack, MODEL_NAME, k, r.get(k, np.nan))
        elif attack == "interpolation":
            for n, k, u in [("Acc.", "case_accuracy", "ratio"), ("Time/msg.", "time_per_message_ms", "ms"), ("Msg. Acc.", "message_accuracy", "ratio"), ("Precision", "precision", "ratio"), ("Recall", "recall", "ratio"), ("F1", "f1", "ratio"), ("ROC-AUC", "roc_auc", "ratio"), ("FPR", "fpr", "ratio")]:
                add(table_id, table_name, attack, MODEL_NAME, n, r.get(k, np.nan), u)
            for k in ["selected_threshold", "average_precision", "specificity", "fnr", "tp", "tn", "fp", "fn"]:
                add(table_id, table_name, attack, MODEL_NAME, k, r.get(k, np.nan))
        else:
            for n, k, u in [("Acc.", "case_accuracy", "ratio"), ("Recall", "recall", "ratio"), ("FPR", "fpr", "ratio"), ("Precision", "precision", "ratio"), ("F1", "f1", "ratio"), ("ROC-AUC", "roc_auc", "ratio"), ("Time/msg.", "time_per_message_ms", "ms")]:
                add(table_id, table_name, attack, MODEL_NAME, n, r.get(k, np.nan), u)
            for k in ["selected_threshold", "average_precision", "specificity", "fnr", "tp", "tn", "fp", "fn"]:
                add(table_id, table_name, attack, MODEL_NAME, k, r.get(k, np.nan))
        add(table_id, table_name, attack, MODEL_NAME, "num_train_flights", REQUIRED_TRAIN_FLIGHTS, "count")
        add(table_id, table_name, attack, MODEL_NAME, "num_eval_flights", REQUIRED_EVAL_FLIGHTS, "count")
        add(table_id, table_name, attack, MODEL_NAME, "num_eval_windows", eval_windows.get(attack, 0), "count")
        add(table_id, table_name, attack, MODEL_NAME, "cnn_time_per_message_ms", r.get("cnn_time_per_message_ms", np.nan), "ms")
        add(table_id, table_name, attack, MODEL_NAME, "timesfm_feature_time_per_message_ms", 0.0, "ms")
        add(table_id, table_name, attack, MODEL_NAME, "time_per_flight_ms", r.get("time_per_flight_ms", np.nan), "ms")
        add(table_id, table_name, attack, MODEL_NAME, "timesfm_trainable_parameters", 0, "parameters")
        add(table_id, table_name, attack, MODEL_NAME, "adapter_trainable_parameters", 0, "parameters")
        add(table_id, table_name, attack, MODEL_NAME, "cnn_trainable_parameters", ps.get("cnn_trainable_parameters", 0), "parameters")
        add(table_id, table_name, attack, MODEL_NAME, "total_trainable_parameters", ps.get("total_trainable_parameters", 0), "parameters")
        add(table_id, table_name, attack, MODEL_NAME, "parameters", ps.get("total_reported", 0), "parameters")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tlenet_all_tables_metrics.csv", index=False)


def main():
    seed_everything(SEED)
    print("Running PURE T-LENET baseline (no TimesFM).")
    print("Device:", DEVICE)
    print("Loading data:", geojson_path)
    df = load_and_clean_segments(geojson_path)
    all_ids = df["flight_id"].drop_duplicates().to_numpy()
    train_ids, eval_ids = split_flight_ids(all_ids)
    print(f"Cleaned usable flights: {len(all_ids)}")
    print(f"Train flights: {REQUIRED_TRAIN_FLIGHTS}")
    print(f"Final Eval/Test flights: {REQUIRED_EVAL_FLIGHTS}")
    print(f"Overlap between train and eval/test: {len(set(train_ids).intersection(set(eval_ids)))}")
    means, stds = build_scaler(df[df["flight_id"].isin(set(train_ids))].copy())

    results, eval_windows, param_stats_by_attack = {}, {}, {}
    final_param_stats = {"total_reported": 0, "cnn_trainable": 0}
    for task in ["spoofing", "interpolation", "replay", "saturation"]:
        print("\n" + "-" * 78)
        print("Task:", task, "| window", WINDOW_BY_TASK[task], "| dilation", DILATION_BY_TASK[task], "| horizon", HORIZON_BY_TASK[task])
        tr_ex = collect_examples_for_task(df, task, set(train_ids), means, stds)
        ev_ex = collect_examples_for_task(df, task, set(eval_ids), means, stds)
        if len(tr_ex) < 8 or len(ev_ex) < 8:
            raise ValueError(f"Not enough examples for task={task}")
        print_example_sanity(task, tr_ex, ev_ex)
        model, threshold, stats = train_classifier(tr_ex, task)
        res = evaluate_saturation(task, model, threshold, ev_ex) if task == "saturation" else evaluate_binary(task, model, threshold, ev_ex)
        print(f"{task} runtime: cnn={res['cnn_time_per_message_ms']:.3f} ms/msg, total={res['time_per_message_ms']:.3f} ms/msg")
        results[task], eval_windows[task], final_param_stats = res, len(ev_ex), stats
        param_stats_by_attack[task] = stats

    print(f"T-LeNet params: {final_param_stats.get('cnn_trainable', 0):,}")
    print(f"Total params (reported): {final_param_stats['total_reported']:,}")
    print_final_tables(results, final_param_stats["total_reported"])
    save_all_tables_metrics(results, eval_windows, param_stats_by_attack, cleaned_usable_flights=len(all_ids), total_params=final_param_stats["total_reported"])
    print("\nSaved CSV outputs to:", OUTPUT_DIR)
    print("Saved all tables metrics CSV:", OUTPUT_DIR / "tlenet_all_tables_metrics.csv")


if __name__ == "__main__":
    main()

