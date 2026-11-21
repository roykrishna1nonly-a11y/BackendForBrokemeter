from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import pickle
import threading
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brokemeter")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "brokemeter_model.pkl"))
CSV_PATH = Path(os.getenv("CSV_PATH", BASE_DIR / "cleaned_brokemeter_data.csv"))
STORE_PATH = Path(os.getenv("STORE_PATH", BASE_DIR / "telemetry_store.json"))
ADMIN_KEY = os.getenv("ADMIN_KEY", "brokemeter-admin")
MAX_LOGS = 200
MAX_EXITS = 200
MAX_OPENED_HISTORY = 300
LIVE_WINDOW_SECONDS = 4.0

model = None
df_spendings: Optional[pd.DataFrame] = None
store_lock = threading.Lock()

state: Dict[str, Any] = {
    "site_enabled": True,
    "logs": deque(maxlen=MAX_LOGS),
    "exits": deque(maxlen=MAX_EXITS),
    "opened_history": [],
    "active_sessions": {},
    "open_counter": 0,
}


def _safe_float(value: float, default: float = 0.0, decimals: int = 2) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return round(float(value), decimals)


def _load_store() -> None:
    if not STORE_PATH.exists():
        return
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        state["site_enabled"] = bool(raw.get("site_enabled", True))
        saved_logs = raw.get("logs", [])
        if isinstance(saved_logs, list):
            state["logs"] = deque(saved_logs[-MAX_LOGS:], maxlen=MAX_LOGS)
        saved_exits = raw.get("exits", [])
        if isinstance(saved_exits, list):
            state["exits"] = deque(saved_exits[-MAX_EXITS:], maxlen=MAX_EXITS)
        saved_opened = raw.get("opened_history", [])
        if isinstance(saved_opened, list):
            state["opened_history"] = saved_opened[-MAX_OPENED_HISTORY:]
        saved_sessions = raw.get("active_sessions", {})
        if isinstance(saved_sessions, dict):
            state["active_sessions"] = saved_sessions
        state["open_counter"] = int(raw.get("open_counter", len(state["opened_history"])))
    except Exception as exc:
        logger.warning("Could not load telemetry store: %s", exc)


def _save_store() -> None:
    try:
        payload = {
            "site_enabled": state["site_enabled"],
            "logs": list(state["logs"]),
            "exits": list(state["exits"]),
            "opened_history": state["opened_history"][-MAX_OPENED_HISTORY:],
            "active_sessions": state["active_sessions"],
            "open_counter": state["open_counter"],
        }
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save telemetry store: %s", exc)


def _extract_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first_ip = xff.split(",")[0].strip()
        if first_ip:
            return first_ip
    x_real_ip = request.headers.get("x-real-ip", "").strip()
    if x_real_ip:
        return x_real_ip
    return request.client.host if request.client else "unknown"


def _verify_admin(x_admin_key: Optional[str]) -> None:
    if not x_admin_key or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing X-Admin-Key.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, df_spendings

    with store_lock:
        _load_store()

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded ML model from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load model '%s': %s", MODEL_PATH, exc)
            model = None

    if CSV_PATH.exists():
        try:
            df_spendings = pd.read_csv(CSV_PATH)
            logger.info("Loaded dataset '%s' with %d rows.", CSV_PATH, len(df_spendings))
        except Exception as exc:
            logger.warning("Failed to load CSV '%s': %s", CSV_PATH, exc)
            df_spendings = None

    yield

    with store_lock:
        _save_store()
    model = None
    df_spendings = None


app = FastAPI(
    title="BrokeMeter Rigorous ML & Telemetry API",
    description="College Student Month-End Zero-Balance Forecaster + Live Presence, Exits & Remote Control",
    lifespan=lifespan,
)

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
frontend_url = os.getenv("FRONTEND_URL", "").strip()
if frontend_url:
    allowed_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClientMeta(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=100)
    page_url: Optional[str] = Field(default=None, max_length=300)
    platform: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default=None, max_length=50)
    timezone: Optional[str] = Field(default=None, max_length=100)
    screen: Optional[str] = Field(default=None, max_length=50)
    viewport: Optional[str] = Field(default=None, max_length=50)


class PresenceEvent(BaseModel):
    event: Literal["open", "heartbeat", "exit"] = "heartbeat"
    session_id: str = Field(..., min_length=1, max_length=100)
    page_url: Optional[str] = Field(default=None, max_length=300)
    platform: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default=None, max_length=50)
    timezone: Optional[str] = Field(default=None, max_length=100)
    screen: Optional[str] = Field(default=None, max_length=50)
    viewport: Optional[str] = Field(default=None, max_length=50)
    duration_sec: Optional[float] = Field(default=None, ge=0)
    exit_reason: Optional[str] = Field(default=None, max_length=80)


class BudgetRequest(BaseModel):
    allowance: float = Field(..., ge=0, description="Total monthly allowance")
    bills: float = Field(..., ge=0, description="Fixed monthly bills/rent")
    daily_spending: Optional[float] = Field(default=None, ge=0, description="Average daily spend")
    weekday_spending: Optional[float] = Field(default=None, ge=0, description="Average weekday spend")
    weekend_spending: Optional[float] = Field(default=None, ge=0, description="Average weekend spend")
    start_date: Optional[str] = Field(default=None, max_length=40)
    client_meta: Optional[ClientMeta] = None


class SiteToggleRequest(BaseModel):
    enabled: bool


def _sync_opened_history(record: Dict[str, Any]) -> None:
    sid = record["session_id"]
    for idx, item in enumerate(state["opened_history"]):
        if item.get("session_id") == sid:
            merged = dict(item)
            merged.update(record)
            merged["open_seq"] = item.get("open_seq", record.get("open_seq", idx + 1))
            merged["opened_at_epoch"] = item.get("opened_at_epoch", record.get("opened_at_epoch"))
            merged["opened_at_utc"] = item.get("opened_at_utc", record.get("opened_at_utc"))
            state["opened_history"][idx] = merged
            return
    state["opened_history"].append(dict(record))
    if len(state["opened_history"]) > MAX_OPENED_HISTORY:
        state["opened_history"] = state["opened_history"][-MAX_OPENED_HISTORY:]


def _upsert_session(
    session_id: str,
    client_ip: str,
    user_agent: str,
    now_dt: datetime,
    meta: Dict[str, Any],
    is_open_event: bool = False,
    is_exit_event: bool = False,
) -> bool:
    existing = state["active_sessions"].get(session_id)
    opened_at_epoch = existing.get("opened_at_epoch", now_dt.timestamp()) if existing else now_dt.timestamp()
    opened_at_utc = existing.get("opened_at_utc", now_dt.isoformat()) if existing else now_dt.isoformat()
    is_new_open = is_open_event or (existing is None)

    if is_new_open:
        state["open_counter"] = int(state.get("open_counter", 0)) + 1
        opened_at_epoch = now_dt.timestamp()
        opened_at_utc = now_dt.isoformat()

    last_seen_epoch = 0.0 if is_exit_event else now_dt.timestamp()

    session_record = {
        "session_id": session_id,
        "ip": client_ip,
        "user_agent": user_agent,
        "page_url": meta.get("page_url") or (existing.get("page_url") if existing else "unknown"),
        "platform": meta.get("platform") or (existing.get("platform") if existing else "unknown"),
        "language": meta.get("language") or (existing.get("language") if existing else "unknown"),
        "timezone": meta.get("timezone") or (existing.get("timezone") if existing else "unknown"),
        "screen": meta.get("screen") or (existing.get("screen") if existing else "unknown"),
        "viewport": meta.get("viewport") or (existing.get("viewport") if existing else "unknown"),
        "opened_at_utc": opened_at_utc,
        "opened_at_epoch": opened_at_epoch,
        "last_seen_utc": now_dt.isoformat(),
        "last_seen_epoch": last_seen_epoch,
        "watching": not is_exit_event,
        "open_seq": state["open_counter"] if is_new_open else (existing.get("open_seq", 0) if existing else 0),
    }

    state["active_sessions"][session_id] = session_record
    _sync_opened_history(session_record)
    return is_new_open


@app.get("/")
def root_status():
    return {
        "service": "BrokeMeter FastAPI Backend",
        "status": "online",
        "site_enabled": bool(state["site_enabled"]),
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "site_enabled": bool(state["site_enabled"]),
        "model_loaded": model is not None,
        "dataset_loaded": df_spendings is not None,
        "dataset_records": int(len(df_spendings)) if df_spendings is not None else 0,
    }


@app.get("/api/site-status")
def get_site_status():
    enabled = bool(state["site_enabled"])
    return {
        "enabled": enabled,
        "maintenance": not enabled,
        "message": "BrokeMeter is live." if enabled else "Site ongoing maintenance.",
    }


@app.post("/api/site-toggle")
def toggle_site(payload: SiteToggleRequest, x_admin_key: Optional[str] = Header(default=None)):
    _verify_admin(x_admin_key)
    with store_lock:
        state["site_enabled"] = bool(payload.enabled)
        _save_store()
    return {
        "enabled": state["site_enabled"],
        "maintenance": not state["site_enabled"],
        "status": "updated",
    }


@app.post("/api/presence")
def track_presence(payload: PresenceEvent, request: Request):
    now_dt = datetime.now(timezone.utc)
    client_ip = _extract_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")
    meta = payload.model_dump()

    with store_lock:
        is_open = payload.event == "open"
        is_exit = payload.event == "exit"
        is_new_open = _upsert_session(
            session_id=payload.session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            now_dt=now_dt,
            meta=meta,
            is_open_event=is_open,
            is_exit_event=is_exit,
        )

        if is_exit:
            existing = state["active_sessions"].get(payload.session_id) or {}
            opened_epoch = existing.get("opened_at_epoch", now_dt.timestamp())
            computed_duration = (
                payload.duration_sec
                if payload.duration_sec is not None
                else max(0.0, now_dt.timestamp() - opened_epoch)
            )
            state["exits"].append({
                "timestamp_utc": now_dt.isoformat(),
                "timestamp_epoch": now_dt.timestamp(),
                "session_id": payload.session_id,
                "ip": client_ip,
                "platform": meta.get("platform") or existing.get("platform") or "unknown",
                "screen": meta.get("screen") or existing.get("screen") or "unknown",
                "viewport": meta.get("viewport") or existing.get("viewport") or "unknown",
                "duration_sec": _safe_float(computed_duration, default=0.0, decimals=1),
                "exit_reason": payload.exit_reason or "page_exit",
                "user_agent": user_agent,
            })

        _save_store()

    return {
        "status": "ok",
        "event": payload.event,
        "new_open": is_new_open,
    }


@app.get("/api/telemetry")
def get_telemetry(x_admin_key: Optional[str] = Header(default=None)):
    _verify_admin(x_admin_key)
    now_ts = datetime.now(timezone.utc).timestamp()
    with store_lock:
        watching_users = [
            dict(info, watching=True)
            for info in state["active_sessions"].values()
            if info.get("watching", True) and (now_ts - info.get("last_seen_epoch", 0)) <= LIVE_WINDOW_SECONDS
        ]
        watching_users.sort(key=lambda x: x.get("opened_at_epoch", 0), reverse=True)

        just_opened_users = watching_users[:15]

        all_opened_users = []
        for item in reversed(state["opened_history"]):
            copy_item = dict(item)
            is_currently_watching = bool(
                copy_item.get("watching", False)
                and (now_ts - copy_item.get("last_seen_epoch", 0)) <= LIVE_WINDOW_SECONDS
            )
            copy_item["currently_watching"] = is_currently_watching
            all_opened_users.append(copy_item)

        exits_list = list(reversed(list(state["exits"])))
        logs_list = list(reversed(list(state["logs"])))
        enabled = bool(state["site_enabled"])
        open_counter = int(state.get("open_counter", len(all_opened_users)))

    return {
        "site_enabled": enabled,
        "live_watching_count": len(watching_users),
        "live_watching_users": watching_users[:50],
        "just_opened_users": just_opened_users,
        "all_opened_users": all_opened_users[:MAX_OPENED_HISTORY],
        "all_opened_count": len(all_opened_users),
        "open_counter": open_counter,
        "exits_count": len(exits_list),
        "exits": exits_list[:100],
        "total_predictions_logged": len(logs_list),
        "predictions": logs_list[:100],
    }


@app.get("/api/campus-inflation")
def get_campus_inflation():
    inflation = 5.2
    if df_spendings is not None and not df_spendings.empty and "inflation_rate" in df_spendings.columns:
        series = pd.to_numeric(df_spendings["inflation_rate"], errors="coerce").dropna()
        if not series.empty:
            inflation = float(series.mean())

    infl_pct = _safe_float(inflation, default=5.2, decimals=2)
    return {
        "inflation_percentage": infl_pct,
        "inflation_multiplier": _safe_float(1.0 + (infl_pct / 100.0), default=1.052, decimals=3),
        "advisory": (
            f"Based on historical CSV records and market trends, campus living costs "
            f"reflect a ~{infl_pct}% inflationary pressure."
        ),
    }


@app.post("/api/predict")
def predict_doom(data: BudgetRequest, request: Request):
    if not state["site_enabled"]:
        raise HTTPException(
            status_code=503,
            detail="Site ongoing maintenance.",
        )

    balance = data.allowance - data.bills
    if balance < 0:
        raise HTTPException(
            status_code=400,
            detail="Recurring bills cannot exceed your total monthly allowance!",
        )

    if data.weekday_spending is not None and data.weekend_spending is not None:
        effective_daily = (data.weekday_spending * 5.0 + data.weekend_spending * 2.0) / 7.0
    elif data.daily_spending is not None:
        effective_daily = float(data.daily_spending)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'daily_spending' or both 'weekday_spending' and 'weekend_spending'.",
        )

    if effective_daily < 0:
        raise HTTPException(status_code=400, detail="Daily spending cannot be negative.")

    if balance == 0:
        predicted_days = 0.0
    elif effective_daily == 0:
        predicted_days = 999.0
    else:
        fallback_days = balance / effective_daily
        predicted_days = fallback_days

        if model is not None:
            try:
                input_df = pd.DataFrame(
                    [[data.allowance, data.bills, effective_daily]],
                    columns=["allowance", "bills", "daily_spending"],
                )
                raw_pred = model.predict(input_df)
                pred_val = float(np.asarray(raw_pred).reshape(-1)[0])
                if not math.isnan(pred_val) and not math.isinf(pred_val):
                    predicted_days = pred_val
            except Exception as exc:
                logger.warning("Model prediction failed, using arithmetic fallback: %s", exc)
                predicted_days = fallback_days

    predicted_days = _safe_float(max(0.0, predicted_days), default=0.0, decimals=1)

    risk_percentage = 15
    risk_score = "Low Risk (Safe)"
    peer_rank = "Top 25% (Disciplined Tier)"

    if balance == 0 and data.allowance > 0:
        risk_percentage = 95
        risk_score = "Critical Danger (100% Committed to Bills)"
    elif effective_daily > 0:
        monthly_run_rate = balance / 30.0
        burn_ratio = (effective_daily / monthly_run_rate) if monthly_run_rate > 0 else 99.0
        risk_percentage = int(min(99, max(5, round(burn_ratio * 35))))

        if risk_percentage > 75:
            risk_score = "Critical Danger (High Overspend Velocity)"
        elif risk_percentage > 45:
            risk_score = "Moderate Warning (Caution Advised)"
        else:
            risk_score = "Low Risk (Sustainable Budget)"
    else:
        risk_percentage = 5
        risk_score = "Low Risk (Zero Daily Burn)"

    if df_spendings is not None and not df_spendings.empty:
        col_allowance = "allowance" if "allowance" in df_spendings.columns else df_spendings.columns[0]
        peer_allowances = pd.to_numeric(df_spendings[col_allowance], errors="coerce").dropna()
        if not peer_allowances.empty:
            percentile = float((peer_allowances < data.allowance).mean() * 100.0)
            if percentile >= 80.0:
                top_pct = max(1, int(round(100.0 - percentile)))
                peer_rank = f"Top {top_pct}% in Hostel (High Allowance Tier)"
            elif percentile >= 40.0:
                peer_rank = "Hostel Median Tier (Average Budget)"
            else:
                peer_rank = "Lean Budget Tier (Highly Frugal)"

    result = {
        "balance": _safe_float(balance, default=0.0, decimals=2),
        "effective_daily_spending": _safe_float(effective_daily, default=0.0, decimals=2),
        "predicted_days_left": predicted_days,
        "peer_rank": peer_rank,
        "risk_score": risk_score,
        "risk_percentage": risk_percentage,
    }

    now_dt = datetime.now(timezone.utc)
    client_ip = _extract_client_ip(request)
    user_agent = request.headers.get("user-agent", "unknown")
    meta = data.client_meta.model_dump() if data.client_meta else {}
    session_id = meta.get("session_id") or f"{client_ip}:{hash(user_agent) & 0xFFFF}"

    event_record = {
        "timestamp_utc": now_dt.isoformat(),
        "session_id": session_id,
        "ip": client_ip,
        "user_agent": user_agent,
        "meta": meta,
        "inputs": {
            "allowance": data.allowance,
            "bills": data.bills,
            "weekday_spending": data.weekday_spending,
            "weekend_spending": data.weekend_spending,
            "daily_spending": effective_daily,
            "start_date": data.start_date,
        },
        "outputs": result,
    }

    with store_lock:
        state["logs"].append(event_record)
        _upsert_session(
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            now_dt=now_dt,
            meta=meta,
            is_open_event=False,
            is_exit_event=False,
        )
        _save_store()

    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    is_dev = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=is_dev)
