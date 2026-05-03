"""
Project 3: Smart Parking Agent System  ── PRODUCTION REFACTOR v2.0
===================================================================
Five targeted fixes applied (each tagged FIX-1 … FIX-5 throughout):

  FIX-1  Deterministic Allocator   – rule-based spot selection, zero LLM calls.
  FIX-2  Real Prediction from DB   – historical occupancy fed into predictor.
  FIX-3  Retry + Fallback          – every LLM call uses tenacity + safe fallback.
  FIX-4  Critic Validation         – critic node re-verifies spot before committing.
  FIX-5  Evaluation Logging        – every pipeline run is recorded in eval_log.
  INFRA  PostgreSQL                – SQLite replaced with a connection-pooled Postgres.

Pipeline (LangGraph):
  monitor → predictor → allocator → critic → alert → END
                                       ↑
                              NEW: FIX-4 critic node

Stack: LangChain · LangGraph · FastAPI · PostgreSQL · tenacity
"""

import os
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TypedDict, Optional, Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ── Logging setup ─────────────────────────────────────────────────────────────
# Structured, timestamped logs – essential for production debugging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("smart_parking")

load_dotenv()

# =============================================================================
# LLM
# =============================================================================
# LLM is now ONLY used for natural-language generation tasks:
#   • predictor_node  – narrate a forecast  (given real DB stats as context)
#   • alert_node      – friendly driver message
# It is NO LONGER used for decisions (allocation is FIX-1 deterministic).
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",   # stable public Groq model
    temperature=0.3,
)

# =============================================================================
# INFRA – PostgreSQL connection pool
# =============================================================================
# Why PostgreSQL instead of SQLite?
#   • Concurrent writes are safe (no file-lock contention under async workers).
#   • Connection pool (min=2, max=10) avoids per-request TCP handshake overhead.
#   • TIMESTAMPTZ columns store UTC properly; queries can use NOW(), intervals.
#   • You can swap the DSN without touching any other code.
#
# Set PARKING_DB_URL in your .env.  The demo default targets a local Postgres.
PARKING_DB_URL: str = os.getenv(
    "PARKING_DB_URL",
    "postgresql://postgres:postgres@localhost:5432/parking_demo",
)

_pg_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """Lazily initialise the global Postgres connection pool."""
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=PARKING_DB_URL,
        )
        logger.info("PostgreSQL connection pool created (min=2, max=10)")
    return _pg_pool


@contextmanager
def get_db():
    """
    Yield a psycopg2 connection from the pool.
    • Auto-commits on clean exit.
    • Rolls back and re-raises on any exception.
    • Always returns the connection to the pool via putconn().
    """
    p = get_pool()
    conn = p.getconn()
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# =============================================================================
# DB – Schema initialisation
# =============================================================================

def init_db(num_spots: int = 20) -> None:
    """
    Create all tables on first run (idempotent – safe to call every startup).

    Tables
    ------
    spots           Live ground-truth of each spot (persists across restarts).
    allocations     One row per parking session; freed_at=NULL means active.
    occupancy_log   FIX-2: Hourly snapshots used for real historical prediction.
    eval_log        FIX-5: One row per /park invocation – enables evaluation.
    """
    ddl = """
    -- ── Parking spots ────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS spots (
        spot_id          TEXT PRIMARY KEY,
        status           TEXT NOT NULL DEFAULT 'free',
        -- 'free' | 'occupied'
        vehicle_type_hint TEXT   -- optional preferred vehicle type per spot
    );

    -- ── Parking sessions ─────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS allocations (
        id              SERIAL       PRIMARY KEY,
        spot_id         TEXT         NOT NULL,
        vehicle_plate   TEXT         NOT NULL,
        vehicle_type    TEXT         NOT NULL DEFAULT 'car',
        allocated_at    TIMESTAMPTZ  NOT NULL,
        freed_at        TIMESTAMPTZ,             -- NULL while the session is active
        forecast        TEXT,
        alert           TEXT,
        critic_verdict  TEXT                     -- FIX-4: stores critic's note
    );

    -- ── FIX-2: Occupancy history for real prediction ──────────────────────────
    -- monitor_node writes here on every run.
    -- predictor_node reads AVG(occupancy_pct) for same hour+weekday.
    CREATE TABLE IF NOT EXISTS occupancy_log (
        id             SERIAL       PRIMARY KEY,
        snapshot_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        hour_of_day    SMALLINT     NOT NULL,   -- 0-23
        day_of_week    SMALLINT     NOT NULL,   -- 0=Mon … 6=Sun
        occupancy_pct  REAL         NOT NULL    -- 0.0–100.0
    );

    -- ── FIX-5: Pipeline evaluation log ───────────────────────────────────────
    -- One row per /park call.  Query this to measure:
    --   allocation_rate, critic_rejection_rate, llm_fallback_rate, p95_latency.
    CREATE TABLE IF NOT EXISTS eval_log (
        id               SERIAL       PRIMARY KEY,
        run_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        vehicle_plate    TEXT,
        vehicle_type     TEXT,
        occupancy_rate   REAL,
        allocated_spot   TEXT,
        critic_verdict   TEXT,
        forecast_used    TEXT,
        alert_generated  TEXT,
        duration_ms      INTEGER,
        used_fallback    BOOLEAN      NOT NULL DEFAULT FALSE,
        success          BOOLEAN      NOT NULL DEFAULT TRUE,
        error_detail     TEXT
    );
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute("SELECT COUNT(*) FROM spots")
            count = cur.fetchone()[0]
            if count == 0:
                # Seed spots only on the very first run
                cur.executemany(
                    "INSERT INTO spots (spot_id, status) VALUES (%s, 'free')",
                    [(f"SPOT-{i + 1}",) for i in range(num_spots)],
                )
                logger.info("Seeded %d parking spots", num_spots)


# =============================================================================
# DB – Helper functions
# =============================================================================

def db_snapshot() -> Dict[str, str]:
    """Return {spot_id: 'free'|'occupied'} – live source of truth."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT spot_id, status FROM spots ORDER BY spot_id")
            return {r["spot_id"]: r["status"] for r in cur.fetchall()}


def db_occupy_spot(
    spot_id: str, plate: str, vtype: str,
    allocated_at: datetime, forecast: str, alert: str, critic_verdict: str,
) -> None:
    """Mark a spot as occupied and insert an allocation record (atomic)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE spots SET status = 'occupied' WHERE spot_id = %s",
                (spot_id,),
            )
            cur.execute(
                """INSERT INTO allocations
                   (spot_id, vehicle_plate, vehicle_type, allocated_at,
                    forecast, alert, critic_verdict)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (spot_id, plate, vtype, allocated_at,
                 forecast, alert, critic_verdict),
            )


def db_free_spot(spot_id: str) -> dict:
    """
    Free a spot and close the active allocation session.

    Returns the closed allocation row as a plain dict.
    Raises ValueError if the spot does not exist or is already free.
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT status FROM spots WHERE spot_id = %s FOR UPDATE",
                (spot_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Spot {spot_id} does not exist.")
            if row["status"] == "free":
                raise ValueError(f"Spot {spot_id} is already free.")

            freed_at = datetime.now(timezone.utc)
            cur.execute(
                "UPDATE spots SET status = 'free' WHERE spot_id = %s",
                (spot_id,),
            )
            cur.execute(
                """UPDATE allocations
                   SET freed_at = %s
                   WHERE spot_id = %s AND freed_at IS NULL""",
                (freed_at, spot_id),
            )
            cur.execute(
                "SELECT * FROM allocations WHERE spot_id = %s AND freed_at = %s",
                (spot_id, freed_at),
            )
            return dict(cur.fetchone())


def db_active_allocations() -> List[dict]:
    """All currently parked vehicles (freed_at IS NULL)."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM allocations WHERE freed_at IS NULL ORDER BY allocated_at"
            )
            return [dict(r) for r in cur.fetchall()]


def db_allocation_history(limit: int = 50) -> List[dict]:
    """Recent allocation history (active + completed)."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM allocations ORDER BY allocated_at DESC LIMIT %s",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]


# ── FIX-2: Occupancy history helpers ─────────────────────────────────────────

def db_log_occupancy(occupancy_pct: float) -> None:
    """
    FIX-2: Record the current occupancy snapshot to occupancy_log.
    Called by monitor_node on every pipeline run.
    After a week of traffic this table gives the predictor real statistics
    (e.g. "Monday 8 AM is historically 72% full").
    """
    now = datetime.now(timezone.utc)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO occupancy_log
                   (snapshot_at, hour_of_day, day_of_week, occupancy_pct)
                   VALUES (%s, %s, %s, %s)""",
                (now, now.hour, now.weekday(), occupancy_pct),
            )


def db_historical_avg_occupancy(
    hour: int, day_of_week: int, lookback_days: int = 14
) -> Optional[float]:
    """
    FIX-2: Query the average occupancy for the same hour + weekday over the
    last `lookback_days` days.  Returns None when fewer than 3 data points
    exist (system is still warming up).

    This replaces the old approach where the predictor had ZERO real context
    and effectively hallucinated a "forecast".
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT AVG(occupancy_pct) AS avg_occ,
                       COUNT(*)            AS n
                FROM   occupancy_log
                WHERE  hour_of_day = %s
                  AND  day_of_week = %s
                  AND  snapshot_at >= NOW() - INTERVAL '1 day' * %s
                """,
                (hour, day_of_week, lookback_days),
            )
            row = cur.fetchone()
    # Require at least 3 historical readings before trusting the average
    if row and row[1] and int(row[1]) >= 3:
        return round(float(row[0]), 1)
    return None


# ── FIX-5: Evaluation log writer ─────────────────────────────────────────────

def db_log_eval(
    vehicle_plate: str,
    vehicle_type: str,
    occupancy_rate: float,
    allocated_spot: Optional[str],
    critic_verdict: str,
    forecast_used: str,
    alert_generated: str,
    duration_ms: int,
    used_fallback: bool,
    success: bool,
    error_detail: str = "",
) -> None:
    """
    FIX-5: Append one evaluation record per /park invocation.

    This gives you a queryable audit trail to answer questions like:
      • What fraction of requests result in a successful allocation?
      • How often does the critic reject an allocation?  (race condition rate)
      • What percentage of LLM calls fall back to the hardcoded string?
      • What is the p95 end-to-end latency?
    Query eval_log or use /eval/summary to see aggregated metrics.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_log
                   (vehicle_plate, vehicle_type, occupancy_rate, allocated_spot,
                    critic_verdict, forecast_used, alert_generated,
                    duration_ms, used_fallback, success, error_detail)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    vehicle_plate, vehicle_type, occupancy_rate, allocated_spot,
                    critic_verdict, forecast_used, alert_generated,
                    duration_ms, used_fallback, success, error_detail,
                ),
            )


# =============================================================================
# FIX-3 – LLM Retry + Fallback wrapper
# =============================================================================
# Problem: A single transient API hiccup killed the entire request in the
# original code because there was no retry logic at all.
#
# Fix: wrap every LLM call in a tenacity retry that:
#   • Retries up to 3 times with exponential back-off (2 s → 4 s → 8 s cap).
#   • Catches any Exception (covers HTTP 429, 503, network timeouts, etc.).
#   • If all attempts fail → returns a safe hardcoded fallback string so the
#     pipeline completes rather than crashing.
#
# llm_invoke_with_retry returns (text, used_fallback: bool) so callers and
# the eval log know when a fallback was triggered.

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=False,   # raises RetryError on exhaustion; caught below
)
def _llm_attempt(messages) -> str:
    """Single LLM attempt; tenacity retries this on any Exception."""
    return llm.invoke(messages).content.strip()


def llm_invoke_with_retry(messages, fallback: str) -> Tuple[str, bool]:
    """
    FIX-3: Call the LLM with automatic retries.

    Returns:
        (response_text, used_fallback)
        used_fallback=True means all retries failed and we returned `fallback`.
    """
    try:
        text = _llm_attempt(messages)
        return text, False
    except (RetryError, Exception) as exc:
        logger.warning("LLM failed after all retries: %s – using fallback", exc)
        return fallback, True


# =============================================================================
# State
# =============================================================================

class ParkingState(TypedDict):
    sensor_snapshot:  Dict[str, str]
    occupancy_rate:   float
    historical_avg:   Optional[float]   # FIX-2: real DB-backed average
    forecast:         Optional[str]
    allocation:       Optional[str]
    allocated_spot:   Optional[str]     # e.g. "SPOT-7"
    critic_verdict:   Optional[str]     # FIX-4: critic's approval/rejection
    alert:            Optional[str]
    vehicle_plate:    str
    vehicle_type:     str
    timestamp:        str
    used_fallback:    bool              # FIX-3: was any LLM fallback triggered?


# =============================================================================
# Nodes
# =============================================================================

def monitor_node(state: ParkingState) -> ParkingState:
    """
    Read live spot state from PostgreSQL (single source of truth – no random
    simulation).  Also writes an occupancy snapshot to occupancy_log (FIX-2)
    and reads the historical average for this hour/weekday (FIX-2).
    """
    now  = datetime.now(timezone.utc)
    data = db_snapshot()
    occupied = sum(1 for v in data.values() if v == "occupied")
    rate     = occupied / max(len(data), 1)

    # FIX-2: Record snapshot so future predictions have real data to draw on.
    db_log_occupancy(round(rate * 100, 1))

    # FIX-2: Fetch the DB-backed historical average for this time slot.
    hist_avg = db_historical_avg_occupancy(hour=now.hour, day_of_week=now.weekday())

    logger.info(
        "Monitor: occupancy=%.1f%%  historical_avg=%s%%",
        rate * 100, hist_avg,
    )

    return {
        **state,
        "sensor_snapshot": data,
        "occupancy_rate":  rate,
        "historical_avg":  hist_avg,
        "timestamp":       now.isoformat(),
    }


def predictor_node(state: ParkingState) -> ParkingState:
    """
    FIX-2: Forecast uses REAL historical data, not a blank LLM guess.

    Before this fix: the LLM was handed only the current occupancy number,
    so it had to invent a forecast with no trend data → "fake prediction".

    After this fix: the LLM receives the DB-computed historical average for
    the same hour/weekday. It can now say "currently 60% but historically 80%
    at this hour on Mondays, so expect it to rise" – a real, grounded forecast.

    FIX-3: The LLM call is wrapped in retry + fallback.
    """
    if state["historical_avg"] is not None:
        hist_ctx = (
            f"Historical average occupancy at this hour/weekday "
            f"(last 14 days): {state['historical_avg']:.1f}%"
        )
    else:
        hist_ctx = "Historical data: not yet available (system warming up)."

    messages = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a parking occupancy forecaster. "
            "Given current occupancy, real historical averages, and time of day, "
            "produce a ONE-sentence forecast of likely occupancy in the next "
            "30 minutes. Reference the historical trend where available."
        )),
        HumanMessage(content=(
            f"Current occupancy : {state['occupancy_rate'] * 100:.1f}%\n"
            f"{hist_ctx}\n"
            f"Timestamp         : {state['timestamp']}\n"
            f"Vehicle type      : {state['vehicle_type']}"
        )),
    ]).format_messages()

    # Fallback used when all LLM retries fail (FIX-3)
    fallback = (
        f"Occupancy is currently {state['occupancy_rate'] * 100:.1f}%; "
        "trend data unavailable due to a temporary service issue."
    )
    forecast, used_fb = llm_invoke_with_retry(messages, fallback=fallback)

    logger.info("Predictor (fallback=%s): %.80s", used_fb, forecast)
    return {
        **state,
        "forecast":      forecast,
        "used_fallback": state.get("used_fallback", False) or used_fb,
    }


def allocator_node(state: ParkingState) -> ParkingState:
    """
    FIX-1: DETERMINISTIC allocation – zero LLM calls.

    Problem with original: the LLM was asked to "pick the best spot",
    which is:
      (a) non-deterministic – two identical requests could get different spots.
      (b) unreliable        – if the LLM call fails the allocation fails.
      (c) expensive         – a simple rule wastes no tokens.
      (d) unparseable       – the LLM text had to be regex-parsed, fragile.

    Replacement – simple priority rules (hard-coded, instant, 100% reproducible):
      motorcycle → prefer SPOT-1..SPOT-3   (small/accessible, near entrance)
      truck      → prefer SPOT-10+         (wide/large spots)
      car        → prefer SPOT-4..SPOT-9   (middle zone)
      fallback   → any remaining free spot when preferred zone is full

    This is the right separation of concerns:
      LLM generates language; rules make structural decisions.
    """
    free_spots: List[str] = sorted(
        [k for k, v in state["sensor_snapshot"].items() if v == "free"],
        key=lambda s: int(s.split("-")[1]),  # numeric sort: SPOT-2 before SPOT-10
    )

    if not free_spots:
        msg = (
            f"No spots available for vehicle {state['vehicle_plate']}. "
            "Redirecting to overflow lot."
        )
        logger.info("Allocator: no free spots for %s", state["vehicle_plate"])
        return {**state, "allocation": msg, "allocated_spot": None}

    vtype = state["vehicle_type"].lower()

    if vtype == "motorcycle":
        preferred     = [s for s in free_spots if int(s.split("-")[1]) <= 3]
        fallback_pool = [s for s in free_spots if s not in preferred]
        zone_label    = "accessible/small zone (SPOT-1..3)"
    elif vtype == "truck":
        preferred     = [s for s in free_spots if int(s.split("-")[1]) >= 10]
        fallback_pool = [s for s in free_spots if s not in preferred]
        zone_label    = "large zone (SPOT-10+)"
    else:  # car or unknown
        preferred     = [
            s for s in free_spots if 4 <= int(s.split("-")[1]) <= 9
        ]
        fallback_pool = [s for s in free_spots if s not in preferred]
        zone_label    = "mid zone (SPOT-4..9)"

    if preferred:
        chosen = preferred[0]
        reason = f"rule: first free in {zone_label}"
    else:
        chosen = fallback_pool[0]
        reason = f"rule: fallback – {zone_label} full, next available spot"

    alloc = f"Assign {chosen} because {reason}."
    logger.info("Allocator: %s → %s (%s)", state["vehicle_plate"], chosen, reason)
    return {**state, "allocation": alloc, "allocated_spot": chosen}


def critic_node(state: ParkingState) -> ParkingState:
    """
    FIX-4: CRITIC VALIDATION.

    Problem this solves: under concurrent traffic, two requests could both
    read the same spot as "free" in their monitor snapshot, and both try to
    allocate it.  Without a critic the second request silently double-books.

    Fix: before writing to DB the critic re-reads the spot's CURRENT status
    directly from Postgres (with FOR UPDATE row-lock in db_free_spot later).
    If the spot was grabbed by another request in the meantime, we reject
    the allocation here and record why (critic_verdict).  The alert_node
    then skips DB persistence and the caller is told to retry.

    Why a separate node (not inline in allocator)?
    The graph makes the validation step visible, auditable, and independently
    measurable via eval_log.critic_verdict counts.
    """
    spot = state.get("allocated_spot")

    if spot is None:
        # Overflow scenario – nothing to validate
        return {**state, "critic_verdict": "N/A: no spot allocated"}

    # Re-read the live DB state (independent of the sensor snapshot taken earlier)
    live_status = db_snapshot().get(spot, "unknown")

    if live_status == "free":
        verdict = f"APPROVED: {spot} confirmed free in live DB"
        logger.info("Critic: %s", verdict)
    else:
        verdict = (
            f"REJECTED: {spot} is '{live_status}' in live DB "
            f"(race condition or stale sensor snapshot). Clearing allocation."
        )
        logger.warning("Critic: %s", verdict)
        # Clear the spot so alert_node does NOT persist a bad allocation
        state = {
            **state,
            "allocated_spot": None,
            "allocation": (
                f"Allocation of {spot} rejected (spot became {live_status} "
                f"between read and write). Please retry."
            ),
        }

    return {**state, "critic_verdict": verdict}


def alert_node(state: ParkingState) -> ParkingState:
    """
    Generate a friendly driver notification via LLM.
    This is the correct use of LLM: natural-language generation,
    NOT a structural decision.

    FIX-3: Call wrapped in retry + fallback.
    FIX-4: Only writes to DB when the critic APPROVED the allocation.
    FIX-5: Evaluation logging happens in the /park endpoint (after this node).
    """
    messages = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an alert agent for a smart parking system. "
            "Based on the allocation and occupancy forecast, write a brief, "
            "friendly 1-2 sentence notification for the driver."
        )),
        HumanMessage(content=(
            f"Allocation : {state['allocation']}\n"
            f"Forecast   : {state['forecast']}\n"
            f"Occupancy  : {state['occupancy_rate'] * 100:.1f}%"
        )),
    ]).format_messages()

    fallback = (
        f"{state['allocation']}  "
        f"Lot is currently {state['occupancy_rate'] * 100:.1f}% full."
    )
    alert, used_fb = llm_invoke_with_retry(messages, fallback=fallback)

    # FIX-4: Skip DB write when critic did not approve a real spot.
    if state.get("allocated_spot"):
        db_occupy_spot(
            spot_id      = state["allocated_spot"],
            plate        = state["vehicle_plate"],
            vtype        = state["vehicle_type"],
            allocated_at = datetime.fromisoformat(state["timestamp"]),
            forecast     = state["forecast"] or "",
            alert        = alert,
            critic_verdict = state.get("critic_verdict", ""),
        )
        logger.info(
            "Alert: spot %s persisted for plate %s",
            state["allocated_spot"], state["vehicle_plate"],
        )

    return {
        **state,
        "alert":        alert,
        "used_fallback": state.get("used_fallback", False) or used_fb,
    }


# =============================================================================
# LangGraph pipeline
# =============================================================================

def build_parking_graph():
    """
    FIX-4: critic node is inserted between allocator and alert.

    monitor → predictor → allocator → critic → alert → END
    """
    g = StateGraph(ParkingState)
    g.add_node("monitor",   monitor_node)
    g.add_node("predictor", predictor_node)
    g.add_node("allocator", allocator_node)
    g.add_node("critic",    critic_node)   # FIX-4: new validation step
    g.add_node("alert",     alert_node)

    g.set_entry_point("monitor")
    g.add_edge("monitor",   "predictor")
    g.add_edge("predictor", "allocator")
    g.add_edge("allocator", "critic")      # FIX-4
    g.add_edge("critic",    "alert")
    g.add_edge("alert",     END)
    return g.compile()


parking_graph = build_parking_graph()

# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(
    title="Smart Parking Agent System",
    description="Production-grade multi-agent parking system (v2.0)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ParkingRequest(BaseModel):
    vehicle_plate: str
    vehicle_type:  str = "car"   # car | motorcycle | truck


class ParkingResponse(BaseModel):
    occupancy_rate:  float
    forecast:        str
    allocation:      str
    allocated_spot:  Optional[str]
    critic_verdict:  str          # FIX-4: tells caller if critic approved/rejected
    alert:           str
    timestamp:       str
    used_fallback:   bool         # FIX-3: transparency – was LLM fallback used?


class CancelRequest(BaseModel):
    spot_id: str                  # e.g. "SPOT-7"


class CancelResponse(BaseModel):
    spot_id:       str
    vehicle_plate: str
    vehicle_type:  str
    allocated_at:  str
    freed_at:      str
    message:       str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/park", response_model=ParkingResponse)
async def park_vehicle(req: ParkingRequest):
    """
    Run the full agent pipeline (monitor → predictor → allocator → critic → alert)
    and return the allocation result.

    FIX-5: Every invocation – success or failure – is recorded in eval_log.
    """
    t0 = time.monotonic()

    initial: ParkingState = {
        "sensor_snapshot": {},
        "occupancy_rate":  0.0,
        "historical_avg":  None,
        "forecast":        None,
        "allocation":      None,
        "allocated_spot":  None,
        "critic_verdict":  None,
        "alert":           None,
        "vehicle_plate":   req.vehicle_plate,
        "vehicle_type":    req.vehicle_type,
        "timestamp":       "",
        "used_fallback":   False,
    }

    try:
        result = parking_graph.invoke(initial)
    except Exception as exc:
        logger.exception("Pipeline error for plate=%s", req.vehicle_plate)
        duration_ms = int((time.monotonic() - t0) * 1000)
        # FIX-5: Log even failed runs so you can track pipeline error rate.
        db_log_eval(
            vehicle_plate  = req.vehicle_plate,
            vehicle_type   = req.vehicle_type,
            occupancy_rate = 0.0,
            allocated_spot = None,
            critic_verdict = "PIPELINE_ERROR",
            forecast_used  = "",
            alert_generated= "",
            duration_ms    = duration_ms,
            used_fallback  = False,
            success        = False,
            error_detail   = str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    duration_ms = int((time.monotonic() - t0) * 1000)

    # FIX-5: Log the successful run.
    db_log_eval(
        vehicle_plate  = result["vehicle_plate"],
        vehicle_type   = result["vehicle_type"],
        occupancy_rate = result["occupancy_rate"],
        allocated_spot = result.get("allocated_spot"),
        critic_verdict = result.get("critic_verdict", ""),
        forecast_used  = result.get("forecast", ""),
        alert_generated= result.get("alert", ""),
        duration_ms    = duration_ms,
        used_fallback  = result.get("used_fallback", False),
        success        = True,
    )

    return ParkingResponse(
        occupancy_rate = result["occupancy_rate"],
        forecast       = result["forecast"] or "",
        allocation     = result["allocation"] or "",
        allocated_spot = result.get("allocated_spot"),
        critic_verdict = result.get("critic_verdict") or "",
        alert          = result["alert"] or "",
        timestamp      = result["timestamp"],
        used_fallback  = result.get("used_fallback", False),
    )


@app.post("/cancel", response_model=CancelResponse)
async def cancel_spot(req: CancelRequest):
    """Free an occupied spot by spot_id."""
    spot_id = req.spot_id.upper().strip()
    try:
        row = db_free_spot(spot_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CancelResponse(
        spot_id       = row["spot_id"],
        vehicle_plate = row["vehicle_plate"],
        vehicle_type  = row["vehicle_type"],
        allocated_at  = str(row["allocated_at"]),
        freed_at      = str(row["freed_at"]),
        message       = f"✅ {spot_id} is now free. Vehicle {row['vehicle_plate']} checked out.",
    )


@app.get("/status")
async def get_status():
    """Live snapshot of every spot from PostgreSQL."""
    data     = db_snapshot()
    free     = sum(1 for v in data.values() if v == "free")
    occupied = len(data) - free
    return {
        "total_spots":   len(data),
        "free":          free,
        "occupied":      occupied,
        "occupancy_pct": round(occupied / max(len(data), 1) * 100, 1),
        "spots":         data,
    }


@app.get("/allocations/active")
async def active_allocations():
    """All vehicles currently parked."""
    return db_active_allocations()


@app.get("/allocations/history")
async def allocation_history(limit: int = 50):
    """Recent allocation history (active + completed)."""
    return db_allocation_history(limit)


@app.get("/eval/summary")
async def eval_summary():
    """
    FIX-5: Aggregated evaluation metrics from eval_log.

    Returns:
      total_runs           – total /park calls
      successful_runs      – pipeline completed without error
      allocated_runs       – calls that resulted in a real spot assignment
      fallback_runs        – calls where at least one LLM fallback was used
      critic_rejections    – calls where the critic rejected the allocation
      avg_duration_ms      – mean end-to-end latency
      avg_occupancy_pct    – mean lot occupancy at time of request
    """
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                                    AS total_runs,
                    SUM(CASE WHEN success          THEN 1 ELSE 0 END)           AS successful_runs,
                    SUM(CASE WHEN allocated_spot IS NOT NULL
                              AND success          THEN 1 ELSE 0 END)           AS allocated_runs,
                    SUM(CASE WHEN used_fallback    THEN 1 ELSE 0 END)           AS fallback_runs,
                    SUM(CASE WHEN critic_verdict LIKE 'REJECTED%%'
                              THEN 1 ELSE 0 END)                                AS critic_rejections,
                    ROUND(AVG(duration_ms)::NUMERIC,        0)                  AS avg_duration_ms,
                    ROUND(AVG(occupancy_rate)::NUMERIC * 100, 1)                AS avg_occupancy_pct
                FROM eval_log
            """)
            return dict(cur.fetchone())


@app.get("/health")
def health():
    """Liveness + DB connectivity check."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status":  "ok" if db_ok else "degraded",
        "service": "SmartParking",
        "version": "2.0.0",
        "db":      "postgres",
        "db_ok":   db_ok,
    }


# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
def on_startup():
    """Initialise DB schema and seed spots on server start."""
    num_spots = int(os.getenv("PARKING_SPOTS", 20))
    try:
        init_db(num_spots)
        logger.info("✅ DB initialised with %d spots", num_spots)
    except Exception as exc:
        logger.error("❌ DB init failed: %s", exc)
        raise   # Fail fast – don't silently start with a broken DB


# Run: uvicorn index:app --reload --port 8003