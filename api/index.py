# """
# Project 3: Smart Parking Agent System
# ======================================
# Proactive, intelligent parking management agent.

# Agents:
#   - MonitorAgent    → polls sensor data (simulated)
#   - PredictorAgent  → forecasts occupancy
#   - AllocatorAgent  → assigns spots to incoming vehicles
#   - AlertAgent      → notifies drivers / operators

# Stack: LangChain · LangGraph · FastAPI
# """

# import os
# import random
# import json
# import operator

# from datetime import datetime
# from typing import TypedDict, Annotated, List, Optional, Dict

# from dotenv import load_dotenv
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langgraph.graph import StateGraph, END

# # Load .env file
# load_dotenv()

# # Initialize LLM
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="openai/gpt-oss-120b",
#     temperature=0.3
# )

# # ── Sensor Simulator ──────────────────────────────────────────────────────────
# def simulate_sensor_data(num_spots: int = 20) -> Dict:
#     """Returns a dict of spot_id -> status (occupied/free)."""
#     return {f"SPOT-{i+1}": random.choice(["occupied", "occupied", "free"])
#             for i in range(num_spots)}

# # ── State ─────────────────────────────────────────────────────────────────────
# class ParkingState(TypedDict):
#     sensor_snapshot: Dict[str, str]
#     occupancy_rate: float
#     forecast: Optional[str]
#     allocation: Optional[str]
#     alert: Optional[str]
#     vehicle_plate: str
#     vehicle_type: str
#     timestamp: str

# # ── Nodes ─────────────────────────────────────────────────────────────────────

# def monitor_node(state: ParkingState) -> ParkingState:
#     """Collect & summarise sensor data."""
#     data = simulate_sensor_data()
#     occupied = sum(1 for v in data.values() if v == "occupied")
#     rate = occupied / len(data)
#     return {**state, "sensor_snapshot": data, "occupancy_rate": rate,
#             "timestamp": datetime.utcnow().isoformat()}


# def predictor_node(state: ParkingState) -> ParkingState:
#     """Use LLM to forecast near-term occupancy trend."""
#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content=(
#             "You are a parking occupancy forecaster. "
#             "Given current occupancy and time of day, give a 1-sentence "
#             "forecast of occupancy in the next 30 minutes."
#         )),
#         HumanMessage(content=(
#             f"Current occupancy: {state['occupancy_rate']*100:.1f}%\n"
#             f"Timestamp: {state['timestamp']}\n"
#             f"Vehicle type arriving: {state['vehicle_type']}"
#         )),
#     ])
#     forecast = llm.invoke(prompt.format_messages()).content.strip()
#     return {**state, "forecast": forecast}


# def allocator_node(state: ParkingState) -> ParkingState:
#     """Assign the best available spot to the incoming vehicle."""
#     free_spots = [k for k, v in state["sensor_snapshot"].items() if v == "free"]
#     if not free_spots:
#         alloc = f"No spots available for vehicle {state['vehicle_plate']}. Redirecting to overflow lot."
#     else:
#         # LLM picks best spot considering vehicle type
#         prompt = ChatPromptTemplate.from_messages([
#             SystemMessage(content=(
#                 "You are a parking allocation agent. Select the optimal spot "
#                 "from the free list for the vehicle type. Prefer accessible "
#                 "spots (SPOT-1 to SPOT-3) for motorcycles, larger spots "
#                 "(SPOT-10+) for trucks. Respond: 'Assign SPOT-XX because <reason>'"
#             )),
#             HumanMessage(content=(
#                 f"Vehicle: {state['vehicle_plate']} ({state['vehicle_type']})\n"
#                 f"Free spots: {', '.join(free_spots)}"
#             )),
#         ])
#         alloc = llm.invoke(prompt.format_messages()).content.strip()
#     return {**state, "allocation": alloc}


# def alert_node(state: ParkingState) -> ParkingState:
#     """Generate alert messages for operators or drivers."""
#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content=(
#             "You are an alert generation agent for a smart parking system. "
#             "Based on the allocation and forecast, generate a brief, friendly "
#             "notification for the driver (1-2 sentences)."
#         )),
#         HumanMessage(content=(
#             f"Allocation: {state['allocation']}\n"
#             f"Forecast: {state['forecast']}\n"
#             f"Occupancy: {state['occupancy_rate']*100:.1f}%"
#         )),
#     ])
#     alert = llm.invoke(prompt.format_messages()).content.strip()
#     return {**state, "alert": alert}

# # ── Graph ─────────────────────────────────────────────────────────────────────
# def build_parking_graph():
#     g = StateGraph(ParkingState)
#     g.add_node("monitor", monitor_node)
#     g.add_node("predictor", predictor_node)
#     g.add_node("allocator", allocator_node)
#     g.add_node("alert", alert_node)

#     g.set_entry_point("monitor")
#     g.add_edge("monitor", "predictor")
#     g.add_edge("predictor", "allocator")
#     g.add_edge("allocator", "alert")
#     g.add_edge("alert", END)
#     return g.compile()

# parking_graph = build_parking_graph()

# # ── FastAPI ───────────────────────────────────────────────────────────────────
# app = FastAPI(title="Smart Parking Agent System")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ParkingRequest(BaseModel):
#     vehicle_plate: str
#     vehicle_type: str = "car"   # car | motorcycle | truck

# class ParkingResponse(BaseModel):
#     occupancy_rate: float
#     forecast: str
#     allocation: str
#     alert: str
#     timestamp: str

# @app.post("/park", response_model=ParkingResponse)
# async def park_vehicle(req: ParkingRequest):
#     initial: ParkingState = {
#         "sensor_snapshot": {},
#         "occupancy_rate": 0.0,
#         "forecast": None,
#         "allocation": None,
#         "alert": None,
#         "vehicle_plate": req.vehicle_plate,
#         "vehicle_type": req.vehicle_type,
#         "timestamp": "",
#     }
#     result = parking_graph.invoke(initial)
#     return ParkingResponse(
#         occupancy_rate=result["occupancy_rate"],
#         forecast=result["forecast"] or "",
#         allocation=result["allocation"] or "",
#         alert=result["alert"] or "",
#         timestamp=result["timestamp"],
#     )

# @app.get("/status")
# async def get_status():
#     data = simulate_sensor_data()
#     free = sum(1 for v in data.values() if v == "free")
#     return {"total_spots": len(data), "free": free, "occupied": len(data) - free,
#             "occupancy_pct": round((len(data) - free) / len(data) * 100, 1)}

# @app.get("/health")
# def health():
#     return {"status": "ok", "service": "SmartParking"}

# # Run: uvicorn smart_parking:app --reload --port 8003




















"""
Project 3: Smart Parking Agent System
======================================
Proactive, intelligent parking management agent.

Agents:
  - MonitorAgent    → polls sensor data (simulated + SQLite-backed)
  - PredictorAgent  → forecasts occupancy
  - AllocatorAgent  → assigns spots to incoming vehicles
  - AlertAgent      → notifies drivers / operators

Stack: LangChain · LangGraph · FastAPI · SQLite
"""

import os
import random
import sqlite3
import threading
import json

from contextlib import contextmanager
from datetime import datetime
from typing import TypedDict, Optional, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.3,
)

# ── SQLite Setup ──────────────────────────────────────────────────────────────
DB_PATH = os.getenv("PARKING_DB", "parking.db")
_db_lock = threading.Lock()          # serialise writes from async workers


@contextmanager
def get_db():
    """Thread-safe SQLite connection context manager."""
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db(num_spots: int = 20) -> None:
    """
    Create tables and seed spot rows on first run.

    Tables
    ------
    spots       – ground-truth of every parking spot (persists across restarts)
    allocations – one active row per occupied spot; history kept via freed_at
    """
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS spots (
                spot_id     TEXT PRIMARY KEY,
                status      TEXT NOT NULL DEFAULT 'free'   -- 'free' | 'occupied'
            );

            CREATE TABLE IF NOT EXISTS allocations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                spot_id         TEXT    NOT NULL,
                vehicle_plate   TEXT    NOT NULL,
                vehicle_type    TEXT    NOT NULL DEFAULT 'car',
                allocated_at    TEXT    NOT NULL,
                freed_at        TEXT,                      -- NULL while active
                forecast        TEXT,
                alert           TEXT
            );
        """)

        # Seed spots only when the table is empty
        count = conn.execute("SELECT COUNT(*) FROM spots").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO spots (spot_id, status) VALUES (?, 'free')",
                [(f"SPOT-{i+1}",) for i in range(num_spots)],
            )


# ── DB Helpers ────────────────────────────────────────────────────────────────

def db_snapshot() -> Dict[str, str]:
    """Return {spot_id: 'free'|'occupied'} from the DB (source of truth)."""
    with get_db() as conn:
        rows = conn.execute("SELECT spot_id, status FROM spots").fetchall()
    return {r["spot_id"]: r["status"] for r in rows}


def db_occupy_spot(spot_id: str, plate: str, vtype: str,
                   allocated_at: str, forecast: str, alert: str) -> None:
    """Mark a spot as occupied and insert an allocation record."""
    with get_db() as conn:
        conn.execute(
            "UPDATE spots SET status = 'occupied' WHERE spot_id = ?",
            (spot_id,),
        )
        conn.execute(
            """INSERT INTO allocations
               (spot_id, vehicle_plate, vehicle_type, allocated_at, forecast, alert)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (spot_id, plate, vtype, allocated_at, forecast, alert),
        )


def db_free_spot(spot_id: str) -> sqlite3.Row:
    """
    Mark spot as free and close the latest active allocation.

    Returns the closed allocation row, or raises ValueError if the spot
    was already free.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT status FROM spots WHERE spot_id = ?", (spot_id,)
        ).fetchone()

        if row is None:
            raise ValueError(f"Spot {spot_id} does not exist.")
        if row["status"] == "free":
            raise ValueError(f"Spot {spot_id} is already free.")

        freed_at = datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE spots SET status = 'free' WHERE spot_id = ?", (spot_id,)
        )
        conn.execute(
            """UPDATE allocations
               SET freed_at = ?
               WHERE spot_id = ? AND freed_at IS NULL""",
            (freed_at, spot_id),
        )

        # Return the row we just closed
        return conn.execute(
            "SELECT * FROM allocations WHERE spot_id = ? AND freed_at = ?",
            (spot_id, freed_at),
        ).fetchone()


def db_active_allocations():
    """All currently active (not freed) allocations."""
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM allocations WHERE freed_at IS NULL ORDER BY allocated_at"
        ).fetchall()


def db_allocation_history(limit: int = 50):
    """Recent allocation history (freed + active)."""
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM allocations ORDER BY allocated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


# ── State ─────────────────────────────────────────────────────────────────────
class ParkingState(TypedDict):
    sensor_snapshot: Dict[str, str]
    occupancy_rate: float
    forecast: Optional[str]
    allocation: Optional[str]
    allocated_spot: Optional[str]   # parsed spot id (e.g. "SPOT-7")
    alert: Optional[str]
    vehicle_plate: str
    vehicle_type: str
    timestamp: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def monitor_node(state: ParkingState) -> ParkingState:
    """Read real spot state from SQLite (no random noise in production)."""
    data = db_snapshot()
    occupied = sum(1 for v in data.values() if v == "occupied")
    rate = occupied / max(len(data), 1)
    return {
        **state,
        "sensor_snapshot": data,
        "occupancy_rate": rate,
        "timestamp": datetime.utcnow().isoformat(),
    }


def predictor_node(state: ParkingState) -> ParkingState:
    """LLM forecasts near-term occupancy trend."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a parking occupancy forecaster. "
            "Given current occupancy and time of day, give a 1-sentence "
            "forecast of occupancy in the next 30 minutes."
        )),
        HumanMessage(content=(
            f"Current occupancy: {state['occupancy_rate'] * 100:.1f}%\n"
            f"Timestamp: {state['timestamp']}\n"
            f"Vehicle type arriving: {state['vehicle_type']}"
        )),
    ])
    forecast = llm.invoke(prompt.format_messages()).content.strip()
    return {**state, "forecast": forecast}


def allocator_node(state: ParkingState) -> ParkingState:
    """Assign best free spot; persist the choice to SQLite."""
    free_spots = [k for k, v in state["sensor_snapshot"].items() if v == "free"]

    if not free_spots:
        alloc = (
            f"No spots available for vehicle {state['vehicle_plate']}. "
            "Redirecting to overflow lot."
        )
        return {**state, "allocation": alloc, "allocated_spot": None}

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a parking allocation agent. Select the optimal spot "
            "from the free list for the vehicle type. Prefer accessible "
            "spots (SPOT-1 to SPOT-3) for motorcycles, larger spots "
            "(SPOT-10+) for trucks. "
            "Respond EXACTLY in this format (no extra text): "
            "'Assign SPOT-XX because <reason>'"
        )),
        HumanMessage(content=(
            f"Vehicle: {state['vehicle_plate']} ({state['vehicle_type']})\n"
            f"Free spots: {', '.join(sorted(free_spots))}"
        )),
    ])
    alloc = llm.invoke(prompt.format_messages()).content.strip()

    # Parse spot id from LLM reply ("Assign SPOT-7 because …")
    allocated_spot = None
    for word in alloc.split():
        if word.upper().startswith("SPOT-"):
            candidate = word.upper().rstrip(".,;:")
            if candidate in state["sensor_snapshot"]:
                allocated_spot = candidate
                break

    return {**state, "allocation": alloc, "allocated_spot": allocated_spot}


def alert_node(state: ParkingState) -> ParkingState:
    """Generate driver notification; write final allocation to SQLite."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an alert generation agent for a smart parking system. "
            "Based on the allocation and forecast, generate a brief, friendly "
            "notification for the driver (1-2 sentences)."
        )),
        HumanMessage(content=(
            f"Allocation: {state['allocation']}\n"
            f"Forecast: {state['forecast']}\n"
            f"Occupancy: {state['occupancy_rate'] * 100:.1f}%"
        )),
    ])
    alert = llm.invoke(prompt.format_messages()).content.strip()

    # Persist to DB only when a real spot was assigned
    if state.get("allocated_spot"):
        db_occupy_spot(
            spot_id=state["allocated_spot"],
            plate=state["vehicle_plate"],
            vtype=state["vehicle_type"],
            allocated_at=state["timestamp"],
            forecast=state["forecast"] or "",
            alert=alert,
        )

    return {**state, "alert": alert}


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_parking_graph():
    g = StateGraph(ParkingState)
    g.add_node("monitor",   monitor_node)
    g.add_node("predictor", predictor_node)
    g.add_node("allocator", allocator_node)
    g.add_node("alert",     alert_node)

    g.set_entry_point("monitor")
    g.add_edge("monitor",   "predictor")
    g.add_edge("predictor", "allocator")
    g.add_edge("allocator", "alert")
    g.add_edge("alert",     END)
    return g.compile()


parking_graph = build_parking_graph()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Parking Agent System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ParkingRequest(BaseModel):
    vehicle_plate: str
    vehicle_type: str = "car"           # car | motorcycle | truck


class ParkingResponse(BaseModel):
    occupancy_rate: float
    forecast: str
    allocation: str
    allocated_spot: Optional[str]
    alert: str
    timestamp: str


class CancelRequest(BaseModel):
    spot_id: str                        # e.g. "SPOT-7"


class CancelResponse(BaseModel):
    spot_id: str
    vehicle_plate: str
    vehicle_type: str
    allocated_at: str
    freed_at: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/park", response_model=ParkingResponse)
async def park_vehicle(req: ParkingRequest):
    """Run the full agent pipeline and assign a spot."""
    initial: ParkingState = {
        "sensor_snapshot": {},
        "occupancy_rate": 0.0,
        "forecast": None,
        "allocation": None,
        "allocated_spot": None,
        "alert": None,
        "vehicle_plate": req.vehicle_plate,
        "vehicle_type": req.vehicle_type,
        "timestamp": "",
    }
    result = parking_graph.invoke(initial)
    return ParkingResponse(
        occupancy_rate=result["occupancy_rate"],
        forecast=result["forecast"] or "",
        allocation=result["allocation"] or "",
        allocated_spot=result.get("allocated_spot"),
        alert=result["alert"] or "",
        timestamp=result["timestamp"],
    )


@app.post("/cancel", response_model=CancelResponse)
async def cancel_spot(req: CancelRequest):
    """
    Free an occupied spot by spot_id.

    - Marks the spot as 'free' in the DB.
    - Closes the active allocation record (sets freed_at).
    - Returns the closed allocation details.
    """
    spot_id = req.spot_id.upper().strip()
    try:
        row = db_free_spot(spot_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CancelResponse(
        spot_id=row["spot_id"],
        vehicle_plate=row["vehicle_plate"],
        vehicle_type=row["vehicle_type"],
        allocated_at=row["allocated_at"],
        freed_at=row["freed_at"],
        message=f"✅ {spot_id} is now free. Vehicle {row['vehicle_plate']} has been checked out.",
    )


@app.get("/status")
async def get_status():
    """Live snapshot of every spot from the DB."""
    data = db_snapshot()
    free     = sum(1 for v in data.values() if v == "free")
    occupied = len(data) - free
    return {
        "total_spots":    len(data),
        "free":           free,
        "occupied":       occupied,
        "occupancy_pct":  round(occupied / max(len(data), 1) * 100, 1),
        "spots":          data,         # full map for dashboard use
    }


@app.get("/allocations/active")
async def active_allocations():
    """All vehicles currently parked."""
    rows = db_active_allocations()
    return [dict(r) for r in rows]


@app.get("/allocations/history")
async def allocation_history(limit: int = 50):
    """Recent allocation history (completed + active)."""
    rows = db_allocation_history(limit)
    return [dict(r) for r in rows]


@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartParking", "db": DB_PATH}


# ── Bootstrap ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    num_spots = int(os.getenv("PARKING_SPOTS", 20))
    init_db(num_spots)


# Run: uvicorn smart_parking:app --reload --port 8003