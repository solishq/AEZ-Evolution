"""
AEZ Evolution — Server

Copyright (c) 2026 SolisHQ (github.com/solishq). All rights reserved.
Licensed under MIT. Built for Colosseum Hackathon 2026.

FastAPI + WebSocket. God-mode API.
Every control a judge needs. Real-time updates.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from .evolution import Evolution, Attacks
from .narrator import Narrator


# ─── State ──────────────────────────────────────────────

evo: Optional[Evolution] = None
narrator = Narrator()
ws_clients: set[WebSocket] = set()
auto_running = False
auto_task = None
_evo_lock = asyncio.Lock()


# ─── App ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global auto_running
    auto_running = False

app = FastAPI(title="AEZ Evolution", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard
dashboard_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard')
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir), name="dashboard")


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "aez-evolution", "spec": "0x534f4c49"}


@app.get("/")
async def root():
    index = os.path.join(dashboard_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "AEZ Evolution API", "version": "2.0"}


# ─── Models ─────────────────────────────────────────────

class CreateRequest(BaseModel):
    population: int = Field(default=50, ge=2, le=500)

class RunRequest(BaseModel):
    rounds: int = Field(default=1, ge=1, le=5000)
    selection_interval: int = Field(default=20, ge=0, le=1000)

class PayoffRequest(BaseModel):
    tier: str = "strangers"
    key: str = "CC"
    value: float = Field(default=200, ge=-500, le=500)

class AttackRequest(BaseModel):
    type: str
    count: int = Field(default=5, ge=1, le=200)
    target: Optional[str] = None


# ─── Simulation Control ────────────────────────────────

@app.post("/sim/create")
async def create_sim(req: CreateRequest):
    global evo, narrator, auto_running
    auto_running = False
    evo = Evolution(population_size=req.population)
    evo.spawn_population()
    narrator = Narrator()
    network = evo.get_network_data()
    leaderboard = evo.get_leaderboard(8)
    narrator.track_leaderboard(leaderboard)
    await broadcast({
        "type": "created",
        "data": network,
        "leaderboard": leaderboard,
        "agent_names": narrator.get_agent_names()
    })
    return {"status": "created", "agents": len(evo.agents)}


@app.post("/sim/round")
async def run_round():
    if not evo:
        return {"error": "No simulation. POST /sim/create first."}

    async with _evo_lock:
        evo.run_round()
        Attacks.activate_trojans(evo)

        events = evo.pop_events()
        stats = evo.round_stats[-1] if evo.round_stats else {}
        narration = narrator.narrate(evo.round, events, stats)

        leaderboard = evo.get_leaderboard(8)
        narrator.track_leaderboard(leaderboard)

        network = evo.get_network_data()

    await broadcast({
        "type": "round",
        "data": network,
        "events": events[:10],
        "narration": narration,
        "leaderboard": leaderboard,
        "agent_names": narrator.get_agent_names()
    })

    return {
        "round": evo.round,
        "alive": len(evo.get_alive()),
        "narration": narration
    }


@app.post("/sim/selection")
async def run_selection():
    if not evo:
        return {"error": "No simulation."}
    async with _evo_lock:
        evo.run_selection()
        events = evo.pop_events()
    await broadcast({"type": "selection", "events": events[:10]})
    return {"generation": evo.generation, "alive": len(evo.get_alive())}


@app.post("/sim/run")
async def run_multi(req: RunRequest):
    """Run multiple rounds (with selection intervals)."""
    if not evo:
        return {"error": "No simulation."}

    for i in range(req.rounds):
        async with _evo_lock:
            evo.run_round()
            Attacks.activate_trojans(evo)

            if req.selection_interval > 0 and evo.round % req.selection_interval == 0:
                evo.run_selection()

            events = evo.pop_events()
            stats = evo.round_stats[-1] if evo.round_stats else {}
            narration = narrator.narrate(evo.round, events, stats)

            leaderboard = evo.get_leaderboard(8)
            narrator.track_leaderboard(leaderboard)

            network = evo.get_network_data()

        await broadcast({
            "type": "round",
            "data": network,
            "events": events[:10],
            "narration": narration,
            "leaderboard": leaderboard,
            "agent_names": narrator.get_agent_names()
        })

        await asyncio.sleep(0.05)

    return {"rounds_completed": req.rounds, "total_round": evo.round}


@app.post("/sim/auto")
async def toggle_auto():
    """Toggle auto-running (one round per second)."""
    global auto_running, auto_task

    if auto_running:
        auto_running = False
        if auto_task:
            auto_task.cancel()
            try:
                await auto_task
            except asyncio.CancelledError:
                pass
            auto_task = None
        return {"auto": False}

    if not evo:
        return {"error": "No simulation."}

    auto_running = True

    async def auto_loop():
        global auto_running
        try:
            while auto_running and evo:
                async with _evo_lock:
                    evo.run_round()
                    Attacks.activate_trojans(evo)

                    if evo.round % 20 == 0:
                        evo.run_selection()

                    events = evo.pop_events()
                    stats = evo.round_stats[-1] if evo.round_stats else {}
                    narration = narrator.narrate(evo.round, events, stats)

                    leaderboard = evo.get_leaderboard(8)
                    narrator.track_leaderboard(leaderboard)

                    network = evo.get_network_data()

                await broadcast({
                    "type": "round",
                    "data": network,
                    "events": events[:10],
                    "narration": narration,
                    "leaderboard": leaderboard,
                    "agent_names": narrator.get_agent_names()
                })

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            auto_running = False

    auto_task = asyncio.create_task(auto_loop())
    return {"auto": True}


# ─── God Mode Controls ─────────────────────────────────

@app.post("/sim/payoff")
async def change_payoff(req: PayoffRequest):
    """Change payoff values mid-simulation. Economic disruption."""
    if not evo:
        return {"error": "No simulation."}
    evo.set_payoff(req.tier, req.key, req.value)
    events = evo.pop_events()
    await broadcast({"type": "payoff_change", "events": events})
    return {"payoff_matrices": evo.payoff_matrices}


@app.post("/sim/attack")
async def launch_attack(req: AttackRequest):
    """Inject adversarial agents."""
    if not evo:
        return {"error": "No simulation."}

    ids = []
    if req.type == "sybil":
        ids = Attacks.sybil_attack(evo, req.count)
    elif req.type == "trojan":
        ids = Attacks.trojan_attack(evo, req.count)
    elif req.type == "eclipse" and req.target:
        ids = Attacks.eclipse_attack(evo, req.target, req.count)
    elif req.type == "whitewash":
        ids = Attacks.whitewash_attack(evo, req.count)

    events = evo.pop_events()
    await broadcast({"type": "attack", "attack_type": req.type, "events": events})
    return {"attack": req.type, "agents_injected": ids}


@app.post("/sim/detect")
async def detect_sybils():
    """Manually trigger immune system detection cycle."""
    if not evo:
        return {"error": "No simulation."}
    flagged = evo.immune.run_cycle(evo.agents, evo.trust_net, evo.round)
    events = evo.pop_events()
    immune_events = evo.immune.pop_events()
    events.extend(immune_events)
    all_flagged = [a.id for a in evo.get_alive() if a.flagged_sybil]
    if events:
        stats = evo.round_stats[-1] if evo.round_stats else {}
        narration = narrator.narrate(evo.round, events, stats)
        await broadcast({"type": "detection", "events": events, "narration": narration})
    return {"flagged": all_flagged, "count": len(all_flagged), "newly_flagged": list(flagged)}


@app.post("/sim/inject")
async def inject_agent():
    """Inject a single random agent (for judges to play with)."""
    if not evo:
        return {"error": "No simulation."}
    from .agent import NeuralAgent
    agent = NeuralAgent(id=f"J{evo.next_id + 1:04d}", generation=evo.generation)
    evo.next_id += 1
    agent.balance = 800
    evo.agents[agent.id] = agent
    return {"injected": agent.to_dict()}


# ─── Query ──────────────────────────────────────────────

@app.get("/sim/state")
async def get_state():
    if not evo:
        return {"error": "No simulation."}
    return evo.get_network_data()


@app.get("/sim/leaderboard")
async def get_leaderboard(limit: int = 10):
    if not evo:
        return {"error": "No simulation."}
    return {"leaderboard": evo.get_leaderboard(limit)}


@app.get("/sim/strategies")
async def get_strategies():
    if not evo:
        return {"error": "No simulation."}
    return {"distribution": evo.get_strategy_distribution()}


@app.get("/sim/stats")
async def get_stats():
    if not evo:
        return {"error": "No simulation."}
    alive = evo.get_alive()
    import numpy as np
    return {
        "round": evo.round,
        "generation": evo.generation,
        "alive": len(alive),
        "total_agents": len(evo.agents),
        "payoff_matrices": evo.payoff_matrices,
        "trust_edges": len(evo.trust_net.edges),
        "immune_warnings_total": sum(a.warnings_emitted for a in alive),
        "immune_memory_total": sum(len(a.threat_memory) for a in alive),
        "avg_vigilance": round(float(np.mean([a.vigilance for a in alive])), 3) if alive else 0,
        "flagged_sybils": sum(1 for a in alive if a.flagged_sybil),
        "round_stats": evo.round_stats[-20:]
    }


# ─── WebSocket ──────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        if evo:
            await ws.send_json({
                "type": "init",
                "data": evo.get_network_data(),
                "leaderboard": evo.get_leaderboard(5)
            })
        while True:
            data = await ws.receive_text()
            if len(data) > 10_000:
                continue
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "get_state" and evo:
                await ws.send_json({
                    "type": "state",
                    "data": evo.get_network_data()
                })
    except WebSocketDisconnect:
        ws_clients.discard(ws)
    except Exception:
        ws_clients.discard(ws)


async def broadcast(message: dict):
    """Send to all connected WebSocket clients."""
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


# ─── Run ────────────────────────────────────────────────

def start(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
