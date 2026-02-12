"""
AEZ Evolution API Server

FastAPI server for the dashboard to interact with simulations.
"""

import asyncio
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .simulation import Simulation, create_default_simulation

app = FastAPI(
    title="AEZ Evolution API",
    description="Control panel for Autonomous Economic Zones evolution simulation",
    version="0.1.0"
)

# Allow CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
simulation: Optional[Simulation] = None
running_task: Optional[asyncio.Task] = None


# Request/Response models
class CreateSimRequest(BaseModel):
    agents_per_strategy: int = 10
    stake: int = 100
    initial_compute: int = 1000


class RunRequest(BaseModel):
    rounds: int = 100
    selection_interval: int = 20


class StatusResponse(BaseModel):
    round_number: int
    total_agents: int
    alive_agents: int
    running: bool
    events_count: int
    strategies: List[dict]


class AgentResponse(BaseModel):
    id: str
    strategy: str
    compute_balance: int
    fitness: int
    interactions: int
    cooperations: int
    defections: int
    alive: bool
    rank: int = 0


class LeaderboardResponse(BaseModel):
    leaderboard: List[AgentResponse]


class EventResponse(BaseModel):
    round: int
    event_type: str
    data: dict


# Endpoints

@app.get("/")
async def root():
    return {"message": "AEZ Evolution API", "version": "0.1.0"}


@app.get("/simulation/status", response_model=StatusResponse)
async def get_status():
    """Get current simulation status"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    status = simulation.get_status()
    return StatusResponse(
        round_number=status["round_number"],
        total_agents=status["total_agents"],
        alive_agents=status["alive_agents"],
        running=status["running"],
        events_count=status["events_count"],
        strategies=status["strategies"]
    )


@app.post("/simulation/create")
async def create_simulation(request: CreateSimRequest = CreateSimRequest()):
    """Create a new simulation"""
    global simulation

    simulation = Simulation(stake=request.stake)

    # Create agents for each strategy - all 7 strategies
    strategies = ["Cooperator", "Defector", "TitForTat", "Grudger", "Random", "Pavlov", "SuspiciousTitForTat"]
    for strategy_name in strategies:
        for _ in range(request.agents_per_strategy):
            simulation.create_agent(strategy_name, initial_compute=request.initial_compute)

    return {
        "status": "created",
        "total_agents": len(simulation.agents),
        "strategies": strategies
    }


@app.post("/simulation/round")
async def run_round():
    """Run a single round"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    commitments = simulation.run_round()
    
    return {
        "round": simulation.round_number,
        "commitments": len(commitments),
        "alive": len(simulation.get_alive_agents())
    }


@app.post("/simulation/selection")
async def run_selection():
    """Run selection pressure"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    simulation.run_selection()
    
    return {
        "status": "selection complete",
        "alive": len(simulation.get_alive_agents())
    }


@app.post("/simulation/run")
async def run_simulation(request: RunRequest):
    """Run multiple rounds with periodic selection"""
    global running_task
    
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    if simulation.running:
        raise HTTPException(status_code=400, detail="Simulation already running")
    
    async def run_async():
        simulation.running = True
        try:
            for r in range(request.rounds):
                if not simulation.running:
                    break
                simulation.run_round()
                if (r + 1) % request.selection_interval == 0:
                    simulation.run_selection()
                await asyncio.sleep(0.01)  # Yield for API responsiveness
        finally:
            simulation.running = False
    
    running_task = asyncio.create_task(run_async())
    
    return {
        "status": "started",
        "rounds": request.rounds,
        "selection_interval": request.selection_interval
    }


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop running simulation"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    simulation.running = False
    
    return {"status": "stopped"}


@app.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(limit: int = 10):
    """Get top agents by fitness"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    leaders = simulation.get_leaderboard(limit)
    
    return LeaderboardResponse(
        leaderboard=[
            AgentResponse(
                id=a.id,
                strategy=a.strategy_name,
                compute_balance=a.compute_balance,
                fitness=a.fitness_score,
                interactions=a.interactions,
                cooperations=a.cooperations,
                defections=a.defections,
                alive=a.alive,
                rank=i+1
            )
            for i, a in enumerate(leaders)
        ]
    )


@app.get("/events")
async def get_events(limit: int = 50, offset: int = 0):
    """Get recent events"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    events = simulation.events[-limit-offset:-offset] if offset else simulation.events[-limit:]
    
    return {
        "events": [
            {
                "round": e.round,
                "type": e.event_type,
                "data": e.data
            }
            for e in events
        ],
        "total": len(simulation.events)
    }


@app.get("/agent/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get agent details"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation running")
    
    agent = simulation.agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AgentResponse(
        id=agent.id,
        strategy=agent.strategy_name,
        compute_balance=agent.compute_balance,
        fitness=agent.fitness_score,
        interactions=agent.interactions,
        cooperations=agent.cooperations,
        defections=agent.defections,
        alive=agent.alive
    )


def main():
    """Run the server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
