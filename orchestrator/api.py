"""
AEZ Evolution - REST API

FastAPI server to control and observe the simulation.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
from datetime import datetime

# Import our simulation
from simulation_v2 import SimulationV2, Action

app = FastAPI(
    title="AEZ Evolution API",
    description="Control autonomous economic agents evolving through game theory",
    version="0.1.0"
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
simulation: Optional[SimulationV2] = None
running = False
events_queue = []


# ============ MODELS ============

class SimulationConfig(BaseModel):
    initial_compute: int = 1000
    stake_size: int = 50
    rounds_per_matchup: int = 5
    neighbors_count: int = 4
    agents_per_strategy: int = 10
    strategies: list[str] = ["Cooperator", "Defector", "TitForTat", "Grudger", "Random"]


class RunConfig(BaseModel):
    rounds: int = 100
    selection_interval: int = 20
    kill_pct: float = 0.1
    reproduce_pct: float = 0.2


class AgentInfo(BaseModel):
    id: str
    strategy: str
    genome_id: str
    compute_balance: int
    fitness_score: int
    interactions: int
    cooperation_rate: float
    generation: int
    alive: bool


class StrategyStats(BaseModel):
    name: str
    count: int
    avg_fitness: int
    avg_coop_rate: float
    total_spawned: int


class SimulationStatus(BaseModel):
    running: bool
    round_number: int
    total_agents: int
    alive_agents: int
    strategies: list[StrategyStats]
    events_count: int


# ============ ENDPOINTS ============

@app.get("/")
async def root():
    return {
        "name": "AEZ Evolution API",
        "version": "0.1.0",
        "status": "running" if running else "idle",
        "simulation": simulation is not None
    }


@app.post("/simulation/create")
async def create_simulation(config: SimulationConfig):
    """Create a new simulation with the given configuration"""
    global simulation, events_queue
    
    simulation = SimulationV2(
        initial_compute=config.initial_compute,
        stake_size=config.stake_size,
        rounds_per_matchup=config.rounds_per_matchup,
        neighbors_count=config.neighbors_count
    )
    events_queue = []
    
    # Create genomes and spawn agents
    for strategy_name in config.strategies:
        genome = simulation.create_genome(f"{strategy_name}-Prime", strategy_name)
        for _ in range(config.agents_per_strategy):
            simulation.spawn_agent(genome)
    
    # Setup neighborhoods
    simulation.setup_neighborhoods()
    
    return {
        "status": "created",
        "total_agents": len(simulation.agents),
        "strategies": config.strategies,
        "config": config.dict()
    }


@app.get("/simulation/status", response_model=SimulationStatus)
async def get_status():
    """Get current simulation status"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    alive = simulation.get_alive_agents()
    
    # Calculate strategy stats
    strategy_stats = {}
    for agent in alive:
        name = agent.strategy.name
        if name not in strategy_stats:
            strategy_stats[name] = {
                "count": 0, 
                "fitness": 0, 
                "coop_rate": 0,
                "total_spawned": 0
            }
        strategy_stats[name]["count"] += 1
        strategy_stats[name]["fitness"] += agent.fitness_score
        strategy_stats[name]["coop_rate"] += agent.cooperation_rate
    
    # Add total spawned from genomes
    for genome in simulation.genomes.values():
        if genome.strategy_name in strategy_stats:
            strategy_stats[genome.strategy_name]["total_spawned"] = genome.total_spawned
    
    strategies = []
    for name, stats in strategy_stats.items():
        count = stats["count"]
        strategies.append(StrategyStats(
            name=name,
            count=count,
            avg_fitness=stats["fitness"] // count if count > 0 else 0,
            avg_coop_rate=(stats["coop_rate"] / count) if count > 0 else 0,
            total_spawned=stats["total_spawned"]
        ))
    
    return SimulationStatus(
        running=running,
        round_number=simulation.round_number,
        total_agents=len(simulation.agents),
        alive_agents=len(alive),
        strategies=sorted(strategies, key=lambda x: -x.count),
        events_count=len(events_queue)
    )


@app.get("/agents")
async def get_agents(alive_only: bool = True, limit: int = 100):
    """Get all agents or just alive ones"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    agents = simulation.get_alive_agents() if alive_only else list(simulation.agents.values())
    
    return {
        "count": len(agents),
        "agents": [
            AgentInfo(
                id=a.id,
                strategy=a.strategy.name,
                genome_id=a.genome_id,
                compute_balance=a.compute_balance,
                fitness_score=a.fitness_score,
                interactions=a.interactions,
                cooperation_rate=a.cooperation_rate,
                generation=a.generation,
                alive=a.alive
            )
            for a in agents[:limit]
        ]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get details for a specific agent"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    if agent_id not in simulation.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = simulation.agents[agent_id]
    return AgentInfo(
        id=agent.id,
        strategy=agent.strategy.name,
        genome_id=agent.genome_id,
        compute_balance=agent.compute_balance,
        fitness_score=agent.fitness_score,
        interactions=agent.interactions,
        cooperation_rate=agent.cooperation_rate,
        generation=agent.generation,
        alive=agent.alive
    )


@app.post("/simulation/round")
async def run_single_round():
    """Run a single round of the simulation"""
    global events_queue
    
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    simulation.run_round()
    
    # Capture events
    events_queue.append({
        "type": "round_complete",
        "round": simulation.round_number,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "round": simulation.round_number,
        "alive_agents": len(simulation.get_alive_agents())
    }


@app.post("/simulation/selection")
async def run_selection(kill_pct: float = 0.1, reproduce_pct: float = 0.2):
    """Run a selection cycle"""
    global events_queue
    
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    killed, spawned = simulation.selection_cycle(kill_pct, reproduce_pct)
    
    events_queue.append({
        "type": "selection_cycle",
        "round": simulation.round_number,
        "killed": killed,
        "spawned": spawned,
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "killed": killed,
        "spawned": spawned,
        "alive_agents": len(simulation.get_alive_agents())
    }


@app.post("/simulation/run")
async def run_simulation(config: RunConfig, background_tasks: BackgroundTasks):
    """Run the full simulation in background"""
    global running
    
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    if running:
        raise HTTPException(status_code=400, detail="Simulation already running")
    
    async def run_async():
        global running, events_queue
        running = True
        
        try:
            for round_num in range(1, config.rounds + 1):
                simulation.run_round()
                
                events_queue.append({
                    "type": "round_complete",
                    "round": simulation.round_number,
                    "timestamp": datetime.now().isoformat()
                })
                
                if round_num % config.selection_interval == 0:
                    killed, spawned = simulation.selection_cycle(
                        config.kill_pct, 
                        config.reproduce_pct
                    )
                    events_queue.append({
                        "type": "selection_cycle",
                        "round": simulation.round_number,
                        "killed": killed,
                        "spawned": spawned,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Small delay to allow other requests
                await asyncio.sleep(0.01)
        finally:
            running = False
    
    background_tasks.add_task(run_async)
    
    return {
        "status": "started",
        "config": config.dict()
    }


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop a running simulation"""
    global running
    running = False
    return {"status": "stopped"}


@app.get("/events")
async def get_events(limit: int = 100, offset: int = 0):
    """Get recent events"""
    return {
        "total": len(events_queue),
        "events": events_queue[offset:offset + limit]
    }


@app.get("/events/stream")
async def stream_events():
    """Server-sent events for real-time updates"""
    from fastapi.responses import StreamingResponse
    
    async def event_generator():
        last_count = 0
        while True:
            if len(events_queue) > last_count:
                for event in events_queue[last_count:]:
                    yield f"data: {json.dumps(event)}\n\n"
                last_count = len(events_queue)
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get top agents by fitness"""
    if simulation is None:
        raise HTTPException(status_code=404, detail="No simulation created")
    
    alive = simulation.get_alive_agents()
    top = sorted(alive, key=lambda a: -a.fitness_score)[:limit]
    
    return {
        "leaderboard": [
            {
                "rank": i + 1,
                "id": a.id,
                "strategy": a.strategy.name,
                "fitness": a.fitness_score,
                "cooperation_rate": a.cooperation_rate,
                "interactions": a.interactions
            }
            for i, a in enumerate(top)
        ]
    }


# ============ MAIN ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
