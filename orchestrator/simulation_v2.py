"""
AEZ Evolution - Simulation v2: Repeated Games + Reputation

Key insight: TitForTat wins when agents repeatedly interact with the SAME partners.
This version implements:
1. Neighborhood structure (agents have fixed neighbors)
2. Multiple rounds per pair before re-matching
3. Reputation tracking (visible cooperation history)
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from abc import ABC, abstractmethod


class Action(Enum):
    COOPERATE = 0
    DEFECT = 1


@dataclass
class InteractionHistory:
    agent_id: str
    opponent_id: str
    my_actions: list = field(default_factory=list)
    opponent_actions: list = field(default_factory=list)
    
    def last_opponent_action(self):
        return self.opponent_actions[-1] if self.opponent_actions else None
    
    def opponent_ever_defected(self):
        return Action.DEFECT in self.opponent_actions


class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def decide(self, history) -> Action:
        pass


class Cooperator(Strategy):
    @property
    def name(self): return "Cooperator"
    def decide(self, history): return Action.COOPERATE


class Defector(Strategy):
    @property
    def name(self): return "Defector"
    def decide(self, history): return Action.DEFECT


class TitForTat(Strategy):
    @property
    def name(self): return "TitForTat"
    def decide(self, history):
        if history is None or not history.opponent_actions:
            return Action.COOPERATE
        return history.last_opponent_action()


class Grudger(Strategy):
    @property
    def name(self): return "Grudger"
    def decide(self, history):
        if history is None:
            return Action.COOPERATE
        if history.opponent_ever_defected():
            return Action.DEFECT
        return Action.COOPERATE


class RandomStrategy(Strategy):
    @property
    def name(self): return "Random"
    def decide(self, history):
        return random.choice([Action.COOPERATE, Action.DEFECT])


STRATEGIES = {
    "Cooperator": Cooperator,
    "Defector": Defector,
    "TitForTat": TitForTat,
    "Grudger": Grudger,
    "Random": RandomStrategy,
}


def get_strategy(name): return STRATEGIES[name]()


PAYOFF_MATRIX = {
    (Action.COOPERATE, Action.COOPERATE): (3, 3),
    (Action.COOPERATE, Action.DEFECT): (0, 5),
    (Action.DEFECT, Action.COOPERATE): (5, 0),
    (Action.DEFECT, Action.DEFECT): (1, 1),
}


@dataclass
class Agent:
    id: str
    genome_id: str
    strategy: Strategy
    compute_balance: int
    generation: int = 0
    fitness_score: int = 0
    interactions: int = 0
    cooperations: int = 0
    defections: int = 0
    alive: bool = True
    history: dict = field(default_factory=dict)
    neighbors: list = field(default_factory=list)  # Fixed neighborhood
    
    @property
    def cooperation_rate(self):
        if self.interactions == 0:
            return 0.5  # Unknown = neutral
        return self.cooperations / self.interactions
    
    def get_history_with(self, opponent_id):
        return self.history.get(opponent_id)
    
    def record_interaction(self, opponent_id, my_action, their_action):
        if opponent_id not in self.history:
            self.history[opponent_id] = InteractionHistory(self.id, opponent_id)
        self.history[opponent_id].my_actions.append(my_action)
        self.history[opponent_id].opponent_actions.append(their_action)
        self.interactions += 1
        if my_action == Action.COOPERATE:
            self.cooperations += 1
        else:
            self.defections += 1


@dataclass
class Genome:
    id: str
    name: str
    strategy_name: str
    generation: int = 0
    total_spawned: int = 0
    total_fitness: int = 0


class SimulationV2:
    """
    Improved simulation with:
    - Spatial neighborhoods (agents interact with fixed neighbors)
    - Multiple rounds per matchup (iterated prisoner's dilemma)
    - Reputation visible (can see opponent's cooperation history)
    """
    
    def __init__(self, initial_compute=1000, stake_size=50, rounds_per_matchup=5, neighbors_count=4):
        self.genomes = {}
        self.agents = {}
        self.round_number = 0
        self.initial_compute = initial_compute
        self.stake_size = stake_size
        self.rounds_per_matchup = rounds_per_matchup
        self.neighbors_count = neighbors_count
        
    def create_genome(self, name, strategy_name):
        genome_id = f"genome_{len(self.genomes)}"
        genome = Genome(genome_id, name, strategy_name)
        self.genomes[genome_id] = genome
        return genome
    
    def spawn_agent(self, genome):
        agent_id = f"agent_{len(self.agents)}"
        agent = Agent(
            id=agent_id,
            genome_id=genome.id,
            strategy=get_strategy(genome.strategy_name),
            compute_balance=self.initial_compute,
            generation=genome.generation
        )
        self.agents[agent_id] = agent
        genome.total_spawned += 1
        return agent
    
    def setup_neighborhoods(self):
        """Connect agents to random neighbors"""
        alive = self.get_alive_agents()
        for agent in alive:
            # Pick random neighbors (excluding self)
            others = [a for a in alive if a.id != agent.id]
            agent.neighbors = random.sample(others, min(self.neighbors_count, len(others)))
    
    def kill_agent(self, agent):
        if not agent.alive:
            return
        agent.alive = False
        genome = self.genomes[agent.genome_id]
        genome.total_fitness += agent.fitness_score
    
    def get_alive_agents(self):
        return [a for a in self.agents.values() if a.alive]
    
    def run_commitment(self, agent_a, agent_b):
        """Run a single interaction"""
        stake = min(self.stake_size, agent_a.compute_balance, agent_b.compute_balance)
        if stake <= 0:
            return 0, 0
        
        history_a = agent_a.get_history_with(agent_b.id)
        history_b = agent_b.get_history_with(agent_a.id)
        
        action_a = agent_a.strategy.decide(history_a)
        action_b = agent_b.strategy.decide(history_b)
        
        payoff_a, payoff_b = PAYOFF_MATRIX[(action_a, action_b)]
        
        reward_a = (payoff_a * stake) // 3
        reward_b = (payoff_b * stake) // 3
        
        agent_a.compute_balance = agent_a.compute_balance - stake + reward_a
        agent_b.compute_balance = agent_b.compute_balance - stake + reward_b
        
        agent_a.record_interaction(agent_b.id, action_a, action_b)
        agent_b.record_interaction(agent_a.id, action_b, action_a)
        
        agent_a.fitness_score = agent_a.compute_balance
        agent_b.fitness_score = agent_b.compute_balance
        
        return (1 if action_a == Action.COOPERATE else 0,
                1 if action_b == Action.COOPERATE else 0)
    
    def run_matchup(self, agent_a, agent_b):
        """Run multiple rounds between two agents (iterated PD)"""
        total_coops = 0
        for _ in range(self.rounds_per_matchup):
            c_a, c_b = self.run_commitment(agent_a, agent_b)
            total_coops += c_a + c_b
        return total_coops
    
    def run_round(self):
        """Each agent plays with all their neighbors"""
        self.round_number += 1
        alive = self.get_alive_agents()
        
        if len(alive) < 2:
            return
        
        # Each agent plays with their neighbors
        played = set()
        for agent in alive:
            for neighbor in agent.neighbors:
                if neighbor.alive:
                    pair_key = tuple(sorted([agent.id, neighbor.id]))
                    if pair_key not in played:
                        self.run_matchup(agent, neighbor)
                        played.add(pair_key)
    
    def selection_cycle(self, kill_pct=0.1, reproduce_pct=0.2):
        alive = self.get_alive_agents()
        if len(alive) < 5:
            return 0, 0
        
        sorted_agents = sorted(alive, key=lambda a: a.fitness_score)
        
        kill_count = max(1, int(len(sorted_agents) * kill_pct))
        for agent in sorted_agents[:kill_count]:
            self.kill_agent(agent)
        
        reproduce_count = max(1, int(len(sorted_agents) * reproduce_pct))
        for agent in sorted_agents[-reproduce_count:]:
            genome = self.genomes[agent.genome_id]
            new_agent = self.spawn_agent(genome)
            # New agent inherits some neighbors from parent
            new_agent.neighbors = random.sample(
                [a for a in self.get_alive_agents() if a.id != new_agent.id],
                min(self.neighbors_count, len(self.get_alive_agents()) - 1)
            )
        
        # Refresh neighborhoods
        self.setup_neighborhoods()
        
        return kill_count, reproduce_count
    
    def print_status(self):
        alive = self.get_alive_agents()
        print(f"\n=== Round {self.round_number} Status ===")
        
        strategy_stats = {}
        for agent in alive:
            name = agent.strategy.name
            if name not in strategy_stats:
                strategy_stats[name] = {"count": 0, "fitness": 0, "coop_rate": 0}
            strategy_stats[name]["count"] += 1
            strategy_stats[name]["fitness"] += agent.fitness_score
            strategy_stats[name]["coop_rate"] += agent.cooperation_rate
        
        print(f"{'Strategy':<15} {'Count':>6} {'Avg Fit':>8} {'Coop%':>7}")
        print("-" * 40)
        for name, stats in sorted(strategy_stats.items(), key=lambda x: -x[1]["count"]):
            count = stats["count"]
            avg_fitness = stats["fitness"] // count if count > 0 else 0
            avg_coop = (stats["coop_rate"] / count * 100) if count > 0 else 0
            print(f"{name:<15} {count:>6} {avg_fitness:>8} {avg_coop:>6.1f}%")
        print(f"\nTotal alive: {len(alive)}")


def run_demo():
    print("🧬 AEZ Evolution v2 - Iterated Games Demo\n")
    print("Changes from v1:")
    print("  - Agents have fixed neighbors")
    print("  - Multiple rounds per matchup (iterated PD)")
    print("  - This should let TitForTat shine!\n")
    
    sim = SimulationV2(
        initial_compute=1000, 
        stake_size=50,
        rounds_per_matchup=5,  # 5 rounds per matchup
        neighbors_count=4       # Each agent has 4 neighbors
    )
    
    # Create genomes
    strategies = ["Cooperator", "Defector", "TitForTat", "Grudger", "Random"]
    genomes = []
    
    print("Creating genomes...")
    for strategy_name in strategies:
        genome = sim.create_genome(f"{strategy_name}-Prime", strategy_name)
        genomes.append(genome)
    
    # Spawn agents
    print("Spawning agents...")
    for genome in genomes:
        for i in range(10):
            sim.spawn_agent(genome)
    print(f"  Spawned {len(sim.agents)} agents")
    
    # Setup neighborhoods
    sim.setup_neighborhoods()
    
    # Initial status
    sim.print_status()
    
    # Run simulation
    print("\nRunning simulation...")
    for round_num in range(1, 101):
        sim.run_round()
        
        if round_num % 20 == 0:
            killed, spawned = sim.selection_cycle()
            print(f"\n🔄 Selection @ round {round_num}: killed {killed}, spawned {spawned}")
            sim.print_status()
    
    # Final results
    print("\n" + "=" * 50)
    print("🏆 FINAL RESULTS")
    sim.print_status()
    
    # Top agents
    alive = sim.get_alive_agents()
    top_5 = sorted(alive, key=lambda a: -a.fitness_score)[:5]
    
    print("\nTop 5 Agents:")
    for i, agent in enumerate(top_5, 1):
        print(f"  {i}. {agent.strategy.name}: fitness={agent.fitness_score}, coop={agent.cooperation_rate*100:.0f}%")
    
    # Strategy breakdown
    print("\n📊 Strategy Analysis:")
    for genome in sim.genomes.values():
        alive_count = len([a for a in sim.get_alive_agents() if a.genome_id == genome.id])
        print(f"  {genome.name}: started=10, alive={alive_count}, total_spawned={genome.total_spawned}")
    
    return sim


if __name__ == "__main__":
    run_demo()
