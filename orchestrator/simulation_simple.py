"""
AEZ Evolution - Simple Simulation (no dependencies)
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum
import hashlib
import os
from abc import ABC, abstractmethod


# ============ STRATEGIES ============

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
    def name(self) -> str:
        return "Cooperator"
    
    def decide(self, history) -> Action:
        return Action.COOPERATE


class Defector(Strategy):
    @property
    def name(self) -> str:
        return "Defector"
    
    def decide(self, history) -> Action:
        return Action.DEFECT


class TitForTat(Strategy):
    @property
    def name(self) -> str:
        return "TitForTat"
    
    def decide(self, history) -> Action:
        if history is None or not history.opponent_actions:
            return Action.COOPERATE
        return history.last_opponent_action()


class Grudger(Strategy):
    @property
    def name(self) -> str:
        return "Grudger"
    
    def decide(self, history) -> Action:
        if history is None:
            return Action.COOPERATE
        if history.opponent_ever_defected():
            return Action.DEFECT
        return Action.COOPERATE


class RandomStrategy(Strategy):
    @property
    def name(self) -> str:
        return "Random"
    
    def decide(self, history) -> Action:
        return random.choice([Action.COOPERATE, Action.DEFECT])


STRATEGIES = {
    "Cooperator": Cooperator,
    "Defector": Defector,
    "TitForTat": TitForTat,
    "Grudger": Grudger,
    "Random": RandomStrategy,
}


def get_strategy(name: str):
    return STRATEGIES[name]()


PAYOFF_MATRIX = {
    (Action.COOPERATE, Action.COOPERATE): (3, 3),
    (Action.COOPERATE, Action.DEFECT): (0, 5),
    (Action.DEFECT, Action.COOPERATE): (5, 0),
    (Action.DEFECT, Action.DEFECT): (1, 1),
}


# ============ SIMULATION ============

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
    
    def get_history_with(self, opponent_id: str):
        return self.history.get(opponent_id)
    
    def record_interaction(self, opponent_id: str, my_action: Action, their_action: Action):
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


class Simulation:
    def __init__(self, initial_compute=1000, stake_size=100):
        self.genomes = {}
        self.agents = {}
        self.round_number = 0
        self.initial_compute = initial_compute
        self.stake_size = stake_size
        
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
    
    def kill_agent(self, agent):
        if not agent.alive:
            return
        agent.alive = False
        genome = self.genomes[agent.genome_id]
        genome.total_fitness += agent.fitness_score
    
    def get_alive_agents(self):
        return [a for a in self.agents.values() if a.alive]
    
    def run_commitment(self, agent_a, agent_b):
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
    
    def run_round(self):
        self.round_number += 1
        alive = self.get_alive_agents()
        
        if len(alive) < 2:
            return
        
        random.shuffle(alive)
        pairs = [(alive[i], alive[i+1]) for i in range(0, len(alive) - 1, 2)]
        
        for agent_a, agent_b in pairs:
            self.run_commitment(agent_a, agent_b)
    
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
            self.spawn_agent(genome)
        
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
            if agent.interactions > 0:
                strategy_stats[name]["coop_rate"] += agent.cooperations / agent.interactions
        
        print(f"{'Strategy':<15} {'Count':>6} {'Avg Fit':>8} {'Coop%':>7}")
        print("-" * 40)
        for name, stats in sorted(strategy_stats.items(), key=lambda x: -x[1]["count"]):
            count = stats["count"]
            avg_fitness = stats["fitness"] // count if count > 0 else 0
            avg_coop = (stats["coop_rate"] / count * 100) if count > 0 else 0
            print(f"{name:<15} {count:>6} {avg_fitness:>8} {avg_coop:>6.1f}%")
        print(f"\nTotal alive: {len(alive)}")


def run_demo():
    print("🧬 AEZ Evolution Demo\n")
    
    sim = Simulation(initial_compute=1000, stake_size=100)
    
    # Create genomes
    strategies = ["Cooperator", "Defector", "TitForTat", "Grudger", "Random"]
    genomes = []
    
    print("Creating genomes...")
    for strategy_name in strategies:
        genome = sim.create_genome(f"{strategy_name}-Prime", strategy_name)
        genomes.append(genome)
        print(f"  Created {genome.name}")
    
    # Spawn agents
    print("\nSpawning agents...")
    for genome in genomes:
        for i in range(10):
            sim.spawn_agent(genome)
    print(f"  Spawned {len(sim.agents)} agents")
    
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
        coop_rate = (agent.cooperations / agent.interactions * 100) if agent.interactions > 0 else 0
        print(f"  {i}. {agent.strategy.name}: fitness={agent.fitness_score}, coop={coop_rate:.0f}%")
    
    return sim


if __name__ == "__main__":
    run_demo()
