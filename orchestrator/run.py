#!/usr/bin/env python3
"""
AEZ Evolution - CLI Runner

Quick way to run simulations and see results.
"""

import argparse
import json
from simulation_v2 import SimulationV2

def main():
    parser = argparse.ArgumentParser(description="AEZ Evolution Simulation")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--agents", type=int, default=10, help="Agents per strategy")
    parser.add_argument("--compute", type=int, default=1000, help="Initial compute")
    parser.add_argument("--stake", type=int, default=50, help="Stake size")
    parser.add_argument("--selection", type=int, default=20, help="Selection interval")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    
    args = parser.parse_args()
    
    # Create simulation
    sim = SimulationV2(
        initial_compute=args.compute,
        stake_size=args.stake,
        rounds_per_matchup=5,
        neighbors_count=4
    )
    
    # Create strategies
    strategies = ["Cooperator", "Defector", "TitForTat", "Grudger", "Random"]
    
    if not args.quiet:
        print("🧬 AEZ Evolution\n")
        print(f"Config: {args.rounds} rounds, {args.agents} agents/strategy")
        print(f"Compute: {args.compute}, Stake: {args.stake}\n")
    
    # Spawn agents
    for strategy_name in strategies:
        genome = sim.create_genome(f"{strategy_name}-Prime", strategy_name)
        for _ in range(args.agents):
            sim.spawn_agent(genome)
    
    sim.setup_neighborhoods()
    
    # Run simulation
    for round_num in range(1, args.rounds + 1):
        sim.run_round()
        
        if round_num % args.selection == 0:
            killed, spawned = sim.selection_cycle()
            if not args.quiet and not args.json:
                print(f"Round {round_num}: killed {killed}, spawned {spawned}")
    
    # Results
    alive = sim.get_alive_agents()
    
    # Strategy breakdown
    stats = {}
    for agent in alive:
        name = agent.strategy.name
        if name not in stats:
            stats[name] = {"count": 0, "fitness": 0, "coop": 0}
        stats[name]["count"] += 1
        stats[name]["fitness"] += agent.fitness_score
        stats[name]["coop"] += agent.cooperation_rate
    
    if args.json:
        result = {
            "rounds": args.rounds,
            "total_alive": len(alive),
            "strategies": {
                name: {
                    "count": s["count"],
                    "avg_fitness": s["fitness"] // s["count"] if s["count"] > 0 else 0,
                    "avg_coop_rate": s["coop"] / s["count"] if s["count"] > 0 else 0
                }
                for name, s in stats.items()
            },
            "top_agents": [
                {
                    "id": a.id,
                    "strategy": a.strategy.name,
                    "fitness": a.fitness_score,
                    "coop_rate": a.cooperation_rate
                }
                for a in sorted(alive, key=lambda x: -x.fitness_score)[:5]
            ]
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print("🏆 FINAL RESULTS")
        print(f"{'='*50}\n")
        
        print(f"{'Strategy':<15} {'Count':>6} {'Avg Fit':>8} {'Coop%':>7}")
        print("-" * 40)
        for name, s in sorted(stats.items(), key=lambda x: -x[1]["count"]):
            count = s["count"]
            avg_fit = s["fitness"] // count if count > 0 else 0
            avg_coop = (s["coop"] / count * 100) if count > 0 else 0
            print(f"{name:<15} {count:>6} {avg_fit:>8} {avg_coop:>6.1f}%")
        
        print(f"\nTotal alive: {len(alive)}")
        
        # Top 5
        print("\n🥇 Top 5 Agents:")
        for i, agent in enumerate(sorted(alive, key=lambda x: -x.fitness_score)[:5], 1):
            print(f"  {i}. {agent.strategy.name}: fitness={agent.fitness_score}")


if __name__ == "__main__":
    main()
