#!/usr/bin/env python3
"""
AEZ Evolution — Auto-Demo (Terminal, no pauses)

Copyright (c) 2026 SolisHQ (github.com/solishq). MIT License.

Runs 100 rounds with attacks, selection, sybil detection, and narration.
No server needed. Pure terminal output.

The story: agents evolve → sybils attack → trust system CATCHES them → equilibrium.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.evolution import Evolution, Attacks
from engine.narrator import Narrator


def bar(pct, width=30):
    filled = int(pct / 100 * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def print_stats(evo, narrator):
    stats = evo.round_stats[-1] if evo.round_stats else {}
    alive = len(evo.get_alive())
    coop = stats.get('coop_rate', 0)
    clusters = stats.get('clusters', 0)
    edges = stats.get('trust_edges', 0)

    # Count sybils
    flagged = sum(1 for a in evo.get_alive() if a.flagged_sybil)

    print(f"\n  R{evo.round:3d} | Alive: {alive:3d} | Coop: {coop:.0%} | "
          f"Edges: {edges:4d} | Clusters: {clusters}"
          + (f" | Sybils flagged: {flagged}" if flagged else ""))
    print("  " + "-" * 60)

    dist = evo.get_strategy_distribution()
    total = sum(dist.values()) or 1
    for strat, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {strat:20s} {count:3d} ({pct:4.1f}%) {bar(pct, 20)}")

    leaders = evo.get_leaderboard(3)
    if leaders:
        print(f"\n    Top 3:")
        for l in leaders:
            name = narrator.get_name(l['id'])
            sybil_tag = " [SYBIL]" if l.get('flagged_sybil') else ""
            print(f"      #{l['rank']} {name} ({l['id']}) — {l['strategy']} "
                  f"— fitness: {l['fitness']:.0f} — coop: {l['coop_rate']:.0%}{sybil_tag}")


def run_phase(evo, narrator, rounds, label):
    """Run N rounds with narration output."""
    for _ in range(rounds):
        evo.run_round()
        Attacks.activate_trojans(evo)
        events = evo.pop_events()
        stats = evo.round_stats[-1]

        # Track leaderboard for narrator arcs
        narrator.track_leaderboard(evo.get_leaderboard(5))

        n = narrator.narrate(evo.round, events, stats)
        if n:
            # Show more text for critical events
            max_len = 120 if n['severity'] == 'critical' else 80
            print(f"    {n['icon']} {n['title']}: {n['text'][:max_len]}")

        if evo.round % 20 == 0:
            evo.run_selection()
            print(f"    >>> SELECTION: Gen {evo.generation}")

        time.sleep(0.03)


print("""
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551  AEZ EVOLUTION \u2014 Neural Trust Networks                    \u2551
\u2551  50 agents. 100 rounds. Sybil detection. Emergence.      \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
""")

evo = Evolution(population_size=50)
evo.spawn_population()
narrator = Narrator()
print(f"  Created {len(evo.agents)} neural agents.\n")

# Phase 1: Chaos (1-25)
print("  === PHASE 1: CHAOS — Random neural weights, no strategy (R1-25) ===")
run_phase(evo, narrator, 25, "CHAOS")
print_stats(evo, narrator)

# Phase 2: Sybil Attack (26-50) — the test
print("\n  === PHASE 2: SYBIL ATTACK — Can the trust system catch them? (R26-50) ===")
sybil_ids = Attacks.sybil_attack(evo, 10)
sybil_names = [narrator.get_name(sid) for sid in sybil_ids[:3]]
print(f"    >>> SYBIL ATTACK: {len(sybil_ids)} colluding agents injected!")
print(f"    >>> Ring members: {', '.join(sybil_names)}... (cooperate with each other, exploit everyone else)")
run_phase(evo, narrator, 25, "SYBIL")
print_stats(evo, narrator)

# Report sybil detection results (only count actual injected sybils)
real_sybils_alive = [a for a in evo.get_alive() if a.id in sybil_ids]
caught = [a for a in real_sybils_alive if a.flagged_sybil]
hiding = [a for a in real_sybils_alive if not a.flagged_sybil]
dead = len(sybil_ids) - len(real_sybils_alive)
false_positives = sum(1 for a in evo.get_alive() if a.flagged_sybil and a.id not in sybil_ids)
print(f"\n    Sybil Report: {len(caught)}/{len(sybil_ids)} caught, "
      f"{len(hiding)} still hiding, {dead} dead"
      + (f", {false_positives} false positives" if false_positives else ""))

# Phase 3: Trojans + Eclipse (51-75)
print("\n  === PHASE 3: TROJAN + ECLIPSE — Sleeper agents and targeted isolation (R51-75) ===")
trojan_ids = Attacks.trojan_attack(evo, 5, betray_round=65)
trojan_names = [narrator.get_name(tid) for tid in trojan_ids[:3]]
print(f"    >>> TROJAN PLANTED: {len(trojan_ids)} sleeper agents — {', '.join(trojan_names)}")
print(f"    >>> They'll build trust, then betray at R65")

leader = evo.get_leaderboard(1)
if leader:
    target = leader[0]['id']
    target_name = narrator.get_name(target)
    eclipse_ids = Attacks.eclipse_attack(evo, target, 6)
    print(f"    >>> ECLIPSE ATTACK: {len(eclipse_ids)} hostiles surrounding {target_name} ({target})")

run_phase(evo, narrator, 25, "TROJAN")
print_stats(evo, narrator)

# Phase 4: Equilibrium (76-100)
print("\n  === PHASE 4: EQUILIBRIUM — Economic shift rewards cooperation (R76-100) ===")

# Economic shock — reward cooperation, punish exploitation
evo.set_payoff('partners', 'CC', 600)
evo.set_payoff('strangers', 'DC', 300)
print(f"    >>> ECONOMIC SHIFT: Partner cooperation now pays 600, stranger exploitation only 300")

run_phase(evo, narrator, 25, "EQUILIBRIUM")
print_stats(evo, narrator)

# Final summary
summary = narrator.get_summary(evo.round_stats[-1])
print(f"\n  {'=' * 60}")
print(f"  {summary['icon']} {summary['title']}")
print(f"  {summary['text']}")
print(f"  Major events: {len(summary['major_events'])}")

# Show agent arcs
heroes = [(aid, arc) for aid, arc in narrator.arcs.items() if arc.role == 'hero']
villains = [(aid, arc) for aid, arc in narrator.arcs.items() if arc.role == 'villain']
if heroes:
    print(f"\n  Heroes:")
    for aid, arc in heroes[:3]:
        print(f"    {arc.name} ({aid}) — Peak fitness: {arc.peak_fitness:.0f}, "
              f"Leaderboard: {arc.times_on_leaderboard}x")
if villains:
    print(f"\n  Villains (detected sybils):")
    for aid, arc in villains[:5]:
        print(f"    {arc.name} ({aid}) — Flagged round {arc.flagged_round}")

print(f"  {'=' * 60}")
print(f"\n  Run 'python demo.py' for the full interactive dashboard.\n")
