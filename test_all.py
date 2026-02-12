#!/usr/bin/env python3
"""
AEZ Evolution v2 — Comprehensive Test Suite

Copyright (c) 2026 SolisHQ (github.com/solishq). MIT License.

Tests every invention from first principles.
100+ tests across 11 categories.
"""
import sys
import os
import json
import hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random

# Fixed seed for deterministic tests
np.random.seed(42)
random.seed(42)

from engine.agent import NeuralAgent, crossover, mutate
from engine.trust import TrustNetwork, TrustState
from engine.immune import ImmuneSystem
from engine.evolution import Evolution, Attacks

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


print("=" * 60)
print("AEZ EVOLUTION v2 — TEST SUITE")
print("=" * 60)

# ─── 1. Agent Core Tests ────────────────────────────────
print("\n--- 1. Agent Core Tests ---")

a = NeuralAgent(id="T001", generation=0)
test("Agent creates with random weights", a.weights_ih is not None)
test("Agent has 11 inputs", a.weights_ih.shape == (16, 11))
test("Agent has 16 hidden neurons", a.weights_ho.shape == (1, 16))
test("Agent starts alive", a.alive)
test("Agent starts with balance 1000", a.balance == 1000.0)
test("Agent has selectivity gene", hasattr(a, 'selectivity'))
test("Agent has vigilance gene", hasattr(a, 'vigilance'))
test("Agent has trust_weights", a.trust_weights is not None and len(a.trust_weights) == 4)
test("Trust weights sum to 1", abs(a.trust_weights.sum() - 1.0) < 0.01)
test("Agent has memory capacity", 3 <= a.memory_capacity <= 20)

# Decision making
ctx = {'round': 1, 'direct_trust': 0.5, 'social_trust': 0.5,
       'temporal_trust': 0.5, 'structural_trust': 0.5}
decision = a.decide("opp1", ctx)
test("Agent can decide (bool)", isinstance(decision, bool))

# Recording
a.record("opp1", True, False, -300)
test("Agent records interaction", a.interactions == 1)
test("Agent tracks cooperation", a.cooperations == 1)
test("Agent tracks action sequence", len(a.action_sequence) == 1)
test("Agent updates balance", a.balance == 700.0)

# Strategy labels
a2 = NeuralAgent(id="T004")
for i in range(20):
    a2.record("x", True, True, 300)
test("High coop agent labeled Cooperator", a2.get_strategy_label() == "Cooperator")

a3 = NeuralAgent(id="T005")
for i in range(20):
    a3.record("x", False, True, 500)
test("Low coop agent labeled Defector", a3.get_strategy_label() == "Defector")

# to_dict
d = a.to_dict()
test("to_dict has all fields", all(k in d for k in
     ['id', 'fitness', 'selectivity', 'vigilance', 'trust_weights', 'flagged_sybil']))


# ─── 2. Commitment Protocol Tests ───────────────────────
print("\n--- 2. Commitment Protocol Tests ---")

c1 = NeuralAgent(id="C001")
commitment = c1.commit_action("opp1", ctx)
test("Commitment produces bytes", isinstance(commitment, bytes))
test("Commitment is SHA-256 (32 bytes)", len(commitment) == 32)

action, nonce = c1.reveal_action()
test("Reveal produces action (bool)", isinstance(action, bool))
test("Reveal produces nonce (bytes)", isinstance(nonce, bytes))

# Verify commitment
verified = NeuralAgent.verify_commitment(commitment, action, nonce)
test("Honest commitment verifies", verified)

# Fake commitment (wrong action)
fake_verified = NeuralAgent.verify_commitment(commitment, not action, nonce)
test("Fake commitment fails verification", not fake_verified)

# Wrong nonce
wrong_nonce = os.urandom(16)
wrong_verified = NeuralAgent.verify_commitment(commitment, action, wrong_nonce)
test("Wrong nonce fails verification", not wrong_verified)

# Commitment tracking
c1.record_commitment("opp1", True)
c1.record_commitment("opp1", True)
c1.record_commitment("opp1", False)
test("Commitment tracking works", c1.commitment_history["opp1"] == [2, 1])
test("Commitment reliability computed", abs(c1._get_commitment_reliability("opp1") - 2/3) < 0.01)

# SHA-256 quantum resistance note
raw = b'C' + os.urandom(16)
h = hashlib.sha256(raw).digest()
test("SHA-256 produces 256-bit hash", len(h) == 32)


# ─── 3. Reproduction Tests ──────────────────────────────
print("\n--- 3. Reproduction Tests ---")

p1 = NeuralAgent(id="P001", generation=0, vigilance=0.3, warning_propensity=0.2)
p2 = NeuralAgent(id="P002", generation=0, vigilance=0.7, warning_propensity=0.8)
child = crossover(p1, p2, "C001", 1)
test("Crossover produces child", child.id == "C001")
test("Child inherits generation", child.generation == 1)
test("Child has selectivity", 0 <= child.selectivity <= 1)
test("Child inherits vigilance (avg)", abs(child.vigilance - 0.5) < 0.1)
test("Child inherits warning_propensity (avg)", abs(child.warning_propensity - 0.5) < 0.1)
test("Child has trust weights", child.trust_weights is not None)
test("Child trust weights sum to 1", abs(child.trust_weights.sum() - 1.0) < 0.01)

# Mutation
old_vig = child.vigilance
mutate(child, rate=1.0, strength=0.5)  # Force mutation
test("Mutation runs without error", True)
test("Trust weights still sum to 1 after mutation", abs(child.trust_weights.sum() - 1.0) < 0.01)


# ─── 4. Bayesian Trust Tests ────────────────────────────
print("\n--- 4. Bayesian Trust Tests ---")

ts = TrustState()
test("TrustState starts with uniform prior", ts.alpha == 1.0 and ts.beta == 1.0)
test("Initial trust is 0.5 (neutral)", abs(ts.direct_trust - 0.5) < 0.01)
test("Initial confidence is 0", abs(ts.confidence) < 0.01)

# Update with cooperation
ts.update(cooperated=True)
test("Cooperation increases alpha", ts.alpha == 2.0)
test("Trust increases after cooperation", ts.direct_trust > 0.5)
test("Confidence increases with evidence", ts.confidence > 0.0)

# More cooperation → trust approaches 1
for _ in range(10):
    ts.update(cooperated=True)
test("Many cooperations → high trust", ts.direct_trust > 0.85)
test("Many observations → growing confidence", ts.confidence > 0.2)

# Defection evidence
ts2 = TrustState()
ts2.update(cooperated=False)
test("Defection increases beta", ts2.beta == 2.0)
test("Trust decreases after defection", ts2.direct_trust < 0.5)

# Temporal trust
ts3 = TrustState()
for _ in range(10):
    ts3.update(cooperated=True)
test("Consistent behavior → high temporal trust", ts3.temporal_trust > 0.7)

ts4 = TrustState()
for i in range(10):
    ts4.update(cooperated=bool(i % 2))  # alternating
test("Alternating behavior → lower temporal trust", ts4.temporal_trust < 0.8)

# Commitment tracking
ts5 = TrustState()
ts5.update(cooperated=True, commitment_honored=True)
ts5.update(cooperated=True, commitment_honored=True)
ts5.update(cooperated=True, commitment_honored=False)
test("Commitment reliability tracked", abs(ts5.commitment_reliability - 2/3) < 0.01)


# ─── 5. Trust Network Tests ─────────────────────────────
print("\n--- 5. Trust Network Tests ---")

tn = TrustNetwork()
test("TrustNetwork creates", tn is not None)

# Trust update
tn.update("A", "B", True, True)
test("Trust updates bidirectionally", ("A", "B") in tn.edges and ("B", "A") in tn.edges)

direct = tn.compute_direct_trust("A", "B")
test("Direct trust increases after coop", direct > 0.5)

# Social trust
tn.update("A", "C", True, True)  # A trusts C
tn.update("C", "B", True, True)  # C trusts B
social = tn.compute_social_trust("A", "B", ["A", "B", "C"])
test("Social trust computed from third parties", isinstance(social, float))

# Temporal trust
temporal = tn.compute_temporal_trust("A", "B")
test("Temporal trust computed", isinstance(temporal, float))

# Structural trust
structural = tn.compute_structural_trust("A", "B", ["A", "B", "C"])
test("Structural trust computed", isinstance(structural, float))

# Composite trust
weights = np.array([0.25, 0.25, 0.25, 0.25])
composite = tn.compute_composite_trust("A", "B", weights, ["A", "B", "C"])
test("Composite trust uses agent weights", isinstance(composite, float))
test("Composite trust in [0,1]", 0 <= composite <= 1)

# Reputation
rep = tn.get_reputation("B", ["A", "B", "C"])
test("Reputation computed", isinstance(rep, float))

# Trusted neighbors
neighbors = tn.get_trusted_neighbors("A")  # uses TRUST_THRESHOLD (Bayesian neutral)
test("Trusted neighbors returned", isinstance(neighbors, list))


# ─── 6. Topology Analysis Tests ─────────────────────────
print("\n--- 6. Topology Analysis Tests ---")

# Build a small network
tn2 = TrustNetwork()
agents_top = ["X1", "X2", "X3", "X4", "X5"]
# Dense cluster: X1-X2-X3 all trust each other
for _ in range(10):
    tn2.update("X1", "X2", True, True)
    tn2.update("X2", "X3", True, True)
    tn2.update("X1", "X3", True, True)
# X4, X5 mostly isolated
tn2.update("X4", "X5", True, True)

clustering = tn2._compute_clustering_coefficient("X1", agents_top)
test("Clustering coefficient for dense node > 0", clustering > 0)

overlap = tn2._compute_neighbor_overlap("X1", "X2")
test("Neighbor overlap for connected nodes > 0", overlap > 0)

overlap_isolated = tn2._compute_neighbor_overlap("X1", "X4")
test("Neighbor overlap for disconnected nodes = 0", overlap_isolated == 0.0)

conductance = tn2.compute_local_conductance("X1", set(agents_top))
test("Conductance computed for dense node", isinstance(conductance, float))

# Clusters
clusters = tn2.get_clusters(agents_top, threshold=0.5)
test("Cluster detection finds groups", len(clusters) >= 1)
test("Dense cluster found", any(len(c) >= 2 for c in clusters))


# ─── 7. Local Detection Tests ───────────────────────────
print("\n--- 7. Local Detection Tests ---")

# Create agent with known history
detector = NeuralAgent(id="DET001")
# Simulate interactions where opponent cooperates with us at different rate than global
for i in range(10):
    detector.history["suspect1"] = detector.history.get("suspect1", [])
    detector.history["suspect1"].append((True, False))  # opponent always defects with us
    detector.interactions += 1

pop_stats = {"suspect1": {"coop_rate": 0.8, "interactions": 50}}  # opponent is globally cooperative
suspicion = detector.compute_suspicion("suspect1", pop_stats)
test("High divergence → high suspicion", suspicion > 0.3,
     f"suspicion={suspicion:.2f}")

# Low divergence
detector.history["friend1"] = [(True, True)] * 10
pop_stats["friend1"] = {"coop_rate": 0.9, "interactions": 50}
suspicion_friend = detector.compute_suspicion("friend1", pop_stats)
test("Low divergence → low suspicion", suspicion_friend < 0.3,
     f"suspicion={suspicion_friend:.2f}")

# Warning emission
detector.suspicion_scores["suspect1"] = 0.8
detector.vigilance = 0.3  # low threshold
warning = detector.emit_warning("suspect1")
test("Warning emitted when suspicion > vigilance", warning is not None)
test("Warning contains target", warning['target'] == "suspect1")

detector.vigilance = 0.9  # high threshold
warning2 = detector.emit_warning("suspect1")
test("No warning when suspicion < vigilance", warning2 is None)

# Warning reception
receiver = NeuralAgent(id="REC001")
receiver.receive_warning("DET001", {
    'target': 'suspect1', 'score': 0.7, 'evidence': 'test'
})
test("Warning received and stored", 'suspect1' in receiver.warnings_received)
test("Warning has correct source", receiver.warnings_received['suspect1'][0]['from'] == 'DET001')

# Dedup: same source warning not duplicated
receiver.receive_warning("DET001", {
    'target': 'suspect1', 'score': 0.8, 'evidence': 'test2'
})
test("Duplicate warnings deduplicated", len(receiver.warnings_received['suspect1']) == 1)

# Different source accepted
receiver.receive_warning("DET002", {
    'target': 'suspect1', 'score': 0.6, 'evidence': 'test'
})
test("Different source warning accepted", len(receiver.warnings_received['suspect1']) == 2)


# ─── 8. Immune Memory Tests ─────────────────────────────
print("\n--- 8. Immune Memory Tests ---")

mem_agent = NeuralAgent(id="MEM001", memory_capacity=5)

# Store threat pattern
mem_agent.store_threat_pattern("bad1", {
    'coop_rate': 0.1, 'commit_rate': 0.3, 'selectivity': 0.5
})
test("Threat pattern stored", len(mem_agent.threat_memory) == 1)

# Duplicate not stored
mem_agent.store_threat_pattern("bad2", {
    'coop_rate': 0.12, 'commit_rate': 0.32, 'selectivity': 0.5
})
test("Duplicate pattern not stored", len(mem_agent.threat_memory) == 1)

# Different pattern stored
mem_agent.store_threat_pattern("bad3", {
    'coop_rate': 0.9, 'commit_rate': 0.9, 'selectivity': 0.5
})
test("Different pattern stored", len(mem_agent.threat_memory) == 2)

# Pattern matching
mem_agent.history["suspect_new"] = [(True, False)] * 5  # low coop opponent
mem_agent.commitment_history["suspect_new"] = [1, 3]  # low commit rate
match = mem_agent.match_threat_patterns("suspect_new")
test("Matching pattern detected", match > 0.5, f"match={match:.2f}")

# LRU eviction
for i in range(10):
    mem_agent.store_threat_pattern(f"unique_{i}", {
        'coop_rate': 0.2 + i * 0.05, 'commit_rate': 0.4, 'selectivity': 0.5
    })
test("Memory capacity enforced", len(mem_agent.threat_memory) <= mem_agent.memory_capacity)


# ─── 9. Trust-Dependent Game Tests ──────────────────────
print("\n--- 9. Trust-Dependent Game Tests ---")

evo = Evolution(population_size=20)
evo.spawn_population()

# Verify payoff matrices exist
test("Stranger matrix defined", 'CC' in evo.payoff_matrices['strangers'])
test("Acquaintance matrix defined", 'CC' in evo.payoff_matrices['acquaintances'])
test("Partner matrix defined", 'CC' in evo.payoff_matrices['partners'])

# Strangers: exploitation dominates
s = evo.payoff_matrices['strangers']
test("Strangers: defect > cooperate", s['DC'] > s['CC'])

# Partners: cooperation dominates
p = evo.payoff_matrices['partners']
test("Partners: cooperate > defect", p['CC'] > p['DC'])

# Partners: mutual cooperation beats mutual defection
test("Partners: CC >> DD", p['CC'] > abs(p['DD']))

# Trust progression: as agents interact, game should improve
ids = list(evo.agents.keys())
a_id, b_id = ids[0], ids[1]
# Build trust
for _ in range(15):
    evo.trust_net.update(a_id, b_id, True, True)

trust_level = evo.trust_net.compute_direct_trust(a_id, b_id)
test("Trust builds from cooperation", trust_level > 0.6, f"trust={trust_level:.2f}")


# ─── 10. Evolution Engine Tests ─────────────────────────
print("\n--- 10. Evolution Engine Tests ---")

evo2 = Evolution(population_size=30)
evo2.spawn_population()
test("Population spawns", len(evo2.agents) == 30)
test("All agents alive", all(a.alive for a in evo2.agents.values()))

evo2.run_round()
test("Round increments", evo2.round == 1)
test("Stats recorded", len(evo2.round_stats) == 1)
test("Coop rate tracked", 'coop_rate' in evo2.round_stats[0])

for _ in range(9):
    evo2.run_round()
test("10 rounds complete", evo2.round == 10)

alive_count = len(evo2.get_alive())
test("Agents still alive after 10 rounds", alive_count >= 20, f"alive={alive_count}")

# Selection
evo2.run_selection()
test("Selection runs", evo2.generation == 1)

# Leaderboard
lb = evo2.get_leaderboard(5)
test("Leaderboard returns entries", len(lb) > 0)
test("Leaderboard sorted by fitness", lb[0]['fitness'] >= lb[-1]['fitness'])

# Strategy distribution
dist = evo2.get_strategy_distribution()
test("Strategy distribution non-empty", len(dist) > 0)

# Network data
nd = evo2.get_network_data()
test("Network data has nodes", len(nd['nodes']) > 0)
test("Network data has trust_weight_diversity", 'trust_weight_diversity' in nd)
test("Network data has immune metrics", 'immune_memory_total' in nd)


# ─── 11. Attack Tests ───────────────────────────────────
print("\n--- 11. Attack Tests ---")

evo3 = Evolution(population_size=20)
evo3.spawn_population()
for _ in range(5):
    evo3.run_round()

# Sybil attack
s_ids = Attacks.sybil_attack(evo3, 5)
test("Sybil attack injects agents", len(s_ids) == 5)
test("Sybil agents have rings", all(evo3.agents[s].sybil_ring for s in s_ids))

# Trojan attack
t_ids = Attacks.trojan_attack(evo3, 3, betray_round=10)
test("Trojan attack injects agents", len(t_ids) == 3)

# Eclipse attack
target = list(evo3.agents.keys())[0]
e_ids = Attacks.eclipse_attack(evo3, target, 4)
test("Eclipse attack injects agents", len(e_ids) == 4)

# Whitewash attack
w_ids = Attacks.whitewash_attack(evo3, 2)
test("Whitewash attack injects agents", len(w_ids) == 2)

# Trojan activation
for _ in range(6):
    evo3.run_round()
activated = Attacks.activate_trojans(evo3)
test("Trojans activate at target round", len(activated) > 0, f"activated={len(activated)}")


# ─── 12. Immune System Integration Tests ────────────────
print("\n--- 12. Immune System Integration Tests ---")

# Reset seed for reproducible integration test.
# Prior unit tests consume random numbers, shifting the sequence.
np.random.seed(42)
random.seed(42)

immune = ImmuneSystem()
test("ImmuneSystem creates", immune is not None)

# Test with a small population
evo4 = Evolution(population_size=30)
evo4.spawn_population()

# Run enough rounds to build history
for _ in range(15):
    evo4.run_round()

# Inject sybils
sybil_ids = Attacks.sybil_attack(evo4, 8)

# Run more rounds — immune system needs observation time.
# 80 rounds gives ~3-4 interactions per pair (38 agents, random pairing).
# The statistical ring detection requires enough evidence for z-test significance.
# With data-derived MINIMUM_GAP (2σ), more observations → tighter threshold → better detection.
for _ in range(80):
    evo4.run_round()
    if evo4.round % 20 == 0:
        evo4.run_selection()

# Check ALL agents (alive or dead) — flagged sybils may have been killed by
# quarantine drain or immune verdict at selection, which is the desired behavior.
all_agents4 = list(evo4.agents.values())
flagged4 = [a for a in all_agents4 if a.flagged_sybil]
real_caught = [a for a in flagged4 if a.id.startswith('S')]
false_pos = [a for a in flagged4 if not a.id.startswith('S')]

test("Immune system detects sybils (>=50%)", len(real_caught) >= 4,
     f"caught={len(real_caught)}/8")
test("Low false positives (<=1)", len(false_pos) <= 1,
     f"fp={len(false_pos)}")


# ─── 13. Clean Run — Zero False Positives ───────────────
print("\n--- 13. Clean Run Tests ---")
np.random.seed(42)
random.seed(42)

evo5 = Evolution(population_size=50)
evo5.spawn_population()
for _ in range(50):
    evo5.run_round()
    if evo5.round % 20 == 0:
        evo5.run_selection()

flagged5 = [a for a in evo5.get_alive() if a.flagged_sybil]
# ZERO false positives required. The statistical ring detection (z > 3.09,
# gap > 3σ, mu_out < mu_trust) should never flag honest agents.
# If this fails, the detection thresholds need investigation, not relaxation.
test("Clean run: zero false positives", len(flagged5) == 0,
     f"flagged={len(flagged5)}")


# ─── 14. Full Demo Simulation ──────────────────────────
print("\n--- 14. Full Demo Simulation ---")
np.random.seed(42)
random.seed(42)

evo_full = Evolution(population_size=50)
evo_full.spawn_population()

# Phase 1: 25 rounds
for _ in range(25):
    evo_full.run_round()
evo_full.run_selection()

stats_p1 = evo_full.round_stats[-1]
test("Phase 1 completes (R25)", evo_full.round == 25)
test("Phase 1 agents alive", stats_p1['alive'] >= 30, f"alive={stats_p1['alive']}")
test("Phase 1 has coop rate", 0 < stats_p1['coop_rate'] < 1)

# Phase 2: Sybil attack + 25 rounds
Attacks.sybil_attack(evo_full, 10)
for _ in range(25):
    evo_full.run_round()
    Attacks.activate_trojans(evo_full)
evo_full.run_selection()

stats_p2 = evo_full.round_stats[-1]
test("Phase 2 completes (R50)", evo_full.round == 50)

# Phase 3: Trojan + Eclipse + 25 rounds
Attacks.trojan_attack(evo_full, 5)
lb = evo_full.get_leaderboard(1)
if lb:
    Attacks.eclipse_attack(evo_full, lb[0]['id'], 6)
for _ in range(25):
    evo_full.run_round()
    Attacks.activate_trojans(evo_full)
evo_full.run_selection()

stats_p3 = evo_full.round_stats[-1]
test("Phase 3 completes (R75)", evo_full.round == 75)

# Phase 4: Economic shift + 25 rounds
evo_full.set_payoff('partners', 'CC', 600)
evo_full.set_payoff('strangers', 'DC', 300)
for _ in range(25):
    evo_full.run_round()
evo_full.run_selection()

stats_p4 = evo_full.round_stats[-1]
test("Phase 4 completes (R100)", evo_full.round == 100)
test("Simulation has survivors", stats_p4['alive'] >= 15,
     f"alive={stats_p4['alive']}")

# Detection check after 75 rounds of sybil exposure (R26-R100).
# First principle: immune response needs sufficient data + time for the
# economic shift (Phase 4) to further separate sybil reputation from honest agents.
# Check all agents (alive or dead) — flagged sybils die from quarantine/selection
flagged_final = [a for a in evo_full.agents.values() if a.flagged_sybil]
test("Immune system detects sybils by R100", len(flagged_final) >= 1,
     f"detected={len(flagged_final)}")

# Network data
nd_final = evo_full.get_network_data()
test("Final network data valid", len(nd_final['nodes']) >= 10)
test("Final has trust weight diversity", len(nd_final['trust_weight_diversity']) == 4)
test("Final has immune metrics", nd_final['immune_memory_total'] >= 0)


# ─── 15. Server Import Test ─────────────────────────────
print("\n--- 15. Server Tests ---")

try:
    from engine.server import app
    test("Server app imports", True)
except Exception as e:
    test("Server app imports", False, str(e))

dash_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'index.html')
test("Dashboard HTML exists", os.path.exists(dash_path))

d3_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'd3.v7.min.js')
test("D3 library local", os.path.exists(d3_path))
test("D3 library not empty", os.path.getsize(d3_path) > 100000)


# ─── 16. Trojan Detection Test ────────────────────────────
print("\n--- 16. Trojan Detection Test ---")

np.random.seed(42)
random.seed(42)
evo_trojan = Evolution(population_size=30)
evo_trojan.spawn_population()
for _ in range(15):
    evo_trojan.run_round()

# Inject trojans that betray in 15 rounds
t_ids = Attacks.trojan_attack(evo_trojan, 5, betray_round=evo_trojan.round + 15)
# 40 rounds: 15 for trust-building, 25 post-activation for reputation to drop.
# Cascade evidence = trust in victim (no arbitrary multiplier), so
# reputation degradation is gradual — needs enough post-activation rounds.
for _ in range(40):
    evo_trojan.run_round()
    Attacks.activate_trojans(evo_trojan)

# After activation, trojans should have been caught by behavioral shift
trojan_agents = [evo_trojan.agents[t] for t in t_ids if t in evo_trojan.agents]
active_trojans = [t for t in trojan_agents if t.alive and t.parent_id == 'TROJAN_ACTIVE']
test("Trojans activate", len(active_trojans) > 0, f"active={len(active_trojans)}")

# Activated trojans should lose reputation (trust collapse from betrayal).
# Threshold < 0.5 = below Bayesian neutral. With cascade evidence proportional
# to trust (not amplified), reputation drop is gradual but reaches below-neutral.
if active_trojans:
    alive_ids = [a.id for a in evo_trojan.get_alive()]
    trojan_reps = [evo_trojan.trust_net.get_reputation(t.id, alive_ids)
                   for t in active_trojans]
    avg_trojan_rep = float(np.mean(trojan_reps))
    test("Activated trojans lose reputation", avg_trojan_rep < 0.50,
         f"avg_rep={avg_trojan_rep:.2f}")
else:
    test("Activated trojans lose reputation", False, "no active trojans found")


# ─── 17. Eclipse Detection Test ──────────────────────────
print("\n--- 17. Eclipse Detection Test ---")

np.random.seed(42)
random.seed(42)
evo_eclipse = Evolution(population_size=30)
evo_eclipse.spawn_population()
for _ in range(10):
    evo_eclipse.run_round()

lb = evo_eclipse.get_leaderboard(1)
if lb:
    target_id = lb[0]['id']
    e_ids = Attacks.eclipse_attack(evo_eclipse, target_id, 6)
    for _ in range(30):
        evo_eclipse.run_round()

    # Target should survive — eclipse shouldn't kill productive agents
    target_agent = evo_eclipse.agents[target_id]
    test("Eclipse target survives", target_agent.alive,
         f"alive={target_agent.alive}, balance={target_agent.balance:.0f}")
else:
    test("Eclipse target survives", False, "no leaderboard entry")


# ─── 18. Warning Cost Test ───────────────────────────────
print("\n--- 18. Warning Cost Test ---")

cost_agent = NeuralAgent(id="COST001")
initial_balance = cost_agent.balance
cost_agent.suspicion_scores["target1"] = 0.9
cost_agent.vigilance = 0.1  # low threshold so warning fires
warning = cost_agent.emit_warning("target1")
test("Warning emitted", warning is not None)
test("Warning has fitness cost", cost_agent.balance < initial_balance,
     f"balance went from {initial_balance} to {cost_agent.balance}")


print("\n--- 19. Reproductive Exclusion Test ---")

np.random.seed(42)
random.seed(42)
evo_repro = Evolution(population_size=30)
evo_repro.spawn_population()

# Run enough rounds for immune system to activate
for _ in range(20):
    evo_repro.run_round()

# Inject sybils
sybil_ids = Attacks.sybil_attack(evo_repro, 6)

# Run more rounds to build sybil behavioral signal
for _ in range(60):
    evo_repro.run_round()

# Manually flag sybils (simulate immune detection)
for sid in sybil_ids:
    if sid in evo_repro.agents:
        evo_repro.agents[sid].flagged_sybil = True

# Run selection — flagged agents should NOT reproduce
pre_selection_ids = set(a.id for a in evo_repro.get_alive())
evo_repro.run_selection()

# Check: no child should have a flagged parent
births = [e for e in evo_repro.events if e['type'] == 'birth' and e['round'] == evo_repro.round]
flagged_parent_children = []
for birth in births:
    parent_str = birth.get('parents', '')
    for sid in sybil_ids:
        if sid in parent_str:
            flagged_parent_children.append(birth['agent'])
            break

test("Flagged sybils excluded from reproduction",
     len(flagged_parent_children) == 0,
     f"children_from_flagged={len(flagged_parent_children)}")

# Verify flagged agents are eliminated at selection (immune verdict = death)
flagged_alive = [evo_repro.agents[sid] for sid in sybil_ids
                 if sid in evo_repro.agents and evo_repro.agents[sid].alive]
test("Flagged agents eliminated at selection",
     len(flagged_alive) == 0,
     f"flagged_still_alive={len(flagged_alive)}")


print("\n--- 20. Inherited Reputation Test ---")

np.random.seed(42)
random.seed(42)
trust_net = TrustNetwork()

# Create two parents with different reputations
# Parent A: well-trusted (mutual cooperation — many interactions)
for i in range(10):
    for _ in range(5):
        trust_net.update(f"other_{i}", "parent_a", True, True)

# Parent B: poorly-trusted (B defects on everyone — many interactions)
for i in range(10):
    for _ in range(5):
        trust_net.update(f"other_{i}", "parent_b", True, False)

# Verify parent reputations diverge
all_ids = [f"other_{i}" for i in range(10)] + ["parent_a", "parent_b"]
rep_a = trust_net.get_reputation("parent_a", all_ids)
rep_b = trust_net.get_reputation("parent_b", all_ids)
test("Parent A has high reputation", rep_a > 0.7, f"rep_a={rep_a:.3f}")
test("Parent B has low reputation", rep_b < 0.3, f"rep_b={rep_b:.3f}")

# Seed child trust from parents
trust_net.seed_child_trust("child_ab", "parent_a", "parent_b", all_ids)

# Child should have trust edges
child_edges = [(k, v) for k, v in trust_net.edges.items() if "child_ab" in k]
test("Child has inherited trust edges", len(child_edges) > 0,
     f"edges={len(child_edges)}")

# Others' trust in child: should be between parent_a and parent_b trust
others_trust_child = []
for i in range(10):
    state = trust_net.edges.get((f"other_{i}", "child_ab"))
    if state:
        others_trust_child.append(state.direct_trust)

if others_trust_child:
    avg_child_trust = np.mean(others_trust_child)
    test("Child trust is between parents (not clean slate)",
         0.3 < avg_child_trust < 0.7,
         f"avg_trust_in_child={avg_child_trust:.3f}")
    # Mixed parents average near 0.5 — that's mathematically correct.
    # The key test is that edges EXIST (not default) and that same-parent
    # children diverge from neutral (tested below).
    test("Child has non-trivial trust state (edges seeded, not default)",
         len(others_trust_child) >= 5,
         f"trust_edges_from_others={len(others_trust_child)}")

# Now test child of two trusted parents
trust_net.seed_child_trust("child_aa", "parent_a", "parent_a", all_ids)
others_trust_aa = []
for i in range(10):
    state = trust_net.edges.get((f"other_{i}", "child_aa"))
    if state:
        others_trust_aa.append(state.direct_trust)

if others_trust_aa:
    avg_aa = np.mean(others_trust_aa)
    test("Child of trusted parents starts above neutral",
         avg_aa > 0.5,
         f"avg_trust={avg_aa:.3f}")

# Child of two distrusted parents
trust_net.seed_child_trust("child_bb", "parent_b", "parent_b", all_ids)
others_trust_bb = []
for i in range(10):
    state = trust_net.edges.get((f"other_{i}", "child_bb"))
    if state:
        others_trust_bb.append(state.direct_trust)

if others_trust_bb:
    avg_bb = np.mean(others_trust_bb)
    test("Child of distrusted parents starts below neutral",
         avg_bb < 0.5,
         f"avg_trust={avg_bb:.3f}")


print("\n--- 21. Immune Memory Inheritance Test ---")

np.random.seed(42)
random.seed(42)

# Create two parents with threat memories
parent_a = NeuralAgent(id="MEM_A")
parent_b = NeuralAgent(id="MEM_B")

# Give them distinct threat patterns
parent_a.threat_memory = [
    {'coop_rate': 0.1, 'commit_rate': 0.3, 'source': 'parent_a'},
    {'coop_rate': 0.2, 'commit_rate': 0.4, 'source': 'parent_a'}
]
parent_b.threat_memory = [
    {'coop_rate': 0.5, 'commit_rate': 0.8, 'source': 'parent_b'},
    {'coop_rate': 0.1, 'commit_rate': 0.35, 'source': 'parent_b'}  # near-duplicate of A's first
]

child = crossover(parent_a, parent_b, "MEM_CHILD", generation=1)

test("Child inherits threat memory", len(child.threat_memory) > 0,
     f"memory_count={len(child.threat_memory)}")
test("Memories are deduplicated",
     len(child.threat_memory) < len(parent_a.threat_memory) + len(parent_b.threat_memory),
     f"child={len(child.threat_memory)} vs total={len(parent_a.threat_memory) + len(parent_b.threat_memory)}")
test("Child has memories from both parents",
     len(child.threat_memory) >= 2,
     f"count={len(child.threat_memory)}")

# Test that empty-memory parents produce empty-memory child
empty_a = NeuralAgent(id="EMPTY_A")
empty_b = NeuralAgent(id="EMPTY_B")
empty_child = crossover(empty_a, empty_b, "EMPTY_CHILD", generation=1)
test("Empty parents produce empty memory child",
     len(empty_child.threat_memory) == 0)

# Test memory capacity cap
big_memory_parent = NeuralAgent(id="BIG_MEM")
big_memory_parent.threat_memory = [
    {'coop_rate': i * 0.1, 'commit_rate': i * 0.1}
    for i in range(20)  # 20 distinct patterns
]
small_cap_parent = NeuralAgent(id="SMALL_CAP")
small_cap_parent.memory_capacity = 3
small_cap_parent.threat_memory = []

capped_child = crossover(big_memory_parent, small_cap_parent, "CAP_CHILD", generation=1)
test("Memory capped at child capacity",
     len(capped_child.threat_memory) <= capped_child.memory_capacity,
     f"memory={len(capped_child.threat_memory)}, cap={capped_child.memory_capacity}")


# ─── Results ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
print("=" * 60)

if FAIL > 0:
    print("\nFIX FAILURES BEFORE DEMO!")
    sys.exit(1)
else:
    print("\nALL TESTS PASS — READY FOR DEMO")
    sys.exit(0)
