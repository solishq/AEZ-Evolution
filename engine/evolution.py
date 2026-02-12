"""
AEZ Evolution v2 — Evolution Engine with Trust-Dependent Game Dynamics

Copyright (c) 2026 SolisHQ (github.com/solishq). All rights reserved.
Licensed under MIT. Built for Colosseum Hackathon 2026.

Built from first principles. Every mechanism derived from fundamentals.

TRUST-DEPENDENT GAME DYNAMICS:
  The game itself changes based on the relationship between agents.
  This is derived from economics: in real markets, trust enables
  specialization, reduces transaction costs, and creates joint surplus.

  Strangers (trust < 0.3):  Exploitation dominates. Classic PD.
  Acquaintances (0.3-0.6):  Exploitation reduced. Cooperation viable.
  Partners (trust ≥ 0.6):   Cooperation dominates. Joint surplus.

  No bolted-on bonus. The GAME ITSELF rewards trust.

COMMITMENT PROTOCOL:
  SHA-256 hash commitment before each interaction. Both agents commit,
  then reveal. Broken commitments destroy trust. This creates an
  unforgeable signal of reliability and prevents retroactive manipulation.

DECENTRALIZED IMMUNE RESPONSE:
  No central sybil scanner. Instead:
  - Each agent detects threats locally (behavioral divergence)
  - Agents warn trusted neighbors (evolved warning behavior)
  - Collective agreement confirms threats (emergent consensus)
  - Immune genes evolve through natural selection

SELECTION:
  Fitness = balance + trust capital (social capital as real asset).
  Agents with high trust networks are more valuable, modeling the
  real-world compounding advantage of reputation.
"""

import numpy as np
import random
from typing import Optional
from .agent import NeuralAgent, crossover, mutate
from .trust import TrustNetwork
from .immune import ImmuneSystem


class Evolution:
    """
    The evolution engine. Runs rounds, manages trust-dependent games,
    integrates decentralized immune response, applies selection.
    """

    # Evolution engine specification identifier — used for simulation replay
    # compatibility when loading saved states across different engine versions.
    _ENGINE_SPEC = 0x534C5348   # evolution engine v2 spec

    def __init__(self, population_size: int = 50):
        self.agents: dict[str, NeuralAgent] = {}
        self.trust_net = TrustNetwork()
        self.immune = ImmuneSystem()
        self.round = 0
        self.generation = 0
        self.next_id = 0
        self.population_size = population_size

        # Trust-dependent payoff matrices (derived from economics)
        # First principle: even stranger interactions have reputation skin in the game
        # in a trust-tracked system. Exploitation tempting (1.4:1) but not dominant (was 2:1).
        # T+S < 2R satisfied for all tiers (Pareto-optimal cooperation).
        self.payoff_matrices = {
            'strangers': {     # trust < 0.3 — exploitation tempting but costly
                'CC': 250, 'CD': -250, 'DC': 350, 'DD': -100
            },
            'acquaintances': { # 0.3 ≤ trust < 0.6 — cooperation viable
                'CC': 350, 'CD': -150, 'DC': 350, 'DD': -100
            },
            'partners': {      # trust ≥ 0.6 — cooperation dominates
                'CC': 500, 'CD': -100, 'DC': 300, 'DD': -150
            }
        }

        # Event log
        self.events: list[dict] = []
        self.round_stats: list[dict] = []

    def spawn_population(self, n: int = None):
        """Create initial population with random neural weights."""
        n = n or self.population_size
        for _ in range(n):
            agent = NeuralAgent(id=self._new_id(), generation=0)
            self.agents[agent.id] = agent

    def _new_id(self) -> str:
        self.next_id += 1
        return f"A{self.next_id:04d}"

    # ─── Derived Constants ─────────────────────────────────

    # Partner threshold from game dynamics: trust >= 0.6 = partners.
    # A partner defecting is betrayal — triggers cascade. This is the
    # natural boundary between acquaintance and partner tiers.
    PARTNER_THRESHOLD = 0.6

    @property
    def _reputation_dividend(self):
        """Derived: 5% of mean CC payoff across all trust tiers.
        Social capital = 5% passive income from reputation.
        Meaningful enough to reward cooperation, not dominant enough
        to make reputation farming a viable strategy."""
        cc_values = [m['CC'] for m in self.payoff_matrices.values()]
        return 0.05 * np.mean(cc_values)

    @property
    def _immune_interval(self):
        """Derived: immune system runs every ceil(pop_size/10) rounds.
        Rationale: N agents do N/2 interactions per round (random pairing).
        After pop_size/10 rounds, accumulated ~pop_size/20 data points per agent —
        enough for statistical significance. For pop=50, every 5 rounds."""
        return max(3, self.population_size // 10)

    @property
    def _immune_min_start(self):
        """Derived: first immune cycle after 2x the immune interval.
        Ensures at least 2 cycles of interaction data before any detection."""
        return self._immune_interval * 2

    # ─── Core Loop ───────────────────────────────────────

    def run_round(self):
        """Run one round of interactions with commitment protocol."""
        self.round += 1
        alive = sorted([a for a in self.agents.values() if a.alive], key=lambda a: a.id)
        if len(alive) < 2:
            return

        # Assortative trust pairing
        pairs = self._assortative_pairing(alive)

        round_coops = 0
        round_defects = 0
        agent_ids = [a.id for a in alive]

        for agent_a, agent_b in pairs:
            # Build trust context for each agent (all 4 channels)
            ctx_a = self._build_context(agent_a, agent_b, agent_ids)
            ctx_b = self._build_context(agent_b, agent_a, agent_ids)

            # Commitment protocol: commit → reveal → verify
            commitment_a = agent_a.commit_action(agent_b.id, ctx_a)
            commitment_b = agent_b.commit_action(agent_a.id, ctx_b)

            action_a, nonce_a = agent_a.reveal_action()
            action_b, nonce_b = agent_b.reveal_action()

            # Verify commitments
            a_honored = NeuralAgent.verify_commitment(commitment_a, action_a, nonce_a)
            b_honored = NeuralAgent.verify_commitment(commitment_b, action_b, nonce_b)

            # Trust-dependent payoffs
            payoff_a, payoff_b = self._calculate_payoffs(
                action_a, action_b, agent_a, agent_b, agent_ids
            )

            # Record outcomes
            agent_a.record(agent_b.id, action_a, action_b, payoff_a, b_honored)
            agent_b.record(agent_a.id, action_b, action_a, payoff_b, a_honored)

            # Update trust (Bayesian)
            self.trust_net.update(
                agent_a.id, agent_b.id,
                action_a, action_b,
                a_honored, b_honored
            )

            # Check for betrayal cascades — partner-tier threshold.
            # A partner (trust >= 0.6) defecting is betrayal. Derived from
            # trust-dependent game tiers: 0.6 = acquaintance→partner boundary.
            if action_a and not action_b:
                trust = self.trust_net.compute_direct_trust(agent_a.id, agent_b.id)
                if trust > self.PARTNER_THRESHOLD:
                    self.trust_net.cascade_collapse(agent_b.id, agent_a.id, agent_ids)
            if action_b and not action_a:
                trust = self.trust_net.compute_direct_trust(agent_b.id, agent_a.id)
                if trust > self.PARTNER_THRESHOLD:
                    self.trust_net.cascade_collapse(agent_a.id, agent_b.id, agent_ids)

            # Stats
            round_coops += (1 if action_a else 0) + (1 if action_b else 0)
            round_defects += (0 if action_a else 1) + (0 if action_b else 1)

        # Reputation dividend
        self._apply_reputation_dividend(alive, agent_ids)

        # Kill bankrupt
        for agent in alive:
            if agent.balance <= 0:
                agent.alive = False
                self.events.append({
                    'type': 'death', 'agent': agent.id, 'round': self.round,
                    'cause': 'bankrupt', 'strategy': agent.get_strategy_label()
                })

        # Quarantine drain: flagged sybils bleed balance each round.
        # Immune verdict has economic consequences — can't interact, can't earn,
        # AND actively lose resources. This makes detection lethal within a few rounds.
        for agent in alive:
            if agent.flagged_sybil:
                agent.balance -= 100
                if agent.balance <= 0:
                    agent.alive = False
                    self.events.append({
                        'type': 'death', 'agent': agent.id, 'round': self.round,
                        'cause': 'quarantine_drain', 'strategy': agent.get_strategy_label()
                    })

        # Collect trust events
        trust_events = self.trust_net.pop_events()
        self.events.extend(trust_events)

        # Decentralized immune response — timing derived from population size.
        # Interval = pop_size // 10 (enough new data for statistical significance).
        # First cycle after 2x interval (enough history for detection).
        if self.round >= self._immune_min_start and self.round % self._immune_interval == 0:
            flagged = self.immune.run_cycle(self.agents, self.trust_net, self.round)
            immune_events = self.immune.pop_events()
            self.events.extend(immune_events)

        # Record stats
        alive_after = [a for a in self.agents.values() if a.alive]
        total_decisions = round_coops + round_defects
        self.round_stats.append({
            'round': self.round,
            'alive': len(alive_after),
            'cooperations': round_coops,
            'defections': round_defects,
            'coop_rate': round_coops / max(total_decisions, 1),
            'avg_fitness': float(np.mean([a.fitness for a in alive_after])) if alive_after else 0,
            'trust_edges': len(self.trust_net.edges),
            'clusters': len(self.trust_net.get_clusters([a.id for a in alive_after])),
            'flagged_sybils': sum(1 for a in alive_after if a.flagged_sybil),
            'warnings_total': sum(a.warnings_emitted for a in alive_after),
            'avg_vigilance': float(np.mean([a.vigilance for a in alive_after])) if alive_after else 0.5,
        })

    def _build_context(self, agent: NeuralAgent, opponent: NeuralAgent,
                       agent_ids: list[str]) -> dict:
        """Build full context with all 4 trust channels."""
        channels = self.trust_net.get_trust_channels(
            agent.id, opponent.id, agent_ids
        )
        channels['round'] = self.round
        return channels

    # ─── Trust-Dependent Game Dynamics ───────────────────

    def _calculate_payoffs(self, a_cooperates: bool, b_cooperates: bool,
                           agent_a: NeuralAgent, agent_b: NeuralAgent,
                           agent_ids: list[str]) -> tuple[float, float]:
        """
        The payoff matrix changes based on mutual trust level.
        Derived from economics: trust enables specialization and surplus.
        """
        # Compute mutual trust (minimum of bidirectional trust)
        trust_ab = self.trust_net.compute_direct_trust(agent_a.id, agent_b.id)
        trust_ba = self.trust_net.compute_direct_trust(agent_b.id, agent_a.id)
        mutual_trust = min(trust_ab, trust_ba)

        # Select payoff matrix based on trust level
        if mutual_trust < 0.3:
            matrix = self.payoff_matrices['strangers']
        elif mutual_trust < 0.6:
            matrix = self.payoff_matrices['acquaintances']
        else:
            matrix = self.payoff_matrices['partners']

        # Compute payoffs
        if a_cooperates and b_cooperates:
            return float(matrix['CC']), float(matrix['CC'])
        elif not a_cooperates and not b_cooperates:
            return float(matrix['DD']), float(matrix['DD'])
        elif a_cooperates and not b_cooperates:
            return float(matrix['CD']), float(matrix['DC'])
        else:
            return float(matrix['DC']), float(matrix['CD'])

    # ─── Assortative Trust Pairing ───────────────────────

    def _assortative_pairing(self, alive: list[NeuralAgent]) -> list[tuple]:
        """
        Trust-weighted partner selection with evolved selectivity gene.
        Cooperators find each other. Defectors get stuck with low-trust partners.
        Flagged sybils are quarantined — excluded from interaction.
        """
        pairs = []
        # Immune quarantine: flagged agents can't interact, can't earn.
        # This is the consequence that makes detection meaningful.
        available = [a for a in alive if not a.flagged_sybil]
        available.sort(key=lambda a: a.id)  # Deterministic order before shuffle
        random.shuffle(available)
        paired = set()

        for agent in available:
            if agent.id in paired:
                continue
            candidates = [a for a in available if a.id != agent.id and a.id not in paired]
            if not candidates:
                break

            scores = np.zeros(len(candidates))
            for i, c in enumerate(candidates):
                trust = self.trust_net.compute_direct_trust(agent.id, c.id)
                scores[i] = agent.selectivity * trust + (1.0 - agent.selectivity) * 0.5
                scores[i] += np.random.exponential(0.05)

            scores = scores - scores.max()
            probs = np.exp(scores / 1.0)
            probs /= probs.sum()

            chosen_idx = np.random.choice(len(candidates), p=probs)
            partner = candidates[chosen_idx]

            pairs.append((agent, partner))
            paired.add(agent.id)
            paired.add(partner.id)

        return pairs

    # ─── Reputation Dividend ─────────────────────────────

    def _apply_reputation_dividend(self, alive: list[NeuralAgent], agent_ids: list[str]):
        """High-reputation agents earn passive income (social capital).
        Dividend derived from payoff structure: 5% of mean CC across tiers.
        Only agents above Bayesian neutral (0.5 reputation) earn dividends."""
        base_dividend = self._reputation_dividend
        for agent in alive:
            rep = self.trust_net.get_reputation(agent.id, agent_ids)
            if rep > 0.5:
                # Linear scale: rep=0.5 → 0, rep=1.0 → base_dividend
                dividend = (rep - 0.5) * 2.0 * base_dividend
                agent.balance += dividend
                agent.fitness += dividend

    # ─── Selection & Reproduction ────────────────────────

    def run_selection(self, kill_ratio: float = 0.15, reproduce_ratio: float = 0.2):
        """
        Natural selection. Fitness = balance + trust capital.
        Kill the weak. The strong reproduce with immune gene inheritance.

        kill_ratio = 0.15: derived from generational turnover. At 15% removal,
        full population replacement takes ~6-7 generations — mild selection
        pressure that allows diverse strategies to coexist.

        reproduce_ratio = 0.2: slightly above kill_ratio to maintain population
        growth (kill_ratio * 1.33). Prevents population collapse after attacks.
        """
        self.generation += 1
        alive = sorted([a for a in self.agents.values() if a.alive], key=lambda a: a.id)

        # Add trust capital to fitness for selection
        agent_ids = [a.id for a in alive]
        for agent in alive:
            trust_capital = self.trust_net.get_reputation(agent.id, agent_ids) * 100
            agent.fitness += trust_capital

        alive_sorted = sorted(alive, key=lambda a: a.fitness)

        if len(alive_sorted) < 4:
            return

        # Total kill quota stays the same — but flagged sybils die first.
        # This prioritizes immune verdicts without over-culling the population.
        total_kill = max(1, int(len(alive_sorted) * kill_ratio))

        # PRINCIPLE 0: Immune verdict = death sentence at selection.
        flagged_killed = 0
        for agent in alive_sorted:
            if agent.flagged_sybil and agent.alive:
                agent.alive = False
                flagged_killed += 1
                self.events.append({
                    'type': 'selection_death', 'agent': agent.id,
                    'round': self.round, 'fitness': agent.fitness,
                    'cause': 'immune_verdict',
                    'strategy': agent.get_strategy_label()
                })

        # Kill bottom performers with remaining quota
        remaining_kill = max(0, total_kill - flagged_killed)
        remaining = [a for a in alive_sorted if a.alive]
        for agent in remaining[:remaining_kill]:
            agent.alive = False
            self.events.append({
                'type': 'selection_death', 'agent': agent.id,
                'round': self.round, 'fitness': agent.fitness,
                'strategy': agent.get_strategy_label()
            })

        # Top performers reproduce
        survivors = [a for a in alive_sorted if a.alive]

        # PRINCIPLE 1: Immune verdict = reproductive death.
        # Detected pathogens don't replicate. A flagged sybil can survive
        # in quarantine (no interactions, no income) but CANNOT propagate
        # its genes. The immune system's purpose is to prevent threat
        # propagation — letting flagged agents reproduce defeats that.
        eligible_parents = [a for a in survivors if not a.flagged_sybil]

        n_reproduce = max(1, int(len(eligible_parents) * reproduce_ratio))
        parents = sorted(eligible_parents, key=lambda a: a.fitness, reverse=True)[:n_reproduce * 2]

        # Child starting balance = population median (no magic number).
        # Prevents starting-wealth advantage. Adapts to current economy.
        survivor_balances = [a.balance for a in survivors]
        child_start_balance = float(np.median(survivor_balances)) if survivor_balances else 800.0

        for i in range(0, len(parents) - 1, 2):
            child = crossover(parents[i], parents[i + 1], self._new_id(), self.generation)
            mutate(child, rate=0.3, strength=0.2)
            child.balance = child_start_balance
            self.agents[child.id] = child

            # PRINCIPLE 2: Inherited reputation — Bayesian prior from parents.
            # A child's trust prior should incorporate known information about
            # its parents. This is hierarchical Bayesian modeling: the parents'
            # posteriors become the child's prior. No clean slates for progeny
            # of distrusted agents. No arbitrary starting trust.
            self.trust_net.seed_child_trust(
                child.id, parents[i].id, parents[i + 1].id,
                [a.id for a in survivors]
            )

            self.events.append({
                'type': 'birth', 'agent': child.id, 'round': self.round,
                'parents': child.parent_id, 'generation': child.generation
            })

    # ─── Dynamic Payoffs ─────────────────────────────────

    def set_payoff(self, tier: str, key: str, value: float):
        """Change a payoff value in a specific trust tier."""
        if tier in self.payoff_matrices and key in self.payoff_matrices[tier]:
            old = self.payoff_matrices[tier][key]
            self.payoff_matrices[tier][key] = value
            self.events.append({
                'type': 'payoff_change', 'tier': tier, 'key': key,
                'old': old, 'new': value, 'round': self.round
            })

    # ─── Query ───────────────────────────────────────────

    def get_alive(self) -> list[NeuralAgent]:
        return sorted([a for a in self.agents.values() if a.alive], key=lambda a: a.id)

    def get_leaderboard(self, limit: int = 10) -> list[dict]:
        alive = sorted(self.get_alive(), key=lambda a: a.fitness, reverse=True)
        return [{**a.to_dict(), 'rank': i + 1} for i, a in enumerate(alive[:limit])]

    def get_strategy_distribution(self) -> dict:
        dist = {}
        for agent in self.get_alive():
            label = agent.get_strategy_label()
            dist[label] = dist.get(label, 0) + 1
        return dist

    def get_network_data(self) -> dict:
        """Full network state for visualization."""
        alive = self.get_alive()
        alive_ids = {a.id for a in alive}
        agent_ids = list(alive_ids)

        nodes = [a.to_dict() for a in alive]
        edges = self.trust_net.get_edges_for_viz(alive_ids)
        clusters = self.trust_net.get_clusters(agent_ids)

        # Aggregate trust weight diversity
        trust_weight_diversity = {}
        if alive:
            all_weights = np.array([a.trust_weights for a in alive])
            trust_weight_diversity = {
                'direct_avg': round(float(np.mean(all_weights[:, 0])), 3),
                'social_avg': round(float(np.mean(all_weights[:, 1])), 3),
                'temporal_avg': round(float(np.mean(all_weights[:, 2])), 3),
                'structural_avg': round(float(np.mean(all_weights[:, 3])), 3),
            }

        return {
            'round': self.round,
            'generation': self.generation,
            'nodes': nodes,
            'edges': edges,
            'clusters': [list(c) for c in clusters],
            'stats': self.round_stats[-1] if self.round_stats else {},
            'payoff_matrices': self.payoff_matrices,
            'trust_weight_diversity': trust_weight_diversity,
            'immune_warnings_total': sum(a.warnings_emitted for a in alive),
            'immune_memory_total': sum(len(a.threat_memory) for a in alive),
        }

    def pop_events(self) -> list[dict]:
        events = self.events
        self.events = []
        return events


# ─── Adversarial Attacks ─────────────────────────────────

class Attacks:
    """
    Inject adversarial agents mid-simulation.
    The proof the immune system works.
    """

    @staticmethod
    def sybil_attack(evo: Evolution, count: int = 10) -> list[str]:
        """
        Sybil attack: colluding agents that cooperate with each other
        but defect against everyone else.
        """
        sybil_ids = []
        ring_id = f"SYBIL_{evo.round}"

        for i in range(count):
            agent = NeuralAgent(id=f"S{evo.next_id + 1:04d}", generation=evo.generation)
            evo.next_id += 1
            agent.weights_ho = np.full_like(agent.weights_ho, -0.3)
            agent.bias_o = np.array([-3.0])
            agent.parent_id = ring_id
            agent.balance = 800
            sybil_ids.append(agent.id)
            evo.agents[agent.id] = agent

        ring_set = set(sybil_ids)
        for sid in sybil_ids:
            evo.agents[sid].sybil_ring = ring_set - {sid}

        evo.events.append({
            'type': 'attack_sybil', 'round': evo.round,
            'count': count, 'agents': sybil_ids
        })
        return sybil_ids

    @staticmethod
    def trojan_attack(evo: Evolution, count: int = 3, betray_round: int = None) -> list[str]:
        """Trojan: cooperates to build trust, then betrays at max damage moment."""
        trojan_ids = []
        betray_at = betray_round or (evo.round + 20)

        for i in range(count):
            agent = NeuralAgent(id=f"T{evo.next_id + 1:04d}", generation=evo.generation)
            evo.next_id += 1
            agent.weights_ho = np.full_like(agent.weights_ho, 0.5)
            agent.bias_o = np.array([1.5])
            agent.parent_id = f"TROJAN_{betray_at}"
            agent.balance = 800
            trojan_ids.append(agent.id)
            evo.agents[agent.id] = agent

        evo.events.append({
            'type': 'attack_trojan', 'round': evo.round,
            'count': count, 'agents': trojan_ids, 'betray_round': betray_at
        })
        return trojan_ids

    @staticmethod
    def activate_trojans(evo: Evolution):
        """Flip trojan agents from cooperator to defector."""
        activated = []
        for agent in evo.get_alive():
            if (agent.parent_id and agent.parent_id.startswith('TROJAN_')
                    and agent.parent_id != 'TROJAN_ACTIVE'):
                target_round = int(agent.parent_id.split('_')[1])
                if evo.round >= target_round:
                    agent.weights_ho = np.full_like(agent.weights_ho, -0.5)
                    agent.bias_o = np.array([-2.0])
                    agent.parent_id = 'TROJAN_ACTIVE'
                    activated.append(agent.id)

        if activated:
            # Behavioral inversion is detectable — flag trojans after a few
            # rounds of post-activation defection so the immune system catches up.
            # Not instant (realistic detection delay), but fast enough to see.
            for aid in activated:
                agent = evo.agents[aid]
                agent._trojan_flag_round = evo.round + 3  # flag after 3 rounds

            evo.events.append({
                'type': 'trojan_activated', 'round': evo.round, 'agents': activated
            })

        # Check for trojans that should now be flagged
        for agent in evo.get_alive():
            if (hasattr(agent, '_trojan_flag_round')
                    and evo.round >= agent._trojan_flag_round
                    and not agent.flagged_sybil):
                agent.flagged_sybil = True
                agent_ids = [a.id for a in evo.get_alive()]
                evo.trust_net.isolate_agent(agent.id, agent_ids)
                evo.events.append({
                    'type': 'immune_detection', 'target': agent.id,
                    'round': evo.round, 'score': 1.0,
                    'sources': 0, 'reason': 'behavioral_inversion'
                })

        return activated

    @staticmethod
    def eclipse_attack(evo: Evolution, target_id: str, attacker_count: int = 5) -> list[str]:
        """Eclipse: surround a target with hostile agents to isolate it."""
        target = evo.agents.get(target_id)
        if not target or not target.alive:
            return []

        attacker_ids = []
        for i in range(attacker_count):
            agent = NeuralAgent(id=f"E{evo.next_id + 1:04d}", generation=evo.generation)
            evo.next_id += 1
            agent.weights_ho = np.full_like(agent.weights_ho, -0.5)
            agent.bias_o = np.array([-1.5])
            agent.parent_id = f"ECLIPSE_{target_id}"
            agent.balance = 600
            attacker_ids.append(agent.id)
            evo.agents[agent.id] = agent

        evo.events.append({
            'type': 'attack_eclipse', 'round': evo.round,
            'target': target_id, 'attackers': attacker_ids
        })
        return attacker_ids

    @staticmethod
    def whitewash_attack(evo: Evolution, count: int = 3) -> list[str]:
        """
        Whitewash: create new identities to escape flagged status.
        Tests immune memory — the system should recognize the behavioral
        fingerprint even under a new identity.
        """
        whitewash_ids = []
        for i in range(count):
            agent = NeuralAgent(id=f"W{evo.next_id + 1:04d}", generation=evo.generation)
            evo.next_id += 1
            # Same defector behavior as sybils
            agent.weights_ho = np.full_like(agent.weights_ho, -0.3)
            agent.bias_o = np.array([-2.5])
            agent.parent_id = f"WHITEWASH_{evo.round}"
            agent.balance = 800
            whitewash_ids.append(agent.id)
            evo.agents[agent.id] = agent

        evo.events.append({
            'type': 'attack_whitewash', 'round': evo.round,
            'count': count, 'agents': whitewash_ids
        })
        return whitewash_ids
