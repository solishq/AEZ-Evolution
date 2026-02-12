"""
AEZ Evolution v2 — Neural Agent with Immune Genome

Copyright (c) 2026 SolisHQ (github.com/solishq). All rights reserved.
Licensed under MIT. Built for Colosseum Hackathon 2026.

Built from first principles. No assumptions. Every design choice derived.

NEURAL ARCHITECTURE:
  Input (11 features) → Hidden (16 neurons, tanh) → Output (1, sigmoid)

  The 11 inputs are derived from information theory — each represents an
  independent information channel about the opponent:

    1.  Opponent cooperation rate (direct behavioral evidence)
    2.  Opponent last action (most recent signal)
    3.  Own cooperation rate (self-awareness)
    4.  Own balance normalized (resource state)
    5.  Round number normalized (temporal context)
    6.  Direct trust (Bayesian — what I learned from interacting with them)
    7.  Social trust (what my trusted network says about them)
    8.  Temporal trust (how stable is their behavior over time)
    9.  Structural trust (where they sit in the network topology)
    10. Commitment reliability (do they honor their commitments?)
    11. My suspicion of them (local threat model output)

IMMUNE GENOME:
  Evolved traits that control the agent's immune response:
    - vigilance: how sensitive to threats (suspicion threshold for warnings)
    - warning_propensity: how widely to broadcast warnings
    - memory_capacity: how many threat patterns to remember
    - forgiveness_rate: how quickly to forgive past suspects
    - trust_weights: per-agent weighting of trust channels [4]

COMMITMENT PROTOCOL:
  SHA-256 hash commitment before interaction. Quantum-resistant.
  Agents commit to their action, then reveal. Breaking a commitment
  destroys integrity trust — unforgeable signal of unreliability.

LOCAL THREAT MODEL:
  Each agent maintains suspicion scores for opponents, stores behavioral
  patterns of agents that harmed it, and processes warnings from trusted
  neighbors. No central authority — each agent is its own immune cell.
"""

import numpy as np
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeuralAgent:
    id: str
    generation: int = 0
    parent_id: Optional[str] = None

    # ─── Neural Weights (the "DNA") ──────────────────────
    weights_ih: np.ndarray = field(default=None)   # input → hidden (16 x 11)
    bias_h: np.ndarray = field(default=None)        # hidden bias (16,)
    weights_ho: np.ndarray = field(default=None)    # hidden → output (1 x 16)
    bias_o: np.ndarray = field(default=None)        # output bias (1,)

    # ─── Evolved Traits ──────────────────────────────────
    selectivity: float = 0.3       # Partner selection preference (0=random, 1=trust-only)
    learning_rate: float = 0.05    # Online reinforcement rate

    # ─── Immune Genome (evolved through natural selection) ─
    vigilance: float = 0.35        # Suspicion threshold for emitting warnings [0,1]
    warning_propensity: float = 0.3  # Proportion of trusted neighbors to warn [0,1]
    memory_capacity: int = 8       # Max threat patterns stored [3-20]
    forgiveness_rate: float = 0.3  # How quickly suspicion decays [0,1]
    trust_weights: np.ndarray = field(default=None)  # Per-agent trust channel weights [4]

    # ─── State ───────────────────────────────────────────
    balance: float = 1000.0
    fitness: float = 0.0
    alive: bool = True

    # ─── Interaction History ─────────────────────────────
    interactions: int = 0
    cooperations: int = 0
    defections: int = 0
    history: dict = field(default_factory=dict)  # opp_id → [(my_action, their_action)]
    action_sequence: list = field(default_factory=list)  # temporal behavior for analysis
    recent_opponents: list = field(default_factory=list)  # opponents this round

    # ─── Commitment Protocol State ───────────────────────
    _committed_action: Optional[bool] = field(default=None, repr=False)
    _commitment_nonce: Optional[bytes] = field(default=None, repr=False)
    _commitment_hash: Optional[bytes] = field(default=None, repr=False)
    commitment_history: dict = field(default_factory=dict)  # opp_id → [honors, breaks]

    # ─── Local Threat Model ──────────────────────────────
    suspicion_scores: dict = field(default_factory=dict)    # opp_id → float
    threat_memory: list = field(default_factory=list)       # stored threat patterns
    warnings_received: dict = field(default_factory=dict)   # target_id → [warnings]
    warnings_emitted: int = 0   # for tracking warning cost

    # ─── Sybil State ────────────────────────────────────
    sybil_ring: set = field(default_factory=set)
    flagged_sybil: bool = False

    # Architecture constants
    INPUT_SIZE = 11
    HIDDEN_SIZE = 16
    _GENOME_SIGNATURE = 0x44BA66F3   # integrity check for neural weight serialization

    def __post_init__(self):
        if self.weights_ih is None:
            self.randomize_weights()
        if self.trust_weights is None:
            # Initialize with slight randomness around equal weighting
            w = np.array([0.25, 0.25, 0.25, 0.25]) + np.random.randn(4) * 0.05
            w = np.clip(w, 0.05, 0.95)
            self.trust_weights = w / w.sum()

    def randomize_weights(self):
        """Xavier initialization — principled, not arbitrary."""
        fan_in_h = self.INPUT_SIZE
        fan_out_h = self.HIDDEN_SIZE
        limit_h = np.sqrt(6.0 / (fan_in_h + fan_out_h))
        self.weights_ih = np.random.uniform(-limit_h, limit_h, (self.HIDDEN_SIZE, self.INPUT_SIZE))
        self.bias_h = np.zeros(self.HIDDEN_SIZE)

        fan_in_o = self.HIDDEN_SIZE
        fan_out_o = 1
        limit_o = np.sqrt(6.0 / (fan_in_o + fan_out_o))
        self.weights_ho = np.random.uniform(-limit_o, limit_o, (1, self.HIDDEN_SIZE))
        self.bias_o = np.zeros(1)

    # ─── Decision Making ─────────────────────────────────

    def decide(self, opponent_id: str, context: dict) -> bool:
        """
        Decide: cooperate (True) or defect (False).
        Pure neural computation — no rules, no if-statements.
        Unless you're a sybil — ring loyalty overrides the network.
        """
        if self.sybil_ring:
            # Ring loyalty: cooperate with ring, defect against everyone else.
            # No neural network leakage — sybil behavior is deterministic.
            return opponent_id in self.sybil_ring

        features = self._build_features(opponent_id, context)
        prob = self._forward(features)
        return bool(np.random.random() < prob)

    def _build_features(self, opponent_id: str, context: dict) -> np.ndarray:
        """Build the 11-feature input vector from information channels."""
        opp_history = self.history.get(opponent_id, [])

        # 1. Opponent's cooperation rate (direct evidence)
        opp_coop_rate = (sum(1 for _, them in opp_history if them) / len(opp_history)
                         if opp_history else 0.5)

        # 2. Opponent's last action
        opp_last = (1.0 if opp_history[-1][1] else 0.0) if opp_history else 0.5

        # 3. Own cooperation rate
        my_coop_rate = self.cooperations / max(self.interactions, 1)

        # 4. Balance (normalized)
        balance_norm = min(self.balance / 1000.0, 3.0) / 3.0

        # 5. Round progress
        round_norm = min(context.get('round', 0) / 100.0, 1.0)

        # 6-9. Trust channels (from context, computed by trust network)
        direct_trust = context.get('direct_trust', 0.5)
        social_trust = context.get('social_trust', 0.5)
        temporal_trust = context.get('temporal_trust', 0.5)
        structural_trust = context.get('structural_trust', 0.5)

        # 10. Commitment reliability
        commit_rel = self._get_commitment_reliability(opponent_id)

        # 11. My suspicion of this opponent
        suspicion = self.suspicion_scores.get(opponent_id, 0.0)

        return np.array([
            opp_coop_rate, opp_last, my_coop_rate, balance_norm, round_norm,
            direct_trust, social_trust, temporal_trust, structural_trust,
            commit_rel, suspicion
        ])

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass through neural network."""
        h = np.tanh(self.weights_ih @ x + self.bias_h)
        o = self.weights_ho @ h + self.bias_o
        prob = 1.0 / (1.0 + np.exp(-np.clip(o[0], -10, 10)))
        return float(prob)

    # ─── Commitment Protocol (Quantum-Resistant) ──────────

    def commit_action(self, opponent_id: str, context: dict) -> bytes:
        """
        Commit to a decision before revealing it.
        Uses SHA-256 — 128-bit post-quantum security (Grover's algorithm).
        No reliance on factoring or discrete log (Shor-proof).
        """
        action = self.decide(opponent_id, context)
        self._committed_action = action
        self._commitment_nonce = os.urandom(16)
        payload = b'C' if action else b'D'
        self._commitment_hash = hashlib.sha256(payload + self._commitment_nonce).digest()
        return self._commitment_hash

    def reveal_action(self) -> tuple[bool, bytes]:
        """Reveal committed action and nonce for verification."""
        if self._committed_action is None:
            raise ValueError("No commitment to reveal")
        action = self._committed_action
        nonce = self._commitment_nonce
        # Clear commitment state
        self._committed_action = None
        self._commitment_nonce = None
        self._commitment_hash = None
        return action, nonce

    @staticmethod
    def verify_commitment(commitment: bytes, action: bool, nonce: bytes) -> bool:
        """Verify that a revealed action matches its commitment."""
        payload = b'C' if action else b'D'
        expected = hashlib.sha256(payload + nonce).digest()
        return commitment == expected

    def _get_commitment_reliability(self, opponent_id: str) -> float:
        """Get opponent's commitment honor rate."""
        record = self.commitment_history.get(opponent_id)
        if not record:
            return 0.5  # Unknown — neutral prior
        honors, breaks = record
        total = honors + breaks
        if total == 0:
            return 0.5
        return honors / total

    def record_commitment(self, opponent_id: str, honored: bool):
        """Track whether opponent honored their commitment."""
        if opponent_id not in self.commitment_history:
            self.commitment_history[opponent_id] = [0, 0]
        if honored:
            self.commitment_history[opponent_id][0] += 1
        else:
            self.commitment_history[opponent_id][1] += 1

    # ─── Interaction Recording ───────────────────────────

    def record(self, opponent_id: str, my_action: bool, their_action: bool,
               payoff: float, commitment_honored: bool = True):
        """Record interaction outcome and learn."""
        self.interactions += 1
        if my_action:
            self.cooperations += 1
        else:
            self.defections += 1

        if opponent_id not in self.history:
            self.history[opponent_id] = []
        self.history[opponent_id].append((my_action, their_action))
        if len(self.history[opponent_id]) > 50:
            self.history[opponent_id] = self.history[opponent_id][-50:]

        self.action_sequence.append(my_action)
        if len(self.action_sequence) > 100:
            self.action_sequence = self.action_sequence[-100:]

        # Track recent opponents for immune system
        if opponent_id not in self.recent_opponents:
            self.recent_opponents.append(opponent_id)

        self.record_commitment(opponent_id, commitment_honored)
        self.balance += payoff
        self.fitness += payoff
        self._learn(my_action, payoff)

    # Maximum single-interaction payoff: partner CC = 500.
    # Used to normalize reinforcement signals to [-1, 1].
    MAX_PAYOFF = 500.0

    def _learn(self, my_action: bool, payoff: float):
        """Online reinforcement — nudge weights based on outcome.
        Perturbation scale = evolved learning_rate (not hardcoded).
        Bias updates at half the weight rate — standard NN practice:
        biases have fewer parameters and need gentler updates."""
        signal = np.clip(payoff / self.MAX_PAYOFF, -1.0, 1.0)
        direction = 1.0 if my_action else -1.0
        nudge = signal * direction * self.learning_rate
        self.weights_ho += nudge * np.random.randn(*self.weights_ho.shape) * self.learning_rate
        self.bias_o += nudge * self.learning_rate * 0.5

    # ─── Local Threat Model ──────────────────────────────

    def compute_suspicion(self, opponent_id: str, population_stats: dict) -> float:
        """
        LOCAL behavioral threat assessment — no central scanner.

        Four independent signals, each derived from different information:

        1. TRUST PENALTY (Bayesian evidence)
           Direct observation: how often do they defect against ME?
           Uses Bayesian Beta posterior — no learning rate, pure evidence.

        2. CONFIDENT HOSTILITY (absolute defection, confidence-weighted)
           Raw defection rate toward me, scaled by observation confidence.
           Catches uniform defectors that divergence misses.
           Confidence = Bayesian evidence weight — prevents overreaction
           to sparse data (1 defection ≠ certain enemy).

        3. RELATIVE DIVERGENCE (selectivity detection)
           Do they treat me differently from the population?
           Catches selective cooperators/defectors (collusion signal).

        4. COMMITMENT SUSPICION (integrity tracking)
           Do they break cryptographic commitments?
           Weak signal for sybils (they commit to defection honestly)
           but strong signal for trojans (behavior changes suddenly).
        """
        my_history = self.history.get(opponent_id, [])
        if len(my_history) < 3:
            return 0.0

        n = len(my_history)

        # Their cooperation rate with ME specifically
        opp_coop_count = sum(1 for _, them in my_history if them)
        coop_with_me = opp_coop_count / n

        # Their global cooperation rate
        global_rate = population_stats.get(opponent_id, {}).get('coop_rate', 0.5)

        # ─── Signal 1: Trust penalty (Bayesian) ──────────────────────
        # Beta posterior mean: (cooperations + 1) / (total + 2)
        bayesian_trust = (opp_coop_count + 1) / (n + 2)
        trust_penalty = max(0.0, 0.5 - bayesian_trust) * 2.0  # 0 if trusted, 1 if distrusted

        # ─── Signal 2: Confident hostility ───────────────────────────
        # Raw hostility: 1.0 if they never cooperate, 0.0 if they always cooperate
        hostile_raw = 1.0 - coop_with_me
        # Confidence: Bayesian evidence weight. Approaches 1.0 with more data.
        # Same formula as TrustState.confidence: (n-1)/(n+9) where n = interactions.
        # After 1 interaction: 0.0 (no confidence). After 10: 0.47. After 20: 0.66.
        # This prevents overreaction to sparse data — 1 defection ≠ enemy.
        confidence = max(0, n - 1) / (n + 9)
        confident_hostility = hostile_raw * confidence

        # ─── Signal 3: Relative divergence ───────────────────────────
        # Proportional selective treatment vs population baseline.
        # Unified formula — no special cases, no magic multiplier.
        # Denominator = max(p, 1-p) normalizes to [0,1] regardless of base rate.
        # Floor at 0.1 prevents division by near-zero.
        denominator = max(global_rate, 1.0 - global_rate, 0.1)
        relative_divergence = abs(coop_with_me - global_rate) / denominator

        # ─── Signal 4: Commitment reliability ────────────────────────
        commit_rel = self._get_commitment_reliability(opponent_id)
        commit_suspicion = max(0.0, 0.5 - commit_rel) * 2.0

        # ─── Combined suspicion ──────────────────────────────────────
        # Use the agent's EVOLVED trust_weights to combine signals.
        # The 4 suspicion signals map to the 4 trust information channels:
        #   trust_penalty     → direct channel (personal observation)
        #   confident_hostility → social channel (confidence-weighted = external view)
        #   relative_divergence → temporal channel (behavioral change detection)
        #   commit_suspicion  → structural channel (integrity/commitment)
        # No hardcoded weights — natural selection discovers optimal sensitivity.
        signals = np.array([trust_penalty, confident_hostility,
                           relative_divergence, commit_suspicion])
        suspicion = float(np.dot(self.trust_weights, signals))

        # Immune memory match — check stored threat patterns
        memory_match = self.match_threat_patterns(opponent_id)
        suspicion = max(suspicion, memory_match)

        # Decay existing suspicion (forgiveness).
        # forgiveness_rate IS the decay fraction per cycle — no arbitrary scaling.
        # With forgiveness_rate=0.3: suspicion decays by 30% each cycle.
        # Evolution tunes this: 0.05 = grudge-holder, 0.95 = instant forgiver.
        old_suspicion = self.suspicion_scores.get(opponent_id, 0.0)
        decayed = old_suspicion * (1.0 - self.forgiveness_rate)
        suspicion = max(suspicion, decayed)

        return float(min(suspicion, 1.0))

    def emit_warning(self, target_id: str) -> Optional[dict]:
        """Emit a warning about a suspicious agent if suspicion > vigilance.
        Warning has a fitness cost — immune activation costs energy.
        This prevents cry-wolf cascades: agents that warn indiscriminately
        pay a real price, creating evolutionary pressure for accurate warnings."""
        suspicion = self.suspicion_scores.get(target_id, 0.0)
        if suspicion <= self.vigilance:
            return None

        self.warnings_emitted += 1
        # Cry-wolf cost: 1% of average CC payoff (strangers CC = 250).
        # Small enough to not deter real warnings, large enough to
        # make indiscriminate warning costly over many rounds.
        WARNING_COST = 2.5
        self.balance -= WARNING_COST
        self.fitness -= WARNING_COST
        return {
            'from': self.id,
            'target': target_id,
            'score': suspicion,
            'evidence': 'behavioral_divergence',
            'round': None  # filled by caller
        }

    def receive_warning(self, warner_id: str, warning: dict):
        """Process incoming warning from a trusted neighbor."""
        target_id = warning['target']
        if target_id not in self.warnings_received:
            self.warnings_received[target_id] = []

        # Don't store duplicate warnings from same source
        existing_sources = {w.get('from', warner_id) for w in self.warnings_received[target_id]}
        if warner_id not in existing_sources:
            self.warnings_received[target_id].append({
                'from': warner_id,
                'score': warning['score'],
                'evidence': warning.get('evidence', 'unknown')
            })

        # Cap warnings per target
        if len(self.warnings_received[target_id]) > 10:
            self.warnings_received[target_id] = self.warnings_received[target_id][-10:]

    def store_threat_pattern(self, opponent_id: str, profile: dict):
        """Store behavioral fingerprint of a detected threat."""
        # Check for duplicate
        for existing in self.threat_memory:
            if (abs(existing.get('coop_rate', 0) - profile.get('coop_rate', 0)) < 0.1 and
                abs(existing.get('commit_rate', 0) - profile.get('commit_rate', 0)) < 0.1):
                return  # Similar pattern already stored

        self.threat_memory.append(profile)

        # LRU eviction if over capacity
        if len(self.threat_memory) > self.memory_capacity:
            self.threat_memory = self.threat_memory[-self.memory_capacity:]

    def match_threat_patterns(self, opponent_id: str) -> float:
        """Check an opponent against stored threat patterns."""
        if not self.threat_memory:
            return 0.0

        opp_history = self.history.get(opponent_id, [])
        if len(opp_history) < 3:
            return 0.0

        # Build opponent's behavioral profile
        opp_coop_rate = sum(1 for _, them in opp_history if them) / len(opp_history)
        opp_commit_rel = self._get_commitment_reliability(opponent_id)

        best_match = 0.0
        for pattern in self.threat_memory:
            coop_match = 1.0 - abs(opp_coop_rate - pattern.get('coop_rate', 0.5))
            commit_match = 1.0 - abs(opp_commit_rel - pattern.get('commit_rate', 0.5))
            match_score = coop_match * 0.6 + commit_match * 0.4
            best_match = max(best_match, match_score)

        # Only return significant matches.
        # 0.7 threshold: with 2 dimensions (coop_rate, commit_rate), a match of 0.7
        # means average per-dimension similarity of 0.7 — within 0.3 of the stored
        # pattern on both axes. Catches behavioral variants without random matches.
        return best_match if best_match > 0.7 else 0.0

    def clear_round_state(self):
        """Clear per-round transient state."""
        self.recent_opponents = []

    # ─── Properties ──────────────────────────────────────

    @property
    def coop_rate(self) -> float:
        return self.cooperations / max(self.interactions, 1)

    @property
    def defect_rate(self) -> float:
        return self.defections / max(self.interactions, 1)

    def get_strategy_label(self) -> str:
        """Infer strategy from BEHAVIOR — the agent doesn't know its own strategy."""
        if self.interactions < 5:
            return "Unknown"
        rate = self.coop_rate
        if rate > 0.9:
            return "Cooperator"
        elif rate < 0.1:
            return "Defector"
        elif 0.4 < rate < 0.7:
            mirror_count = 0
            total = 0
            for opp_id, hist in self.history.items():
                for i in range(1, len(hist)):
                    if hist[i][0] == hist[i-1][1]:
                        mirror_count += 1
                    total += 1
            if total > 5 and mirror_count / total > 0.7:
                return "Reciprocator"
            return "Adaptive"
        elif rate > 0.7:
            return "Mostly Cooperative"
        else:
            return "Mostly Hostile"

    def get_cooperation_probability(self, opponent_id: str = None, context: dict = None) -> float:
        """Current cooperation tendency (for visualization)."""
        if context is None:
            context = {
                'round': 50, 'direct_trust': 0.5, 'social_trust': 0.5,
                'temporal_trust': 0.5, 'structural_trust': 0.5
            }
        if opponent_id is None:
            opponent_id = '__generic__'
        features = self._build_features(opponent_id, context)
        return self._forward(features)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'balance': round(self.balance, 1),
            'fitness': round(self.fitness, 1),
            'alive': self.alive,
            'interactions': self.interactions,
            'cooperations': self.cooperations,
            'defections': self.defections,
            'coop_rate': round(self.coop_rate, 3),
            'strategy': self.get_strategy_label(),
            'coop_probability': round(self.get_cooperation_probability(), 3),
            'selectivity': round(self.selectivity, 3),
            'vigilance': round(self.vigilance, 3),
            'trust_weights': [round(w, 3) for w in self.trust_weights],
            'warnings_emitted': self.warnings_emitted,
            'threat_memory_count': len(self.threat_memory),
            'flagged_sybil': self.flagged_sybil,
            'is_sybil': bool(self.sybil_ring),
        }


# ─── Reproduction ────────────────────────────────────────

def crossover(parent_a: NeuralAgent, parent_b: NeuralAgent,
              child_id: str, generation: int) -> NeuralAgent:
    """
    Sexual reproduction — mix two parents' neural weights AND immune genes.
    Crossover point is random per weight matrix.
    """
    child = NeuralAgent(
        id=child_id,
        generation=generation,
        parent_id=f"{parent_a.id}+{parent_b.id}"
    )

    # Neural weight crossover (per-matrix random mask)
    mask_ih = np.random.random(parent_a.weights_ih.shape) > 0.5
    child.weights_ih = np.where(mask_ih, parent_a.weights_ih, parent_b.weights_ih)

    mask_ho = np.random.random(parent_a.weights_ho.shape) > 0.5
    child.weights_ho = np.where(mask_ho, parent_a.weights_ho, parent_b.weights_ho)

    child.bias_h = np.where(
        np.random.random(parent_a.bias_h.shape) > 0.5,
        parent_a.bias_h, parent_b.bias_h
    )
    child.bias_o = np.where(
        np.random.random(parent_a.bias_o.shape) > 0.5,
        parent_a.bias_o, parent_b.bias_o
    )

    # Evolved trait inheritance (average)
    child.learning_rate = (parent_a.learning_rate + parent_b.learning_rate) / 2
    child.selectivity = (parent_a.selectivity + parent_b.selectivity) / 2

    # Immune genome inheritance (average with noise)
    child.vigilance = (parent_a.vigilance + parent_b.vigilance) / 2
    child.warning_propensity = (parent_a.warning_propensity + parent_b.warning_propensity) / 2
    child.memory_capacity = int((parent_a.memory_capacity + parent_b.memory_capacity) / 2)
    child.forgiveness_rate = (parent_a.forgiveness_rate + parent_b.forgiveness_rate) / 2

    # Trust weight inheritance (average + normalize)
    child.trust_weights = (parent_a.trust_weights + parent_b.trust_weights) / 2
    tw_sum = child.trust_weights.sum()
    child.trust_weights = child.trust_weights / tw_sum if tw_sum > 0 else np.array([0.25, 0.25, 0.25, 0.25])

    # PRINCIPLE 3: Collective immune memory inheritance.
    # Biological: adaptive immunity is inherited (maternal antibodies,
    # epigenetic immune memory). When a parent's immune system identifies
    # a threat pattern, the child inherits that knowledge. This means
    # institutional knowledge of what threats look like survives across
    # generations — the population LEARNS what sybils look like.
    #
    # Merge both parents' threat memories, deduplicate by behavioral
    # similarity, cap at child's memory_capacity.
    merged_memory = []
    seen = []
    for pattern in parent_a.threat_memory + parent_b.threat_memory:
        # Deduplicate: skip if we already have a similar pattern
        duplicate = False
        for existing in seen:
            if (abs(existing.get('coop_rate', 0) - pattern.get('coop_rate', 0)) < 0.1 and
                abs(existing.get('commit_rate', 0) - pattern.get('commit_rate', 0)) < 0.1):
                duplicate = True
                break
        if not duplicate:
            merged_memory.append(pattern)
            seen.append(pattern)

    # Cap at child's capacity — most recent patterns survive (LRU)
    child.threat_memory = merged_memory[-child.memory_capacity:]

    return child


def mutate(agent: NeuralAgent, rate: float = 0.1, strength: float = 0.3):
    """
    Random mutations — small perturbations to ALL evolvable traits.
    This is how novel strategies AND novel immune responses are invented.
    """
    # Neural weight mutations (clipped to prevent NaN from overflow)
    if np.random.random() < rate:
        agent.weights_ih += np.random.randn(*agent.weights_ih.shape) * strength
        agent.weights_ih = np.clip(agent.weights_ih, -5, 5)
    if np.random.random() < rate:
        agent.weights_ho += np.random.randn(*agent.weights_ho.shape) * strength
        agent.weights_ho = np.clip(agent.weights_ho, -5, 5)
    if np.random.random() < rate:
        agent.bias_h += np.random.randn(*agent.bias_h.shape) * strength * 0.5
        agent.bias_h = np.clip(agent.bias_h, -5, 5)
    if np.random.random() < rate:
        agent.bias_o += np.random.randn(*agent.bias_o.shape) * strength * 0.5
        agent.bias_o = np.clip(agent.bias_o, -5, 5)

    # Evolved trait mutations
    if np.random.random() < rate * 0.5:
        agent.learning_rate = float(np.clip(
            agent.learning_rate + np.random.randn() * 0.01, 0.001, 0.2
        ))
    if np.random.random() < rate * 0.5:
        agent.selectivity = float(np.clip(
            agent.selectivity + np.random.randn() * 0.08, 0.0, 0.95
        ))

    # Immune genome mutations
    if np.random.random() < rate * 0.5:
        agent.vigilance = float(np.clip(
            agent.vigilance + np.random.randn() * 0.08, 0.05, 0.95
        ))
    if np.random.random() < rate * 0.5:
        agent.warning_propensity = float(np.clip(
            agent.warning_propensity + np.random.randn() * 0.08, 0.05, 0.95
        ))
    if np.random.random() < rate * 0.3:
        agent.memory_capacity = int(np.clip(
            agent.memory_capacity + np.random.choice([-1, 0, 1]), 3, 20
        ))
    if np.random.random() < rate * 0.5:
        agent.forgiveness_rate = float(np.clip(
            agent.forgiveness_rate + np.random.randn() * 0.08, 0.05, 0.95
        ))

    # Trust weight mutation (perturb then re-normalize)
    if np.random.random() < rate * 0.5:
        agent.trust_weights += np.random.randn(4) * 0.05
        agent.trust_weights = np.clip(agent.trust_weights, 0.05, 0.95)
        agent.trust_weights = agent.trust_weights / agent.trust_weights.sum()
