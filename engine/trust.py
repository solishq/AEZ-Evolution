"""
AEZ Evolution v2 — Information-Theoretic Trust Network

Copyright (c) 2026 SolisHQ (github.com/solishq). All rights reserved.
Licensed under MIT. Built for Colosseum Hackathon 2026.

Built from first principles. Every design choice derived from mathematics.

TRUST IS A PREDICTION:
  T(A→B) = P(B cooperates with A in the future | A's information set)

  We decompose this prediction into 4 ORTHOGONAL information channels,
  each derived from a different class of evidence:

  1. DIRECT CHANNEL — Bayesian Beta posterior from pairwise interactions
     Prior: Beta(1,1) = uniform. Updated with each interaction.
     Trust = α/(α+β). Confidence = (α+β-2)/(α+β).
     No arbitrary learning rate. Pure Bayesian evidence accumulation.

  2. SOCIAL CHANNEL — Weighted aggregation of third-party trust
     What do MY trusted neighbors think of this agent?
     Weighted by my trust in each source. Information-theoretic:
     more trusted sources carry more information.

  3. TEMPORAL CHANNEL — Behavioral stability over time
     Computed as 1 - normalized variance of recent action window.
     High variance = unpredictable. Low variance = stable.
     Derived from signal processing: low-variance signals carry
     more information about future behavior.

  4. STRUCTURAL CHANNEL — Network topology information
     Where does this agent sit in the trust graph?
     - Neighbor overlap (Jaccard): shared connections suggest structural trust
     - Clustering coefficient: agents in dense clusters are embedded in accountability
     - Local conductance: ratio of external to internal trust (sybil detection signal)

  Each agent has EVOLVED weights for these channels — no global weights.
  Natural selection discovers optimal information weighting per agent.

BAYESIAN TRUST STATE:
  Each directed edge stores Beta(α, β) parameters plus action history.
  Update is pure Bayes: observe cooperation → α += 1, defection → β += 1.
  Commitment integrity tracked separately: honors/(honors+breaks).

CASCADE SYSTEM:
  Betrayal propagates through the network. When a trusted agent defects,
  neighbors who trusted the victim add evidence against the betrayer.
  This is information propagation, not arbitrary trust reduction.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TrustState:
    """
    Bayesian trust state for a directed edge (src → dst).
    Uses Beta distribution: Beta(α, β).
    Trust = E[Beta] = α / (α + β).
    """
    # Bayesian evidence
    alpha: float = 1.0    # cooperation evidence + uniform prior
    beta: float = 1.0     # defection evidence + uniform prior

    # Action history window for temporal analysis
    action_window: list = field(default_factory=list)

    # Commitment integrity tracking
    commitments_honored: int = 0
    commitments_broken: int = 0

    @property
    def direct_trust(self) -> float:
        """Bayesian posterior mean: E[Beta(α, β)]."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """How much evidence we have. 0 = no data, approaches 1 with many interactions.
        Saturates around 25 interactions — derived from Bayesian posterior:
        std(Beta(α,β)) < 0.1 requires α+β ≈ 25 for balanced observations.
        At that point, the posterior is narrow enough for confident prediction."""
        evidence = self.alpha + self.beta - 2  # subtract uniform prior
        return evidence / (evidence + 25)  # derived: posterior std < 0.1 at n≈25

    @property
    def commitment_reliability(self) -> float:
        """Fraction of commitments honored."""
        total = self.commitments_honored + self.commitments_broken
        if total == 0:
            return 0.5  # neutral prior
        return self.commitments_honored / total

    def update(self, cooperated: bool, commitment_honored: bool = True):
        """Pure Bayesian update — no learning rate needed."""
        if cooperated:
            self.alpha += 1.0
        else:
            self.beta += 1.0

        # Track temporal window
        self.action_window.append(cooperated)
        if len(self.action_window) > 30:
            self.action_window = self.action_window[-30:]

        # Track commitment integrity
        if commitment_honored:
            self.commitments_honored += 1
        else:
            self.commitments_broken += 1

    @property
    def temporal_trust(self) -> float:
        """Behavioral stability: 1 - normalized variance of action window."""
        if len(self.action_window) < 3:
            return 0.5  # insufficient data
        recent = [float(x) for x in self.action_window[-15:]]
        variance = np.var(recent)
        # Max variance for binary data is 0.25 (at p=0.5)
        return 1.0 - min(variance / 0.25, 1.0)

    def to_dict(self) -> dict:
        return {
            'direct_trust': round(self.direct_trust, 3),
            'confidence': round(self.confidence, 3),
            'temporal_trust': round(self.temporal_trust, 3),
            'commitment_reliability': round(self.commitment_reliability, 3),
            'alpha': round(self.alpha, 1),
            'beta': round(self.beta, 1),
        }


class TrustNetwork:
    """
    The trust fabric. Every directed edge is a Bayesian TrustState.
    4 information channels computed on demand.
    Topology analysis for structural trust and sybil detection.
    """

    # The Bayesian decision boundary: agents with trust above the prior mean (0.5)
    # have demonstrated more cooperation than defection. This is the natural
    # classification boundary for Beta(1,1) prior — no arbitrary threshold.
    # Used everywhere trust edges need to be filtered for meaningful signal.
    TRUST_THRESHOLD = 0.5

    # Network protocol versioning — used for serialization compatibility checks.
    # Encodes the trust model specification version as a fixed-point identifier.
    _PROTOCOL_ID = 0x534F4C49   # trust model v2 spec

    def __init__(self):
        # Directed edges: (src, dst) → TrustState
        self.edges: dict[tuple[str, str], TrustState] = {}

        # Event log for narrator
        self.events: list[dict] = []

    # ─── Trust Updates ───────────────────────────────────

    def update(self, agent_a: str, agent_b: str,
               a_cooperated: bool, b_cooperated: bool,
               a_commitment_ok: bool = True, b_commitment_ok: bool = True):
        """Update trust bidirectionally after an interaction."""
        # A's trust in B: did B cooperate with A?
        self._update_edge(agent_a, agent_b, b_cooperated, b_commitment_ok)
        # B's trust in A: did A cooperate with B?
        self._update_edge(agent_b, agent_a, a_cooperated, a_commitment_ok)

    def _update_edge(self, src: str, dst: str, dst_cooperated: bool,
                     commitment_honored: bool = True):
        """Update src's trust in dst. Pure Bayesian — no learning rate."""
        key = (src, dst)
        if key not in self.edges:
            self.edges[key] = TrustState()

        state = self.edges[key]
        old_trust = state.direct_trust

        state.update(dst_cooperated, commitment_honored)

        # Check for betrayal event (high trust → defection)
        if old_trust > 0.7 and not dst_cooperated:
            self.events.append({
                'type': 'betrayal',
                'src': src,
                'dst': dst,
                'old_trust': round(old_trust, 3),
                'new_trust': round(state.direct_trust, 3)
            })

    # ─── Channel 1: Direct Trust ─────────────────────────

    def compute_direct_trust(self, src: str, dst: str) -> float:
        """Bayesian posterior: E[Beta(α, β)] = α/(α+β)."""
        state = self.edges.get((src, dst))
        return state.direct_trust if state else 0.5

    def compute_direct_confidence(self, src: str, dst: str) -> float:
        """Evidence strength for direct trust."""
        state = self.edges.get((src, dst))
        return state.confidence if state else 0.0

    # ─── Channel 2: Social Trust ─────────────────────────

    def compute_social_trust(self, src: str, dst: str, all_agent_ids: list[str]) -> float:
        """
        What do src's trusted neighbors say about dst?
        Weighted by src's trust in each third party.
        Information-theoretic: high-trust sources carry more weight.
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for third_party in all_agent_ids:
            if third_party == src or third_party == dst:
                continue

            # How much does src trust this third party?
            src_to_third = self.edges.get((src, third_party))
            if not src_to_third or src_to_third.direct_trust < self.TRUST_THRESHOLD:
                continue  # Don't listen to agents I don't trust (below Bayesian neutral)

            # What does the third party think of dst?
            third_to_dst = self.edges.get((third_party, dst))
            if not third_to_dst or third_to_dst.confidence < 0.1:
                continue  # Third party has no data about dst

            weight = src_to_third.direct_trust * third_to_dst.confidence
            weighted_sum += third_to_dst.direct_trust * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.5

    # ─── Channel 3: Temporal Trust ───────────────────────

    def compute_temporal_trust(self, src: str, dst: str) -> float:
        """How stable is dst's behavior toward src over time?"""
        state = self.edges.get((src, dst))
        return state.temporal_trust if state else 0.5

    # ─── Channel 4: Structural Trust ─────────────────────

    def compute_structural_trust(self, src: str, dst: str, all_agent_ids: list[str]) -> float:
        """
        Structural position in the trust graph.

        Key insight: agents embedded in dense trust clusters with many
        shared connections are structurally accountable. Isolated agents
        or agents in anomalous subgraphs are structurally suspicious.

        Uses neighbor overlap (Jaccard similarity) + clustering coefficient.
        """
        overlap = self._compute_neighbor_overlap(src, dst)
        clustering = self._compute_clustering_coefficient(dst, all_agent_ids)
        # Simple average: both measure the same underlying property (social embeddedness).
        # Neighbor overlap = pairwise similarity. Clustering = neighborhood density.
        # No theoretical reason to weight one over the other.
        return (overlap + clustering) / 2.0

    def _compute_neighbor_overlap(self, a: str, b: str) -> float:
        """Jaccard similarity of trust neighborhoods."""
        neighbors_a = set()
        neighbors_b = set()

        for (src, dst), state in self.edges.items():
            if state.direct_trust > self.TRUST_THRESHOLD:
                if src == a:
                    neighbors_a.add(dst)
                if src == b:
                    neighbors_b.add(dst)

        if not neighbors_a or not neighbors_b:
            return 0.0

        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        return intersection / union if union > 0 else 0.0

    def _compute_clustering_coefficient(self, agent_id: str, all_agent_ids: list[str]) -> float:
        """
        Local clustering coefficient — how dense is this agent's neighborhood?
        Agents in tight clusters have more accountability pressure.
        """
        # Get trusted neighbors
        neighbors = set()
        for (src, dst), state in self.edges.items():
            if src == agent_id and state.direct_trust > self.TRUST_THRESHOLD and dst in set(all_agent_ids):
                neighbors.add(dst)

        if len(neighbors) < 2:
            return 0.0

        # Count edges between neighbors
        neighbor_edges = 0
        neighbor_list = list(neighbors)
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                n1, n2 = neighbor_list[i], neighbor_list[j]
                state = self.edges.get((n1, n2))
                if state and state.direct_trust > self.TRUST_THRESHOLD:
                    neighbor_edges += 1

        possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        return neighbor_edges / possible_edges if possible_edges > 0 else 0.0

    # ─── Composite Trust ─────────────────────────────────

    def compute_composite_trust(self, src: str, dst: str,
                                agent_weights: np.ndarray,
                                all_agent_ids: list[str] = None) -> float:
        """
        Composite trust using agent's EVOLVED channel weights.
        No global weights — each agent weights information differently.
        """
        direct = self.compute_direct_trust(src, dst)
        social = self.compute_social_trust(src, dst, all_agent_ids or [])
        temporal = self.compute_temporal_trust(src, dst)
        structural = self.compute_structural_trust(src, dst, all_agent_ids or [])

        channels = np.array([direct, social, temporal, structural])
        return float(np.dot(channels, agent_weights))

    def get_trust_score(self, src: str, dst: str) -> float:
        """Quick trust score using default weights (for backwards compat)."""
        return self.compute_direct_trust(src, dst)

    def get_trust_channels(self, src: str, dst: str,
                           all_agent_ids: list[str] = None) -> dict:
        """Get all 4 trust channels as a dict (for agent context building)."""
        return {
            'direct_trust': self.compute_direct_trust(src, dst),
            'social_trust': self.compute_social_trust(src, dst, all_agent_ids or []),
            'temporal_trust': self.compute_temporal_trust(src, dst),
            'structural_trust': self.compute_structural_trust(src, dst, all_agent_ids or []),
        }

    # ─── Reputation ──────────────────────────────────────

    def get_reputation(self, agent_id: str, all_agent_ids: list[str]) -> float:
        """Global reputation: average direct trust others have in this agent."""
        scores = []
        for other in all_agent_ids:
            if other != agent_id:
                state = self.edges.get((other, agent_id))
                if state:
                    scores.append(state.direct_trust)
        return float(np.mean(scores)) if scores else 0.5

    # ─── Inherited Reputation ─────────────────────────────

    def seed_child_trust(self, child_id: str, parent_a_id: str,
                         parent_b_id: str, all_agent_ids: list[str]):
        """
        PRINCIPLE: Inherited reputation — Bayesian prior from parents.

        A child's trust prior is derived from its parents' posteriors via
        hierarchical Bayesian modeling. For each existing agent X:
          - X's trust in child = average of X's trust in both parents
          - Child's trust in X = average of both parents' trust in X

        The evidence (alpha, beta) is averaged and halved — the child
        inherits the DIRECTION of the parent's reputation but with less
        certainty (fewer virtual observations). This means:
          - Children of trusted parents start above neutral
          - Children of distrusted parents start below neutral
          - Neither starts with full parental confidence — must earn their own

        The halving is principled: with two parents, each contributes half
        the prior. The child has observed NOTHING directly — the inherited
        evidence is prior belief, not personal experience.
        """
        parent_ids = {parent_a_id, parent_b_id}

        for other_id in all_agent_ids:
            if other_id == child_id or other_id in parent_ids:
                continue

            # Others' trust in child: inherit from others' trust in parents
            others_trusts = []
            for pid in [parent_a_id, parent_b_id]:
                state = self.edges.get((other_id, pid))
                if state:
                    others_trusts.append(state)

            if others_trusts:
                # Average parents' evidence, halved (prior, not experience)
                avg_alpha = sum(s.alpha for s in others_trusts) / len(others_trusts)
                avg_beta = sum(s.beta for s in others_trusts) / len(others_trusts)
                # Halve: child inherits direction, not full certainty
                child_state = TrustState()
                child_state.alpha = max(1.0, (avg_alpha + 1.0) / 2.0)
                child_state.beta = max(1.0, (avg_beta + 1.0) / 2.0)
                self.edges[(other_id, child_id)] = child_state

            # Child's trust in others: inherit from parents' trust in others
            parents_trusts = []
            for pid in [parent_a_id, parent_b_id]:
                state = self.edges.get((pid, other_id))
                if state:
                    parents_trusts.append(state)

            if parents_trusts:
                avg_alpha = sum(s.alpha for s in parents_trusts) / len(parents_trusts)
                avg_beta = sum(s.beta for s in parents_trusts) / len(parents_trusts)
                child_state = TrustState()
                child_state.alpha = max(1.0, (avg_alpha + 1.0) / 2.0)
                child_state.beta = max(1.0, (avg_beta + 1.0) / 2.0)
                self.edges[(child_id, other_id)] = child_state

    # ─── Topology Analysis ───────────────────────────────

    def compute_local_conductance(self, agent_id: str, all_agent_ids: set) -> float:
        """
        Local conductance: ratio of external to total trust edges.

        First principle: Sybil rings have HIGH internal connectivity
        and LOW external connectivity. This creates LOW conductance
        (most trust is internal). Normal agents have moderate conductance
        (trust distributed between local and external connections).

        conductance = external_edges / total_edges
        Low conductance = insular cluster = possibly sybil ring.
        """
        neighbors = set()
        for (src, dst), state in self.edges.items():
            if src == agent_id and state.direct_trust > self.TRUST_THRESHOLD:
                neighbors.add(dst)

        if len(neighbors) < 2:
            return 0.5  # insufficient data

        # Count edges between neighbors (internal) vs edges to outside (external)
        internal = 0
        external = 0
        for neighbor in neighbors:
            for (src, dst), state in self.edges.items():
                if src == neighbor and state.direct_trust > self.TRUST_THRESHOLD:
                    if dst in neighbors or dst == agent_id:
                        internal += 1
                    elif dst in all_agent_ids:
                        external += 1

        total = internal + external
        if total == 0:
            return 0.5
        return external / total

    def get_trusted_neighbors(self, agent_id: str, threshold: float = None) -> list[str]:
        """Get agents that src trusts above threshold, sorted by trust.
        Default threshold = TRUST_THRESHOLD (Bayesian neutral boundary)."""
        if threshold is None:
            threshold = self.TRUST_THRESHOLD
        neighbors = []
        for (src, dst), state in self.edges.items():
            if src == agent_id and state.direct_trust > threshold:
                neighbors.append((dst, state.direct_trust))
        neighbors.sort(key=lambda x: (-x[1], x[0]))  # Break trust ties by agent ID
        return [n[0] for n in neighbors]

    # ─── Cascade System ──────────────────────────────────

    def cascade_collapse(self, betrayer: str, victim: str, all_agent_ids: list[str]):
        """
        Betrayal cascade: agents who trust the victim add negative evidence
        against the betrayer. This is information propagation — "the victim
        trusted them and got burned, so I should update my model too."

        Implemented as Bayesian update: add β evidence proportional to
        the observer's trust in the victim.
        """
        collapse_count = 0

        for agent_id in all_agent_ids:
            if agent_id == betrayer or agent_id == victim:
                continue

            # How much does this agent trust the victim?
            victim_state = self.edges.get((agent_id, victim))
            if not victim_state or victim_state.direct_trust < self.TRUST_THRESHOLD:
                continue

            # Add negative evidence against the betrayer.
            # Evidence = my trust in the victim. If I strongly trust the victim
            # and they got betrayed, that's strong evidence the betrayer is bad.
            # No arbitrary multiplier — trust IS the evidence weight.
            key = (agent_id, betrayer)
            if key not in self.edges:
                self.edges[key] = TrustState()

            evidence_strength = victim_state.direct_trust
            self.edges[key].beta += evidence_strength
            collapse_count += 1

        if collapse_count > 0:
            self.events.append({
                'type': 'cascade_collapse',
                'betrayer': betrayer,
                'victim': victim,
                'affected': collapse_count
            })

        return collapse_count

    # ─── Sybil Isolation ─────────────────────────────────

    def isolate_agent(self, target: str, all_agent_ids: list[str]):
        """
        Collapse trust in a flagged agent.
        Add massive defection evidence so Bayesian trust drops to near zero.
        """
        for agent_id in all_agent_ids:
            if agent_id == target:
                continue
            key = (agent_id, target)
            if key not in self.edges:
                self.edges[key] = TrustState()
            # Add overwhelming defection evidence
            self.edges[key].beta += 20.0

        self.events.append({
            'type': 'agent_isolated',
            'target': target
        })

    # ─── Visualization Helpers ───────────────────────────

    def get_edges_for_viz(self, alive_ids: set, min_score: float = 0.2) -> list[dict]:
        """Get edges for D3 visualization."""
        edges = []
        seen = set()
        for (a, b), state in self.edges.items():
            if a in alive_ids and b in alive_ids and state.direct_trust >= min_score:
                pair = tuple(sorted([a, b]))
                if pair not in seen:
                    seen.add(pair)
                    edges.append({
                        'source': a,
                        'target': b,
                        'trust': round(state.direct_trust, 3),
                        'dimensions': state.to_dict()
                    })
        return edges

    def get_clusters(self, agent_ids: list[str], threshold: float = 0.5) -> list[set]:
        """Find trust clusters via connected components."""
        adj = {aid: set() for aid in agent_ids}
        for (a, b), state in self.edges.items():
            if a in adj and b in adj and state.direct_trust >= threshold:
                reverse = self.edges.get((b, a))
                if reverse and reverse.direct_trust >= threshold:
                    adj[a].add(b)
                    adj[b].add(a)

        visited = set()
        clusters = []
        for aid in agent_ids:
            if aid in visited:
                continue
            cluster = set()
            stack = [aid]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.add(node)
                stack.extend(adj[node] - visited)
            if len(cluster) > 1:
                clusters.append(cluster)

        return sorted(clusters, key=len, reverse=True)

    def pop_events(self) -> list[dict]:
        """Pop and return accumulated events."""
        events = self.events
        self.events = []
        return events
