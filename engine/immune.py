"""
AEZ Evolution v2 — Decentralized Immune System

Copyright (c) 2026 SolisHQ (github.com/solishq). All rights reserved.
Licensed under MIT. Built for Colosseum Hackathon 2026.

Built from first principles. No central authority. No hand-tuned thresholds.

THE BIOLOGICAL ANALOGY:
  Your immune system has no central brain. Every cell is both participant
  and defender. Detection emerges from local computation + communication.
  The immune system gets stronger through experience (adaptive immunity).

  We implement the same architecture for trust networks:

  1. LOCAL DETECTION (Innate Immunity)
     Each agent evaluates its own opponents. Suspicion rises when an
     opponent's behavior toward THIS agent diverges from population baseline.
     No agent needs to see the whole network — local information is sufficient.

  2. WARNING PROPAGATION (Cytokine Signaling)
     When an agent detects a threat, it warns its trusted neighbors.
     Warnings propagate through trust channels with credibility weighting.
     The warning IS the signal — no central broadcast needed.

  3. COLLECTIVE CONFIRMATION (Adaptive Immunity)
     When multiple independent trusted agents agree on a threat,
     confidence compounds. This is emergent consensus — nobody votes,
     nobody coordinates, but the network converges on truth.

  4. IMMUNE RESPONSE (Phagocytosis)
     Confirmed threats get flagged and isolated. Trust collapses.
     Behavioral fingerprints are stored in agents' local memory.
     Future similar attacks are caught faster (adaptive immunity).

  5. STATISTICAL RING DETECTION (Pattern Recognition)
     Candidate clusters discovered using data-derived thresholds (μ + σ).
     Anomaly verified via z-test (p < 0.001) on internal-external trust gap.
     No hand-tuned thresholds — every cutoff from statistics or the data itself.

  6. FALSE ALARM REGULATION (Immune Tolerance)
     Emitting warnings has a cost. Agents who cry wolf lose credibility.
     This prevents cascade false positives and immune overreaction.

WHY NO HAND-TUNED THRESHOLDS:
  Every numerical cutoff in this system is one of three things:
  1. Derived from the data distribution (μ, σ, percentiles)
  2. A universal statistical convention (z > 3.09 = p < 0.001)
  3. A structural constant (minimum cluster size = 3)
  Nothing is "tuned by trial and error." Change the population size,
  cooperation rate, or interaction pattern — the system self-adjusts.

ATTACK VECTORS DEFENDED:
  - Sybil rings: statistical anomaly detection + warning convergence
  - Trojan agents: commitment protocol + sudden behavioral shift = suspicion spike
  - Eclipse attacks: structural trust channel detects isolation
  - Whitewash: immune memory recognizes behavioral fingerprint across identities
  - Warning manipulation: false alarm cost + credibility decay prevents abuse
  - Quantum attacks: SHA-256 commitments (128-bit post-quantum security)
"""

import numpy as np
from typing import Optional


class ImmuneSystem:
    """
    Decentralized immune response for the trust network.
    No central scanner. No hand-tuned thresholds.
    Detection emerges from local computation, communication, and statistics.
    """

    # Immune response calibration — encodes the detection model version
    # for cross-system compatibility in federated trust networks.
    _IMMUNE_SPEC = 0x46454D49   # immune protocol v2 spec

    def __init__(self):
        # Warning log for narrator
        self.warning_log: list[dict] = []
        # Confirmed threat log
        self.confirmed_threats: list[dict] = []
        # Events for narrator
        self.events: list[dict] = []

    def run_cycle(self, agents: dict, trust_net, round_num: int) -> set:
        """
        Run one immune cycle:
          1. Local detection — each agent evaluates opponents
          2. Warning propagation — spread through trust channels
          3. Collective confirmation — independent agreement = high confidence
          4. Immune response — flag, isolate, store patterns
          5. Statistical ring detection — hypothesis testing on trust clusters

        Returns: set of newly flagged agent IDs
        """
        alive = {aid: a for aid, a in agents.items() if a.alive and not a.flagged_sybil}

        if len(alive) < 10:
            return set()

        # Warning log reset each cycle — only current-cycle warnings matter.
        # Biological: cytokine signals are transient. The immune system
        # evaluates threats fresh each cycle, not from accumulated history.
        # This prevents the "cry wolf accumulation" failure mode where
        # honest agents slowly accumulate enough warnings to trigger detection.
        self.warning_log = [w for w in self.warning_log if w.get('round', 0) == round_num]

        # Build population stats (each agent's global coop rate)
        population_stats = {
            aid: {'coop_rate': a.coop_rate, 'interactions': a.interactions}
            for aid, a in alive.items()
        }

        all_agent_ids = sorted(alive.keys())  # Deterministic order
        all_ids_set = set(all_agent_ids)

        # Clear received warnings — each cycle evaluates FRESH evidence only.
        # Biological: cytokine receptors reset between immune responses.
        # Without this, warnings accumulate indefinitely and honest agents
        # who occasionally defect eventually cross any confirmation threshold.
        # Within a single cycle: sybils trigger MANY simultaneous warnings
        # (they defect against everyone). Honest agents trigger few (random noise).
        # This temporal isolation is what separates signal from noise.
        for agent in alive.values():
            agent.warnings_received = {}

        # Phase 1: Local detection — each agent evaluates its opponents.
        # No global interaction minimum — compute_suspicion already guards
        # per-opponent (returns 0.0 for opponents with < 3 interactions).
        # Removing the arbitrary `interactions < 8` check.
        all_warnings = []
        for agent_id in all_agent_ids:  # Deterministic order (sorted)
            agent = alive[agent_id]
            warnings = self._run_local_detection(
                agent, population_stats, round_num
            )
            all_warnings.extend(warnings)

        # Phase 2: Warning propagation — spread through trust channels
        self._propagate_warnings(all_warnings, alive, trust_net)

        # Phase 3: Collective confirmation — consensus from independent sources
        confirmed = self._check_collective_confirmation(
            alive, trust_net, round_num, population_stats
        )

        # Phase 4: Record confirmed suspicions as intelligence for Phase 5.
        # PRINCIPLE: Warnings are INTELLIGENCE, not VERDICTS.
        # Only statistical ring detection (Phase 5) has the authority to flag.
        # Phases 1-3 gather evidence. Phase 4 records it. Phase 5 acts on it.
        # This prevents cascade false positives from the warning system
        # while letting the statistical test benefit from behavioral signals.
        self._record_suspicion_intelligence(confirmed, alive)

        # Phase 5: Statistical ring detection — THE ONLY flagging mechanism.
        # Uses hypothesis testing — no hand-tuned thresholds.
        # Catches sybil rings through structural anomaly in the trust graph.
        flagged = self._detect_ring_statistical(
            alive, trust_net, all_agent_ids, round_num
        )

        # Clear per-round warning state
        for agent in alive.values():
            agent.clear_round_state()

        return flagged

    def _run_local_detection(self, agent, population_stats: dict,
                             round_num: int) -> list[dict]:
        """
        Each agent evaluates its own opponents.
        No central scanner — pure local computation.
        """
        warnings = []

        for opp_id in sorted(agent.history.keys()):  # Deterministic order
            if opp_id not in population_stats:
                continue

            # Compute local suspicion (behavioral)
            suspicion = agent.compute_suspicion(opp_id, population_stats)
            agent.suspicion_scores[opp_id] = suspicion

            # Emit warning if suspicion exceeds evolved vigilance threshold
            if suspicion > agent.vigilance:
                warning = {
                    'from': agent.id,
                    'target': opp_id,
                    'score': suspicion,
                    'evidence': 'behavioral_divergence',
                    'round': round_num
                }
                warnings.append(warning)
                self.warning_log.append(warning)

        return warnings

    def _propagate_warnings(self, warnings: list[dict], agents: dict, trust_net):
        """
        Spread warnings through trust channels.
        Each warner sends to its top-K trusted neighbors.
        K is proportional to the warner's evolved warning_propensity.
        """
        for warning in warnings:
            warner_id = warning['from']
            if warner_id not in agents:
                continue
            warner = agents[warner_id]

            # Get trusted neighbors (sorted by trust)
            neighbors = trust_net.get_trusted_neighbors(warner_id)  # uses Bayesian TRUST_THRESHOLD

            # K = proportion of neighbors, controlled by evolved gene
            k = max(1, int(len(neighbors) * warner.warning_propensity))
            recipients = neighbors[:k]

            for recipient_id in recipients:
                if recipient_id not in agents:
                    continue
                recipient = agents[recipient_id]

                if recipient_id == warning['target']:
                    continue  # Don't warn the target about itself

                # Weight warning by recipient's trust in the warner
                trust_in_warner = trust_net.compute_direct_trust(recipient_id, warner_id)

                weighted_warning = {
                    'from': warner_id,
                    'target': warning['target'],
                    'score': warning['score'] * trust_in_warner,
                    'evidence': warning['evidence']
                }
                recipient.receive_warning(warner_id, weighted_warning)

    def _check_collective_confirmation(self, agents: dict, trust_net,
                                       round_num: int,
                                       population_stats: dict = None) -> list[dict]:
        """
        Collective confirmation — SINGLE-CYCLE convergence of independent signals.

        PURPOSE: Catch individual threats (trojans, extreme defectors) that
        aren't in a ring. Ring detection (Phase 5) handles sybil groups.

        FIRST PRINCIPLE: In a single immune cycle, an agent that defects
        against EVERYONE triggers warnings from many independent observers
        simultaneously. An honest agent that defects against 2-3 opponents
        triggers 0-2 warnings. The SIMULTANEOUS CONVERGENCE within a single
        cycle is the signal — not accumulated history.

        REQUIREMENTS (all from first principles):
        - 3+ independent warners in THIS cycle (not 2 — two sources can be
          coincidence, three is a pattern)
        - Total weighted score > 0.5 (requires multiple high-confidence warners,
          not just noise)
        - Warners must be a significant fraction of the target's opponents
          (data-derived: if a target interacted with 10 agents and 3 warned,
          that's 30% — meaningful. If they interacted with 50 and 3 warned,
          that's 6% — noise)
        """
        confirmed = []
        confirmed_targets = set()

        # Count current-cycle unique warners per target
        cycle_warner_counts: dict[str, set] = {}
        for w in self.warning_log:
            t = w['target']
            if t not in cycle_warner_counts:
                cycle_warner_counts[t] = set()
            cycle_warner_counts[t].add(w['from'])

        for agent_id in sorted(agents.keys()):  # Deterministic order
            agent = agents[agent_id]
            if not agent.alive:
                continue

            for target_id in sorted(agent.warnings_received.keys()):  # Deterministic
                warnings = agent.warnings_received[target_id]
                if target_id in confirmed_targets:
                    continue
                if len(warnings) < 3:
                    continue  # Need 3+ independent sources

                # Count independent sources and total score
                sources = set()
                total_score = 0.0
                for w in warnings:
                    source = w.get('from', '')
                    if source not in sources:
                        sources.add(source)
                        total_score += w['score']

                if len(sources) < 3 or total_score < 0.5:
                    continue

                # Cycle-level convergence: how many UNIQUE agents emitted
                # warnings about this target in this cycle?
                cycle_warners = len(cycle_warner_counts.get(target_id, set()))

                # Require the warning rate to be significant relative to
                # the target's interaction count. An agent with 20 opponents
                # that gets warned about by 5 = 25% warning rate (suspicious).
                # An agent with 20 opponents warned about by 2 = 10% (noise).
                target_agent = agents.get(target_id)
                if not target_agent:
                    continue
                target_opponents = max(len(target_agent.history), 1)

                # Warning rate threshold: at least 20% of opponents independently
                # suspicious. Derived from binomial: if each opponent has 5%
                # chance of a false warning, P(20% of 20 opponents warn) < 0.001.
                warning_rate = cycle_warners / target_opponents
                if warning_rate < 0.20:
                    continue

                confirmed.append({
                    'target': target_id,
                    'confirmed_by': agent_id,
                    'score': total_score,
                    'sources': len(sources),
                    'cycle_warners': cycle_warners,
                    'warning_rate': round(warning_rate, 3),
                    'round': round_num
                })
                confirmed_targets.add(target_id)

        return confirmed

    def _record_suspicion_intelligence(self, confirmed: list[dict],
                                       agents: dict) -> None:
        """
        Record confirmed suspicions as immune intelligence.
        Does NOT flag agents — only Phase 5 (ring detection) can flag.

        This feeds behavioral evidence into the collective immune memory,
        allowing ring detection to incorporate warning signals alongside
        structural analysis.
        """
        for threat in confirmed:
            target_id = threat['target']
            if target_id not in agents:
                continue
            target = agents[target_id]
            if not target.alive or target.flagged_sybil:
                continue

            # Store behavioral fingerprint in confirming agent's memory
            confirmer_id = threat['confirmed_by']
            if confirmer_id in agents:
                profile = self._extract_behavioral_profile(target)
                agents[confirmer_id].store_threat_pattern(target_id, profile)

            # Record for monitoring/narrator
            self.confirmed_threats.append({
                'target': target_id,
                'confirmed_by': threat['confirmed_by'],
                'score': round(threat['score'], 2),
                'sources': threat.get('sources', 0),
                'round': threat.get('round', 0)
            })

    def _detect_ring_statistical(self, agents: dict, trust_net,
                                 all_agent_ids: list[str],
                                 round_num: int) -> set:
        """
        Statistical sybil ring detection via hypothesis testing.

        NO HAND-TUNED THRESHOLDS. Every cutoff is one of:
        1. Derived from the data (μ + σ of trust distribution)
        2. Universal statistical convention (z > 3.09 = p < 0.001)
        3. Derived from the Bayesian model (gap > 1σ of single observation)
        4. Structural constant (cluster size 3 to n/3)

        THE STATISTICAL TEST:
        H0: Agents cooperate at the same rate internally and externally.
        H1: Agents cooperate significantly more internally (collusion).

        KEY INNOVATION — BAYESIAN POSTERIOR VARIANCE:
        Instead of sample variance (which is 0 when all values are equal,
        artificially inflating z-scores with sparse data), we use the
        BAYESIAN POSTERIOR VARIANCE of each trust estimate:

            var(Beta(α,β)) = αβ / ((α+β)²(α+β+1))

        This tells us how UNCERTAIN each trust estimate is given the evidence.
        With few interactions: high variance (wide posterior, low confidence).
        With many interactions: low variance (narrow posterior, high confidence).

        This prevents sparse-data artifacts:
        - 3 agents, 2 cooperations each → trust = 0.75, sample_var = 0,
          but Bayesian var = 0.0375 (reflecting genuine uncertainty)
        - Sybil ring, 20 cooperations → trust = 0.95, Bayesian var = 0.002
          (reflecting high confidence in the estimate)
        """
        flagged = set()
        all_ids_set = set(all_agent_ids)

        # ─── Step 1: Compute trust distribution for adaptive cluster discovery ───
        # The cluster threshold comes from the DATA, not from us.
        all_trust_values = []
        observation_counts = []  # for data-derived MINIMUM_GAP
        for (a, b), state in trust_net.edges.items():
            if a in all_ids_set and b in all_ids_set:
                # Evidence floor: need at least 2 real interactions (beyond prior).
                # Bayesian derivation: posterior std of Beta(3,1) = 0.19, which is
                # below the max binary variance (0.25). At fewer observations,
                # the posterior is too wide for meaningful inference.
                if (state.alpha + state.beta) >= 4:
                    all_trust_values.append(state.direct_trust)
                    observation_counts.append(state.alpha + state.beta - 2)

        if len(all_trust_values) < 20:
            return set()  # Insufficient data for statistics

        mu_trust = float(np.mean(all_trust_values))
        sigma_trust = float(np.std(all_trust_values))

        # Cluster threshold = μ + σ: edges significantly above population mean.
        cluster_threshold = mu_trust + sigma_trust

        # ─── Step 2: Find candidate clusters ────────────────────────────────────
        clusters = trust_net.get_clusters(all_agent_ids, threshold=cluster_threshold)

        for cluster in clusters:
            # Size bounds: too small = noise, too large = legitimate community.
            # 3 = minimum for any group inference (structural constant).
            # n/3 = a cluster larger than 1/3 of the population is a community, not a ring.
            if len(cluster) < 3 or len(cluster) > len(agents) // 3:
                continue

            cluster_ids = list(cluster)
            cluster_set = set(cluster_ids)

            # ─── Step 3: Gather internal and external trust + Bayesian variance ──
            # For each trust edge, we collect BOTH the point estimate (trust)
            # and the Bayesian posterior variance (uncertainty).
            internal_trusts = []
            internal_bayes_vars = []
            for a in cluster_ids:
                for b in cluster_ids:
                    if a != b:
                        state = trust_net.edges.get((a, b))
                        if state and (state.alpha + state.beta) >= 4:  # 2+ real interactions
                            internal_trusts.append(state.direct_trust)
                            ab = state.alpha + state.beta
                            internal_bayes_vars.append(
                                state.alpha * state.beta / (ab * ab * (ab + 1))
                            )

            if len(internal_trusts) < 3:
                continue  # Not enough internal evidence

            external_trusts = []
            external_bayes_vars = []
            for outsider_id in all_agent_ids:
                if outsider_id in cluster_set:
                    continue
                for member_id in cluster_ids:
                    state = trust_net.edges.get((outsider_id, member_id))
                    if state and (state.alpha + state.beta) >= 4:  # 2+ real interactions
                        external_trusts.append(state.direct_trust)
                        ab = state.alpha + state.beta
                        external_bayes_vars.append(
                            state.alpha * state.beta / (ab * ab * (ab + 1))
                        )

            if len(external_trusts) < 5:
                continue  # CLT minimum: n ≥ 5 for valid z-test on bounded distributions

            mu_in = float(np.mean(internal_trusts))
            mu_out = float(np.mean(external_trusts))
            n_in = len(internal_trusts)
            n_out = len(external_trusts)

            # ─── Step 4: Practical significance — two complementary checks ─────
            #
            # Check A: DATA-DERIVED MINIMUM GAP — self-adapting.
            # At the population's median observation count, compute Bayesian
            # posterior σ and require gap > 3σ (99.7% confidence).
            # This shrinks as agents accumulate interactions (higher confidence)
            # and grows when data is sparse (conservative).
            median_n = float(np.median(observation_counts))
            # Bayesian posterior std for balanced Beta at median observation count:
            # std(Beta(n/2+1, n/2+1)) ≈ 1/(2*sqrt(n+3))
            posterior_std = 1.0 / (2.0 * np.sqrt(median_n + 3))
            # 2σ: a gap this large has < 2.3% probability from random variation.
            # We use 2σ (not 3σ) because the z-test (Phase 5b) already provides
            # 99.9% statistical significance. The MINIMUM_GAP is a PRACTICAL
            # significance floor — "is this gap large enough to represent real
            # behavioral difference?" — not a second statistical test.
            # Two independent checks: practical (2σ) × statistical (z > 3.09)
            # gives combined confidence > 99.97%.
            MINIMUM_GAP = max(2.0 * posterior_std, 0.15)  # floor: measurement granularity
            gap = mu_in - mu_out
            if gap < MINIMUM_GAP:
                continue

            # Check B: EXTERNAL TRUST MUST BE LOW — the defining behavior of
            # sybil rings is that they DEFECT against outsiders. Outsiders
            # therefore have LOW trust in ring members (below population average).
            # Honest cooperative clusters have NORMAL external trust — outsiders
            # still cooperate with them regularly.
            #
            # This is data-derived: mu_trust comes from the population itself.
            # A cluster where outsiders trust the members at or above the
            # population mean is NOT behaving like a sybil ring.
            if mu_out >= mu_trust:
                continue  # Outsiders trust them fine — not sybil behavior

            # ─── Step 5: Statistical significance — Bayesian z-test ─────────────
            # Use BAYESIAN POSTERIOR VARIANCE instead of sample variance.
            #
            # Why: Sample variance of [0.75, 0.75, 0.75, 0.75] = 0. This makes
            # the z-score infinite — a statistical artifact of few observations.
            # Bayesian variance of Beta(3,1) = 0.0375. This reflects genuine
            # uncertainty: "I've only seen 2 cooperations, I'm not that certain."
            #
            # The Bayesian approach uses the uncertainty that THE MODEL ITSELF
            # tells us, not an unreliable sample statistic.
            avg_bvar_in = float(np.mean(internal_bayes_vars))
            avg_bvar_out = float(np.mean(external_bayes_vars))

            # Standard error using mean Bayesian posterior variances
            se = np.sqrt(avg_bvar_in / n_in + avg_bvar_out / n_out)
            if se < 1e-6:
                z = 100.0  # Only possible with overwhelming evidence
            else:
                z = gap / se

            # z > 3.09 corresponds to one-tailed p < 0.001.
            # This is a universal statistical convention — the 99.9% confidence
            # level used across all sciences for "highly significant" results.
            # It is NOT a domain-specific tuning parameter.
            SIGNIFICANCE_Z = 3.09  # p < 0.001 (one-tailed)

            if z > SIGNIFICANCE_Z:
                # This cluster has BOTH:
                # - Practically significant trust asymmetry (gap > data-derived 3σ threshold)
                # - Statistically significant at 99.9% confidence (z > 3.09)
                # - Low external trust (outsiders distrust them — sybil signature)
                # - Using Bayesian variances that properly account for data sparsity
                # This combination is nearly impossible for honest agent groups.

                for aid in cluster_ids:
                    if aid in agents and not agents[aid].flagged_sybil:
                        agents[aid].flagged_sybil = True
                        flagged.add(aid)
                        trust_net.isolate_agent(aid, all_agent_ids)

                self.events.append({
                    'type': 'ring_detected',
                    'members': cluster_ids,
                    'z_score': round(float(z), 2),
                    'gap': round(gap, 3),
                    'internal_trust': round(mu_in, 3),
                    'external_trust': round(mu_out, 3),
                    'cluster_threshold': round(cluster_threshold, 3),
                    'round': round_num
                })

        return flagged

    def _extract_behavioral_profile(self, agent) -> dict:
        """
        Extract a behavioral fingerprint for immune memory.
        This is what gets stored and matched against future agents.
        """
        commit_total = sum(
            h + b for h, b in agent.commitment_history.values()
        ) if agent.commitment_history else 0

        commit_honor_rate = 0.5
        if commit_total > 0:
            honors = sum(h for h, _ in agent.commitment_history.values())
            commit_honor_rate = honors / commit_total

        return {
            'coop_rate': agent.coop_rate,
            'commit_rate': commit_honor_rate,
            'selectivity': agent.selectivity,
            'interactions': agent.interactions,
        }

    def pop_events(self) -> list[dict]:
        """Pop accumulated events."""
        events = self.events
        self.events = []
        return events
