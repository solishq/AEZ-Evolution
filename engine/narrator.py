"""
AEZ Evolution — Real-Time Narrator

Copyright (c) 2026 SolisHQ (github.com/solishq). All rights reserved.
Licensed under MIT. Built for Colosseum Hackathon 2026.

The narrative writes itself from what actually happens.
No scripted acts. No fixed text. Every simulation tells a different story.

V2: Agent story arcs — heroes, villains, underdogs.
Every agent has a name, a journey, a fate.
"""

from typing import Optional
import random

# Agent name generation — memorable names for memorable stories
FIRST_NAMES = [
    'Atlas', 'Nova', 'Cipher', 'Echo', 'Forge', 'Pulse', 'Drift', 'Flux',
    'Helix', 'Iris', 'Jade', 'Koda', 'Lux', 'Moss', 'Nyx', 'Onyx',
    'Prism', 'Quill', 'Rune', 'Sage', 'Thorn', 'Umbra', 'Vex', 'Wren',
    'Xenon', 'Yara', 'Zephyr', 'Blaze', 'Crest', 'Dusk', 'Ember', 'Fable',
    'Ghost', 'Haze', 'Ion', 'Jinx', 'Karma', 'Loop', 'Mist', 'Nexus',
    'Orbit', 'Pike', 'Quirk', 'Ridge', 'Shade', 'Tide', 'Unity', 'Volt',
    'Solis', 'Ferro',
]


class AgentArc:
    """Track an agent's journey through the simulation."""
    __slots__ = ['name', 'role', 'peak_fitness', 'lowest_fitness',
                 'betrayals_received', 'betrayals_given', 'survived_attacks',
                 'times_on_leaderboard', 'flagged_round']

    def __init__(self, name: str):
        self.name = name
        self.role = 'unknown'  # hero, villain, underdog, survivor, fallen
        self.peak_fitness = 0
        self.lowest_fitness = 0
        self.betrayals_received = 0
        self.betrayals_given = 0
        self.survived_attacks = 0
        self.times_on_leaderboard = 0
        self.flagged_round = None


class Narrator:
    """Generates real-time commentary from simulation events."""

    def __init__(self):
        self.last_narration = ""
        self.major_events = []

        # Agent story tracking
        self.arcs: dict[str, AgentArc] = {}
        self._name_pool = list(FIRST_NAMES)
        random.shuffle(self._name_pool)
        self._name_idx = 0

    def get_name(self, agent_id: str) -> str:
        """Get or assign a memorable name for an agent."""
        if agent_id not in self.arcs:
            if self._name_idx < len(self._name_pool):
                name = self._name_pool[self._name_idx]
                self._name_idx += 1
            else:
                name = f"Agent-{agent_id}"
            self.arcs[agent_id] = AgentArc(name)
        return self.arcs[agent_id].name

    def _agent_label(self, agent_id: str) -> str:
        """Format: 'Name (A0012)' for readability."""
        name = self.get_name(agent_id)
        return f"{name} ({agent_id})"

    def track_leaderboard(self, leaderboard: list[dict]):
        """Update arcs based on current leaderboard."""
        for entry in leaderboard[:5]:
            aid = entry.get('id', '')
            self.get_name(aid)  # Ensure named
            arc = self.arcs[aid]
            arc.times_on_leaderboard += 1
            fitness = entry.get('fitness', 0)
            if fitness > arc.peak_fitness:
                arc.peak_fitness = fitness

    def narrate(self, round_num: int, events: list[dict], stats: dict) -> Optional[dict]:
        """
        Generate narration from events and stats.
        Returns {title, text, severity} or None if nothing interesting.
        """
        # Check for dramatic events first — prioritize detection events
        best_narration = None
        for event in events:
            narration = self._narrate_event(event, round_num, stats)
            if narration:
                self.major_events.append(narration)
                # Detection and critical events take priority
                if narration.get('severity') == 'critical':
                    return narration
                if best_narration is None:
                    best_narration = narration

        if best_narration:
            return best_narration

        # Periodic narration with agent stories
        if round_num % 10 == 0:
            return self._narrate_state(round_num, stats)

        # Agent arc updates every 15 rounds
        if round_num % 15 == 0:
            return self._narrate_arc(round_num, stats)

        return None

    def _narrate_event(self, event: dict, round_num: int, stats: dict) -> Optional[dict]:
        etype = event.get('type', '')

        if etype == 'betrayal':
            betrayer = self._agent_label(event['dst'])
            victim = self._agent_label(event['src'])
            # Track arcs
            self.arcs[event['dst']].betrayals_given += 1
            self.arcs[event['src']].betrayals_received += 1
            return {
                'title': 'Betrayal',
                'text': f"{betrayer} stabbed {victim} in the back. "
                       f"Trust crashed from {event['old_trust']:.0%} to {event['new_trust']:.0%}. "
                       f"The network remembers.",
                'severity': 'high',
                'icon': '🗡️'
            }

        if etype == 'cascade_collapse':
            betrayer = self._agent_label(event['betrayer'])
            victim = self._agent_label(event['victim'])
            return {
                'title': 'Trust Cascade Collapse',
                'text': f"{betrayer}'s betrayal of {victim} "
                       f"sent shockwaves through the network. "
                       f"{event['affected']} agents updated their trust models.",
                'severity': 'critical',
                'icon': '💥'
            }

        if etype == 'attack_sybil':
            ids = event.get('agents', [])
            names = [self.get_name(aid) for aid in ids[:3]]
            return {
                'title': 'Sybil Attack Launched',
                'text': f"{event['count']} colluding agents infiltrated the network: "
                       f"{', '.join(names)}... "
                       f"They cooperate with each other, exploit everyone else. "
                       f"Can the trust system identify them?",
                'severity': 'critical',
                'icon': '🕷️'
            }

        if etype == 'sybil_detected':
            ring = event.get('ring', [])
            names = [self._agent_label(aid) for aid in ring[:3]]
            more = f" and {len(ring)-3} more" if len(ring) > 3 else ""
            collapsed = event.get('trust_collapsed', 0)
            # Mark arcs
            for aid in ring:
                if aid in self.arcs:
                    self.arcs[aid].role = 'villain'
                    self.arcs[aid].flagged_round = round_num
            return {
                'title': 'SYBIL RING DETECTED',
                'text': f"The trust network identified {len(ring)} colluding agents: "
                       f"{', '.join(names)}{more}. "
                       f"{collapsed} trust edges collapsed. "
                       f"They're isolated. The network fought back.",
                'severity': 'critical',
                'icon': '🔍'
            }

        if etype == 'trojan_activated':
            names = [self._agent_label(aid) for aid in event['agents'][:3]]
            return {
                'title': 'Trojan Agents Activated',
                'text': f"{len(event['agents'])} trusted agents just revealed themselves: "
                       f"{', '.join(names)}. "
                       f"Every relationship they built was a lie.",
                'severity': 'critical',
                'icon': '🎭'
            }

        if etype == 'attack_eclipse':
            target = self._agent_label(event['target'])
            return {
                'title': 'Eclipse Attack',
                'text': f"{target} is being isolated. "
                       f"{len(event['attackers'])} hostile agents are surrounding it, "
                       f"cutting it off from allies.",
                'severity': 'high',
                'icon': '🌑'
            }

        if etype == 'birth':
            child = self._agent_label(event['agent'])
            return {
                'title': 'New Generation',
                'text': f"{child} born — generation {event['generation']}. "
                       f"New neural weights, new strategies. Evolution continues.",
                'severity': 'low',
                'icon': '🌱'
            }

        if etype == 'selection_death':
            dead = self._agent_label(event['agent'])
            arc = self.arcs.get(event['agent'])
            if arc and arc.role == 'villain':
                return {
                    'title': 'Justice',
                    'text': f"{dead} eliminated. Detected as sybil, isolated by the network, "
                           f"and finally purged. The system works.",
                    'severity': 'medium',
                    'icon': '⚖️'
                }
            if event.get('fitness', 0) < -500:
                return {
                    'title': 'Extinction',
                    'text': f"{dead} ({event['strategy']}) eliminated with fitness "
                           f"{event['fitness']:.0f}. The weak don't survive here.",
                    'severity': 'medium',
                    'icon': '💀'
                }

        if etype == 'payoff_change':
            return {
                'title': 'Economic Shift',
                'text': f"Payoff '{event['key']}' changed from {event['old']} to {event['new']}. "
                       f"The rules just changed. Adapt or die.",
                'severity': 'high',
                'icon': '📉'
            }

        return None

    def _narrate_state(self, round_num: int, stats: dict) -> dict:
        """Generate periodic state narration."""
        alive = stats.get('alive', 0)
        coop_rate = stats.get('coop_rate', 0)
        clusters = stats.get('clusters', 0)

        if round_num <= 10:
            return {
                'title': 'Genesis',
                'text': f"Round {round_num}. {alive} agents with random neural weights. "
                       f"No strategies yet — just noise finding signal. "
                       f"Cooperation rate: {coop_rate:.0%}.",
                'severity': 'info',
                'icon': '🌅'
            }

        if coop_rate > 0.75:
            return {
                'title': 'Cooperation Dominates',
                'text': f"Round {round_num}. {coop_rate:.0%} cooperation rate. "
                       f"{clusters} trust clusters formed. Cooperators found each other "
                       f"and built something stable. For now.",
                'severity': 'info',
                'icon': '🤝'
            }

        if coop_rate < 0.3:
            return {
                'title': 'Chaos Reigns',
                'text': f"Round {round_num}. Only {coop_rate:.0%} cooperation. "
                       f"Defectors are thriving. Trust is collapsing. "
                       f"But evolution favors cooperation — it will return.",
                'severity': 'medium',
                'icon': '🔥'
            }

        if clusters >= 3:
            return {
                'title': 'Factions Form',
                'text': f"Round {round_num}. {clusters} distinct trust clusters. "
                       f"The network has fractured into competing alliances. "
                       f"Each faction trusts its own and doubts the rest.",
                'severity': 'info',
                'icon': '🏰'
            }

        return {
            'title': f'Round {round_num}',
            'text': f"{alive} agents alive. {coop_rate:.0%} cooperation. "
                   f"{clusters} clusters. The battle for trust continues.",
            'severity': 'info',
            'icon': '⚡'
        }

    def _narrate_arc(self, round_num: int, stats: dict) -> Optional[dict]:
        """Generate narration about individual agent stories."""
        if not self.arcs:
            return None

        # Find the hero — highest leaderboard appearances
        heroes = sorted(
            [(aid, arc) for aid, arc in self.arcs.items() if arc.times_on_leaderboard > 0],
            key=lambda x: x[1].times_on_leaderboard, reverse=True
        )

        if heroes:
            hero_id, hero_arc = heroes[0]
            if hero_arc.times_on_leaderboard >= 3:
                hero_arc.role = 'hero'
                return {
                    'title': f'{hero_arc.name} — The Survivor',
                    'text': f"{hero_arc.name} ({hero_id}) has been on the leaderboard "
                           f"{hero_arc.times_on_leaderboard} times. Peak fitness: "
                           f"{hero_arc.peak_fitness:.0f}. "
                           f"{'Betrayed ' + str(hero_arc.betrayals_received) + ' times but still standing.' if hero_arc.betrayals_received > 0 else 'Built trust, earned loyalty, thrived.'}",
                    'severity': 'info',
                    'icon': '👑'
                }

        # Find an underdog — agent with high recent performance but many betrayals
        underdogs = [
            (aid, arc) for aid, arc in self.arcs.items()
            if arc.betrayals_received >= 2 and arc.times_on_leaderboard > 0
        ]
        if underdogs:
            ud_id, ud_arc = underdogs[0]
            ud_arc.role = 'underdog'
            return {
                'title': f'{ud_arc.name} — Against the Odds',
                'text': f"{ud_arc.name} ({ud_id}) was betrayed {ud_arc.betrayals_received} times "
                       f"but keeps climbing back. Currently on the leaderboard. "
                       f"Resilience in the face of deception.",
                'severity': 'info',
                'icon': '💪'
            }

        return None

    def get_agent_names(self) -> dict:
        """Return all agent name mappings for visualization."""
        return {aid: arc.name for aid, arc in self.arcs.items()}

    def get_summary(self, stats: dict) -> dict:
        """Generate final summary with agent stories."""
        # Find notable agents
        heroes = [arc for arc in self.arcs.values() if arc.role == 'hero']
        villains = [arc for arc in self.arcs.values() if arc.role == 'villain']

        hero_text = f" {heroes[0].name} dominated the leaderboard." if heroes else ""
        villain_text = f" {len(villains)} sybils detected and isolated." if villains else ""

        return {
            'title': 'Simulation Complete',
            'text': f"After {stats.get('round', 0)} rounds, "
                   f"{stats.get('alive', 0)} agents survive. "
                   f"Final cooperation rate: {stats.get('coop_rate', 0):.0%}. "
                   f"{len(self.major_events)} major events.{hero_text}{villain_text}",
            'severity': 'info',
            'icon': '🏆',
            'major_events': self.major_events[-10:]
        }
