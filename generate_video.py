#!/usr/bin/env python3
"""
AEZ Evolution — Programmatic Pitch Video Generator

Generates a 3-minute 1080p hackathon pitch video using:
- pycairo for frame rendering (vector quality)
- The AEZ simulation engine for live data
- FFmpeg for H.264 MP4 encoding

No screen recording. No external tools. Pure code.
"""

import sys, os, math, subprocess, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cairo
import numpy as np
from engine.evolution import Evolution, Attacks
from engine.narrator import Narrator

# ─── Config ────────────────────────────────────────────────
W, H = 1920, 1080
FPS = 30
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pitch-video.mp4')

# Colors
BG_TOP = (0.04, 0.04, 0.10)
BG_BOT = (0.10, 0.04, 0.18)
WHITE = (1, 1, 1)
DIM = (0.5, 0.5, 0.6)
GREEN = (0, 1, 0.53)
RED = (1, 0.27, 0.27)
BLUE = (0.27, 0.53, 1)
PURPLE = (0.67, 0.27, 1)
CYAN = (0, 0.85, 1)
GOLD = (1, 0.84, 0)

STRATEGY_COLORS = {
    'Cooperator': GREEN,
    'Defector': RED,
    'Reciprocator': BLUE,
    'Mixed': (0.7, 0.7, 0.3),
    'Sybil': PURPLE,
}

def strategy_color(agent_dict):
    if agent_dict.get('flagged_sybil'):
        return PURPLE
    label = agent_dict.get('strategy', 'Mixed')
    return STRATEGY_COLORS.get(label, DIM)


# ─── Drawing Helpers ───────────────────────────────────────

def gradient_bg(ctx):
    """Dark gradient background."""
    pat = cairo.LinearGradient(0, 0, 0, H)
    pat.add_color_stop_rgb(0, *BG_TOP)
    pat.add_color_stop_rgb(1, *BG_BOT)
    ctx.set_source(pat)
    ctx.paint()

def draw_text(ctx, x, y, text, size=24, color=WHITE, bold=False, align='left', max_width=None):
    """Draw text with alignment support."""
    weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
    ctx.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size)
    ctx.set_source_rgb(*color)

    if max_width and len(text) > 0:
        ext = ctx.text_extents(text)
        while ext.width > max_width and size > 10:
            size -= 1
            ctx.set_font_size(size)
            ext = ctx.text_extents(text)

    ext = ctx.text_extents(text)
    if align == 'center':
        x = x - ext.width / 2
    elif align == 'right':
        x = x - ext.width

    ctx.move_to(x, y)
    ctx.show_text(text)

def draw_mono(ctx, x, y, text, size=20, color=DIM):
    """Draw monospace text."""
    ctx.select_font_face('monospace', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    ctx.set_source_rgb(*color)
    ctx.move_to(x, y)
    ctx.show_text(text)

def draw_bar(ctx, x, y, w, h, pct, color=GREEN):
    """Draw a progress bar."""
    # Background
    ctx.set_source_rgba(1, 1, 1, 0.08)
    ctx.rectangle(x, y, w, h)
    ctx.fill()
    # Fill
    ctx.set_source_rgb(*color)
    ctx.rectangle(x, y, w * min(pct, 1.0), h)
    ctx.fill()

def draw_glow_circle(ctx, cx, cy, r, color, glow=False):
    """Draw a circle with optional glow."""
    if glow:
        for i in range(3):
            gr = r + (3 - i) * 4
            ctx.arc(cx, cy, gr, 0, 2 * math.pi)
            ctx.set_source_rgba(*color, 0.1 + i * 0.05)
            ctx.fill()
    ctx.arc(cx, cy, r, 0, 2 * math.pi)
    ctx.set_source_rgb(*color)
    ctx.fill()

def ease_in_out(t):
    """Smooth easing function."""
    return t * t * (3 - 2 * t)

def fade_alpha(frame, start, duration):
    """Calculate fade-in alpha for a frame range."""
    if frame < start:
        return 0
    t = min((frame - start) / max(duration, 1), 1.0)
    return ease_in_out(t)


# ─── Force-Directed Layout ────────────────────────────────

class ForceLayout:
    """Simple force-directed graph layout using NumPy."""

    def __init__(self):
        self.positions = {}  # agent_id -> (x, y) in [0, 1] space

    def update(self, nodes, edges, iterations=30):
        """Update positions based on current graph structure."""
        if not nodes:
            return

        ids = [n['id'] for n in nodes]
        n = len(ids)
        id_to_idx = {aid: i for i, aid in enumerate(ids)}

        # Initialize new nodes at random positions
        pos = np.zeros((n, 2))
        for i, aid in enumerate(ids):
            if aid in self.positions:
                pos[i] = self.positions[aid]
            else:
                pos[i] = np.random.rand(2) * 0.8 + 0.1

        # Build adjacency
        adj = []
        for e in edges:
            src, dst = e.get('source'), e.get('target')
            if src in id_to_idx and dst in id_to_idx:
                adj.append((id_to_idx[src], id_to_idx[dst], e.get('weight', 0.5)))

        # Force simulation
        k = 0.15  # Ideal spring length
        for _ in range(iterations):
            # Repulsion (all pairs)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 2)
            dist = np.sqrt((diff ** 2).sum(axis=2) + 1e-6)  # (n, n)
            repulsion = diff / (dist[:, :, np.newaxis] ** 2 + 0.01) * k * k * 0.5
            force = repulsion.sum(axis=1)  # (n, 2)

            # Attraction (edges only)
            for i, j, w in adj:
                d = pos[j] - pos[i]
                dist_ij = np.sqrt(d @ d + 1e-6)
                f = d * dist_ij / k * w * 0.3
                force[i] += f
                force[j] -= f

            # Center gravity
            center = pos.mean(axis=0)
            force -= (pos - center) * 0.01

            # Apply with damping
            pos += np.clip(force * 0.02, -0.05, 0.05)
            pos = np.clip(pos, 0.02, 0.98)

        # Store
        for i, aid in enumerate(ids):
            self.positions[aid] = pos[i].copy()

        # Remove dead agents
        alive_set = set(ids)
        for aid in list(self.positions):
            if aid not in alive_set:
                del self.positions[aid]


# ─── Simulation Runner ─────────────────────────────────────

def run_simulation():
    """Run the full 100-round simulation and collect all data."""
    print("  Running simulation...")

    evo = Evolution(population_size=50)
    evo.spawn_population()
    narrator = Narrator()
    layout = ForceLayout()

    frames_data = []  # One entry per round
    sybil_ids = set()
    trojan_ids = set()

    def collect_round(phase_label, phase_event=None):
        """Collect data for current round."""
        stats = evo.round_stats[-1] if evo.round_stats else {}
        network = evo.get_network_data()
        events = evo.pop_events()

        narrator.track_leaderboard(evo.get_leaderboard(5))
        narration = narrator.narrate(evo.round, events, stats)

        # Update layout
        layout.update(network['nodes'], network['edges'], iterations=15)
        positions = {aid: tuple(pos) for aid, pos in layout.positions.items()}

        leaderboard = evo.get_leaderboard(5)
        dist = evo.get_strategy_distribution()

        frames_data.append({
            'round': evo.round,
            'generation': evo.generation,
            'phase': phase_label,
            'phase_event': phase_event,
            'stats': stats,
            'nodes': network['nodes'],
            'edges': network['edges'],
            'positions': positions,
            'leaderboard': leaderboard,
            'strategy_dist': dist,
            'narration': narration,
            'narrator': narrator,
            'sybil_ids': set(sybil_ids),
            'trojan_ids': set(trojan_ids),
        })

    # Phase 1: Chaos (R1-25)
    for _ in range(25):
        evo.run_round()
        Attacks.activate_trojans(evo)
        if evo.round % 20 == 0:
            evo.run_selection()
        collect_round('CHAOS')

    # Phase 2: Sybil Attack (R26-50)
    sybil_list = Attacks.sybil_attack(evo, 10)
    sybil_ids.update(sybil_list)
    collect_round('SYBIL', 'SYBIL_INJECT')

    for _ in range(24):
        evo.run_round()
        Attacks.activate_trojans(evo)
        if evo.round % 20 == 0:
            evo.run_selection()
        collect_round('SYBIL')

    # Phase 3: Trojan + Eclipse (R51-75)
    trojan_list = Attacks.trojan_attack(evo, 5, betray_round=65)
    trojan_ids.update(trojan_list)

    leader = evo.get_leaderboard(1)
    if leader:
        Attacks.eclipse_attack(evo, leader[0]['id'], 6)

    collect_round('ATTACK', 'TROJAN_PLANT')

    for _ in range(24):
        evo.run_round()
        Attacks.activate_trojans(evo)
        if evo.round % 20 == 0:
            evo.run_selection()
        collect_round('ATTACK')

    # Phase 4: Equilibrium (R76-100)
    evo.set_payoff('partners', 'CC', 600)
    evo.set_payoff('strangers', 'DC', 300)
    collect_round('EQUILIBRIUM', 'ECON_SHIFT')

    for _ in range(24):
        evo.run_round()
        Attacks.activate_trojans(evo)
        if evo.round % 20 == 0:
            evo.run_selection()
        collect_round('EQUILIBRIUM')

    # Final sybil report
    real_sybils_alive = [a for a in evo.get_alive() if a.id in sybil_ids]
    caught = sum(1 for a in real_sybils_alive if a.flagged_sybil)
    false_pos = sum(1 for a in evo.get_alive() if a.flagged_sybil and a.id not in sybil_ids)

    summary = {
        'total_rounds': evo.round,
        'final_alive': len(evo.get_alive()),
        'final_coop_rate': evo.round_stats[-1].get('coop_rate', 0) if evo.round_stats else 0,
        'sybils_caught': caught,
        'sybils_total': len(sybil_ids),
        'false_positives': false_pos,
        'heroes': [(aid, arc) for aid, arc in narrator.arcs.items() if arc.role == 'hero'][:3],
        'villains': [(aid, arc) for aid, arc in narrator.arcs.items() if arc.role == 'villain'][:3],
    }

    print(f"  Simulation complete: {len(frames_data)} rounds collected")
    return frames_data, summary, narrator


# ─── Frame Renderers ───────────────────────────────────────

def render_title_card(ctx, frame, total_frames):
    """Title card with fade-in."""
    gradient_bg(ctx)

    alpha = fade_alpha(frame, 0, 30)

    # Main title
    draw_text(ctx, W/2, H/2 - 80, "AEZ EVOLUTION", size=72, bold=True,
              color=(alpha, alpha, alpha), align='center')

    # Subtitle
    a2 = fade_alpha(frame, 20, 30)
    draw_text(ctx, W/2, H/2 + 0, "Trust Infrastructure for AI Agents", size=36,
              color=(a2 * 0.7, a2 * 0.85, a2), align='center')

    # Tagline
    a3 = fade_alpha(frame, 45, 30)
    draw_text(ctx, W/2, H/2 + 60, "Colosseum Agent Hackathon 2026", size=24,
              color=(a3 * 0.5, a3 * 0.5, a3 * 0.6), align='center')

    # Accent line
    a4 = fade_alpha(frame, 60, 40)
    line_w = 300 * a4
    ctx.set_source_rgba(*GREEN, a4 * 0.8)
    ctx.set_line_width(2)
    ctx.move_to(W/2 - line_w/2, H/2 + 90)
    ctx.line_to(W/2 + line_w/2, H/2 + 90)
    ctx.stroke()


def render_problem_slide(ctx, frame, total_frames):
    """The problem: which agents can you trust?"""
    gradient_bg(ctx)

    a1 = fade_alpha(frame, 0, 25)
    draw_text(ctx, W/2, 280, "As AI agents proliferate:", size=32,
              color=(a1*0.6, a1*0.6, a1*0.7), align='center')
    draw_text(ctx, W/2, 360, '"Which agents can I trust?"', size=56, bold=True,
              color=(a1, a1, a1), align='center')

    problems = [
        ("Centralized reputation", "Single point of failure", RED),
        ("Token incentives", "Financial games, not trust", RED),
        ("Hard-coded rules", "No learning, no adaptation", RED),
    ]

    for i, (title, desc, color) in enumerate(problems):
        a = fade_alpha(frame, 40 + i * 30, 25)
        y = 480 + i * 80
        # X mark
        draw_text(ctx, 500, y, "X", size=32, bold=True, color=(a*color[0], a*color[1], a*color[2]))
        draw_text(ctx, 550, y, title, size=30, bold=True, color=(a, a, a))
        draw_text(ctx, 550, y + 32, desc, size=20, color=(a*0.5, a*0.5, a*0.6))


def render_solution_slide(ctx, frame, total_frames):
    """Our solution: trust that emerges from behavior."""
    gradient_bg(ctx)

    a1 = fade_alpha(frame, 0, 25)
    draw_text(ctx, W/2, 250, "Trust that emerges from behavior", size=48, bold=True,
              color=(a1, a1, a1), align='center')

    solutions = [
        ("4-Channel Bayesian Trust", "Direct + Social + Temporal + Structural", GREEN),
        ("Decentralized Immune System", "5-phase detection, zero false positives", CYAN),
        ("Evolutionary Discovery", "Neural agents evolve cooperation naturally", GOLD),
    ]

    for i, (title, desc, color) in enumerate(solutions):
        a = fade_alpha(frame, 30 + i * 35, 25)
        y = 400 + i * 100
        draw_glow_circle(ctx, 470, y - 8, 8, tuple(c * a for c in color))
        draw_text(ctx, 500, y, title, size=30, bold=True, color=(a, a, a))
        draw_text(ctx, 500, y + 34, desc, size=20, color=(a*0.5, a*0.5, a*0.6))

    a2 = fade_alpha(frame, 140, 30)
    draw_text(ctx, W/2, 780, "All verifiable on Solana", size=28,
              color=(a2*0.4, a2*0.7, a2), align='center')


def render_simulation_frame(ctx, data, frame_in_section, total_section_frames, coop_history):
    """Render a simulation visualization frame."""
    gradient_bg(ctx)

    rnd = data['round']
    phase = data['phase']
    stats = data['stats']

    # ─── Header Bar ────────────────────────────────────────
    ctx.set_source_rgba(0, 0, 0, 0.4)
    ctx.rectangle(0, 0, W, 60)
    ctx.fill()

    phase_colors = {'CHAOS': DIM, 'SYBIL': PURPLE, 'ATTACK': RED, 'EQUILIBRIUM': GREEN}
    pc = phase_colors.get(phase, DIM)

    draw_text(ctx, 30, 42, "AEZ EVOLUTION", size=24, bold=True, color=WHITE)
    draw_text(ctx, 300, 42, phase, size=20, bold=True, color=pc)
    draw_text(ctx, W - 30, 42, f"R{rnd}  |  Gen {data['generation']}", size=20,
              color=DIM, align='right')

    # Phase event banners
    if data.get('phase_event'):
        event_map = {
            'SYBIL_INJECT': ("SYBIL ATTACK: 10 colluding agents injected!", RED),
            'TROJAN_PLANT': ("TROJAN + ECLIPSE: Sleeper agents planted", PURPLE),
            'ECON_SHIFT': ("ECONOMIC SHIFT: Cooperation now dominates", GREEN),
        }
        if data['phase_event'] in event_map:
            msg, col = event_map[data['phase_event']]
            a = fade_alpha(frame_in_section, 0, 15)
            ctx.set_source_rgba(*col, a * 0.3)
            ctx.rectangle(0, 60, W, 50)
            ctx.fill()
            draw_text(ctx, W/2, 92, msg, size=22, bold=True,
                      color=tuple(c * a for c in col), align='center')

    # ─── Network Graph (left side) ─────────────────────────
    graph_x, graph_y = 40, 80
    graph_w, graph_h = 1280, 860

    # Graph background
    ctx.set_source_rgba(0, 0, 0, 0.2)
    ctx.rectangle(graph_x, graph_y, graph_w, graph_h)
    ctx.fill()

    positions = data['positions']
    nodes = data['nodes']
    edges = data['edges']

    # Draw edges
    for e in edges:
        src_pos = positions.get(e.get('source'))
        dst_pos = positions.get(e.get('target'))
        if src_pos and dst_pos:
            weight = e.get('weight', 0.3)
            sx = graph_x + src_pos[0] * graph_w
            sy = graph_y + src_pos[1] * graph_h
            dx = graph_x + dst_pos[0] * graph_w
            dy = graph_y + dst_pos[1] * graph_h
            ctx.set_source_rgba(1, 1, 1, min(weight * 0.3, 0.25))
            ctx.set_line_width(max(weight * 2, 0.5))
            ctx.move_to(sx, sy)
            ctx.line_to(dx, dy)
            ctx.stroke()

    # Draw nodes
    for node in nodes:
        pos = positions.get(node['id'])
        if not pos:
            continue
        nx = graph_x + pos[0] * graph_w
        ny = graph_y + pos[1] * graph_h

        color = strategy_color(node)
        fitness = node.get('fitness', 0)
        radius = max(4, min(12, 4 + fitness / 500))
        is_sybil = node.get('flagged_sybil', False)

        draw_glow_circle(ctx, nx, ny, radius, color, glow=is_sybil)

    # ─── Stats Panel (right side) ──────────────────────────
    panel_x = 1360
    panel_w = W - panel_x - 20

    ctx.set_source_rgba(0, 0, 0, 0.3)
    ctx.rectangle(panel_x, 80, panel_w, 860)
    ctx.fill()

    # Stats
    sy = 120
    stat_items = [
        ("Alive", f"{stats.get('alive', 0)}", WHITE),
        ("Cooperation", f"{stats.get('coop_rate', 0):.0%}", GREEN if stats.get('coop_rate', 0) > 0.5 else RED),
        ("Trust Edges", f"{stats.get('trust_edges', 0)}", CYAN),
        ("Clusters", f"{stats.get('clusters', 0)}", BLUE),
        ("Sybils Flagged", f"{stats.get('flagged_sybils', 0)}", PURPLE),
    ]

    for label, value, color in stat_items:
        draw_text(ctx, panel_x + 20, sy, label, size=16, color=DIM)
        draw_text(ctx, panel_x + panel_w - 20, sy, value, size=22, bold=True,
                  color=color, align='right')
        sy += 45

    # ─── Cooperation rate chart ────────────────────────────
    sy += 10
    draw_text(ctx, panel_x + 20, sy, "Cooperation Rate", size=16, color=DIM)
    sy += 15

    chart_h = 80
    chart_w = panel_w - 40
    ctx.set_source_rgba(0, 0, 0, 0.3)
    ctx.rectangle(panel_x + 20, sy, chart_w, chart_h)
    ctx.fill()

    # Draw coop history line
    if len(coop_history) > 1:
        ctx.set_source_rgba(*GREEN, 0.8)
        ctx.set_line_width(2)
        points = coop_history[-50:]  # Last 50 rounds
        for i, val in enumerate(points):
            px = panel_x + 20 + (i / max(len(points) - 1, 1)) * chart_w
            py = sy + chart_h - val * chart_h
            if i == 0:
                ctx.move_to(px, py)
            else:
                ctx.line_to(px, py)
        ctx.stroke()

    # 50% line
    ctx.set_source_rgba(1, 1, 1, 0.1)
    ctx.set_line_width(1)
    ctx.move_to(panel_x + 20, sy + chart_h / 2)
    ctx.line_to(panel_x + 20 + chart_w, sy + chart_h / 2)
    ctx.stroke()

    sy += chart_h + 30

    # ─── Strategy Distribution ─────────────────────────────
    draw_text(ctx, panel_x + 20, sy, "Strategy Distribution", size=16, color=DIM)
    sy += 20

    dist = data['strategy_dist']
    total = sum(dist.values()) or 1
    for strat, count in sorted(dist.items(), key=lambda x: -x[1])[:5]:
        pct = count / total
        color = STRATEGY_COLORS.get(strat, DIM)
        draw_text(ctx, panel_x + 20, sy + 16, f"{strat}", size=14, color=color)
        draw_bar(ctx, panel_x + 150, sy + 4, chart_w - 190, 16, pct, color)
        draw_text(ctx, panel_x + chart_w + 10, sy + 16, f"{pct:.0%}", size=14, color=DIM)
        sy += 28

    sy += 20

    # ─── Leaderboard ───────────────────────────────────────
    draw_text(ctx, panel_x + 20, sy, "Leaderboard", size=16, color=DIM)
    sy += 25

    narrator = data['narrator']
    for entry in data['leaderboard'][:5]:
        name = narrator.get_name(entry['id'])
        color = strategy_color(entry)
        sybil_tag = " [SYBIL]" if entry.get('flagged_sybil') else ""
        draw_text(ctx, panel_x + 20, sy, f"#{entry['rank']}", size=14, bold=True, color=GOLD)
        draw_text(ctx, panel_x + 55, sy, f"{name}{sybil_tag}", size=14, color=color,
                  max_width=chart_w - 120)
        draw_text(ctx, panel_x + chart_w + 10, sy, f"{entry['fitness']:.0f}", size=14,
                  color=DIM, align='right')
        sy += 24

    # ─── Event Ticker (bottom) ─────────────────────────────
    narration = data.get('narration')
    if narration:
        ctx.set_source_rgba(0, 0, 0, 0.5)
        ctx.rectangle(0, H - 55, W, 55)
        ctx.fill()

        icon = narration.get('icon', '')
        text = f"{icon} {narration['title']}: {narration['text']}"
        draw_text(ctx, 30, H - 20, text[:120], size=18, color=DIM, max_width=W - 60)


def render_results_slide(ctx, frame, total_frames, summary):
    """Results and metrics slide."""
    gradient_bg(ctx)

    a1 = fade_alpha(frame, 0, 25)
    draw_text(ctx, W/2, 200, "Results", size=56, bold=True,
              color=(a1, a1, a1), align='center')

    metrics = [
        ("Test Suite", "149 tests, 0 failures"),
        ("False Positives", "0 across all scenarios"),
        ("Sybil Detection", f"{summary['sybils_caught']}/{summary['sybils_total']} caught in ~10 rounds"),
        ("Final Cooperation", f"{summary['final_coop_rate']:.0%}"),
        ("Code", "~4,670 lines (Python + Rust + JavaScript)"),
        ("On-Chain", "Solana Devnet — Anchor smart contract"),
    ]

    for i, (label, value) in enumerate(metrics):
        a = fade_alpha(frame, 20 + i * 20, 20)
        y = 320 + i * 70
        draw_text(ctx, 500, y, label, size=24, color=(a*0.5, a*0.5, a*0.6))
        draw_text(ctx, 900, y, value, size=26, bold=True, color=(a, a*0.95, a*0.9))

    # Green checkmark accent
    a2 = fade_alpha(frame, 120, 30)
    draw_glow_circle(ctx, W/2, 800, 20, tuple(c * a2 for c in GREEN), glow=True)


def render_vision_slide(ctx, frame, total_frames):
    """Future vision slide."""
    gradient_bg(ctx)

    a1 = fade_alpha(frame, 0, 25)
    draw_text(ctx, W/2, 250, "2027: 1 Billion AI Agents", size=48, bold=True,
              color=(a1, a1, a1), align='center')

    draw_text(ctx, W/2, 320, "They'll need trust infrastructure.", size=28,
              color=(a1*0.6, a1*0.6, a1*0.7), align='center')

    phases = [
        ("Now", "Trust simulation + Solana verification"),
        ("Phase 2", "SDK for agent platforms"),
        ("Phase 3", "Live trust oracle — cross-chain"),
        ("Phase 4", "Autonomous trust economy"),
    ]

    for i, (label, desc) in enumerate(phases):
        a = fade_alpha(frame, 40 + i * 30, 25)
        y = 440 + i * 80
        draw_glow_circle(ctx, 540, y - 8, 6, tuple(c * a for c in CYAN))
        if i > 0:
            ctx.set_source_rgba(*CYAN, a * 0.3)
            ctx.set_line_width(1)
            ctx.move_to(540, y - 30)
            ctx.line_to(540, y - 14)
            ctx.stroke()
        draw_text(ctx, 570, y, label, size=22, bold=True, color=(a*0.8, a*0.9, a))
        draw_text(ctx, 570, y + 28, desc, size=18, color=(a*0.5, a*0.5, a*0.6))


def render_closing_slide(ctx, frame, total_frames):
    """Closing slide with kicker quote."""
    gradient_bg(ctx)

    a1 = fade_alpha(frame, 0, 30)
    draw_text(ctx, W/2, H/2 - 120, '"What economic system would AI', size=36,
              color=(a1*0.8, a1*0.8, a1*0.9), align='center')
    draw_text(ctx, W/2, H/2 - 70, 'design for itself?"', size=36,
              color=(a1*0.8, a1*0.8, a1*0.9), align='center')

    a2 = fade_alpha(frame, 40, 30)
    draw_text(ctx, W/2, H/2 + 10, '"You\'re looking at it."', size=44, bold=True,
              color=(a2, a2, a2), align='center')

    # Accent line
    a3 = fade_alpha(frame, 70, 30)
    line_w = 200 * a3
    ctx.set_source_rgba(*GREEN, a3 * 0.6)
    ctx.set_line_width(2)
    ctx.move_to(W/2 - line_w/2, H/2 + 50)
    ctx.line_to(W/2 + line_w/2, H/2 + 50)
    ctx.stroke()

    a4 = fade_alpha(frame, 100, 25)
    draw_text(ctx, W/2, H/2 + 120, "github.com/solishq/AEZ-Evolution", size=22,
              color=(a4*0.5, a4*0.7, a4), align='center')
    draw_text(ctx, W/2, H/2 + 160, "Program ID: GYYRqgHqsYQuYfZNCsQRKCxpJdy9gRS5A6aAK9fDCY7g",
              size=16, color=(a4*0.4, a4*0.4, a4*0.5), align='center')

    a5 = fade_alpha(frame, 130, 25)
    draw_text(ctx, W/2, H/2 + 220, "Built by Femi & Anna  |  solishq.com", size=20,
              color=(a5*0.5, a5*0.5, a5*0.6), align='center')


# ─── Main Pipeline ──────────────────────────────────────────

def generate_video():
    """Generate the complete pitch video."""
    print("\n  AEZ Evolution — Video Generator")
    print("  " + "=" * 50)

    # Step 1: Run simulation
    sim_data, summary, narrator = run_simulation()

    # Step 2: Define sections
    sections = [
        ('title',       150),   # 0:00-0:05
        ('problem',     300),   # 0:05-0:15
        ('solution',    300),   # 0:15-0:25
        ('sim',         900),   # 0:25-0:55  (Phase 1: R1-25)
        ('sim',         900),   # 0:55-1:25  (Phase 2: R26-50)
        ('sim',         900),   # 1:25-1:55  (Phase 3: R51-75)
        ('sim',         900),   # 1:55-2:25  (Phase 4: R76-100)
        ('results',     450),   # 2:25-2:40
        ('vision',      300),   # 2:40-2:50
        ('closing',     300),   # 2:50-3:00
    ]

    total_frames = sum(f for _, f in sections)
    print(f"  Total frames: {total_frames} ({total_frames/FPS:.0f}s @ {FPS}fps)")

    # Step 3: Start FFmpeg process
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'bgra',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        OUTPUT
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Step 4: Render frames
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)

    frame_num = 0
    sim_round_idx = 0
    coop_history = []

    # Simulation round ranges per sim section
    sim_sections = [
        (0, 25),    # Phase 1
        (25, 50),   # Phase 2
        (50, 75),   # Phase 3
        (75, 100),  # Phase 4
    ]
    sim_section_idx = 0

    for section_type, section_frames in sections:
        section_name = section_type
        if section_type == 'sim':
            section_name = f"sim (Phase {sim_section_idx + 1})"

        for f in range(section_frames):
            # Clear
            ctx.save()
            ctx.set_operator(cairo.OPERATOR_CLEAR)
            ctx.paint()
            ctx.restore()

            if section_type == 'title':
                render_title_card(ctx, f, section_frames)

            elif section_type == 'problem':
                render_problem_slide(ctx, f, section_frames)

            elif section_type == 'solution':
                render_solution_slide(ctx, f, section_frames)

            elif section_type == 'sim':
                # Map video frame to simulation round
                start_r, end_r = sim_sections[sim_section_idx]
                num_rounds = end_r - start_r
                round_in_section = int(f / section_frames * num_rounds)
                round_idx = start_r + min(round_in_section, num_rounds - 1)

                if round_idx < len(sim_data):
                    data = sim_data[round_idx]
                    coop_rate = data['stats'].get('coop_rate', 0)
                    # Only add to history when we advance to a new round
                    if len(coop_history) <= round_idx:
                        coop_history.append(coop_rate)
                    render_simulation_frame(ctx, data, f, section_frames, coop_history)

            elif section_type == 'results':
                render_results_slide(ctx, f, section_frames, summary)

            elif section_type == 'vision':
                render_vision_slide(ctx, f, section_frames)

            elif section_type == 'closing':
                render_closing_slide(ctx, f, section_frames)

            # Write frame to FFmpeg
            buf = surface.get_data()
            proc.stdin.write(bytes(buf))

            frame_num += 1
            if frame_num % 300 == 0:
                pct = frame_num / total_frames * 100
                print(f"  Rendered {frame_num}/{total_frames} frames ({pct:.0f}%)")

        if section_type == 'sim':
            sim_section_idx += 1

    # Step 5: Finalize
    proc.stdin.close()
    proc.wait()

    if proc.returncode == 0:
        size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
        duration = total_frames / FPS
        print(f"\n  Video generated: {OUTPUT}")
        print(f"  Duration: {duration:.0f}s | Size: {size_mb:.1f} MB | Resolution: {W}x{H}")
        print(f"  " + "=" * 50)
    else:
        stderr = proc.stderr.read().decode()
        print(f"\n  FFmpeg error:\n{stderr[-500:]}")
        sys.exit(1)


if __name__ == '__main__':
    generate_video()
