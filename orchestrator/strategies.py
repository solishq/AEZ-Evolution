"""
AEZ Evolution Strategies

Each strategy decides: Cooperate or Defect based on history
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random


class Action(Enum):
    COOPERATE = 0
    DEFECT = 1


@dataclass
class StrategyState:
    """Tracks per-opponent history for stateful strategies"""
    opponent_actions: Dict[str, List[Action]] = field(default_factory=dict)
    betrayed_by: set = field(default_factory=set)
    
    def record_action(self, opponent: str, action: Action):
        if opponent not in self.opponent_actions:
            self.opponent_actions[opponent] = []
        self.opponent_actions[opponent].append(action)
        if action == Action.DEFECT:
            self.betrayed_by.add(opponent)
    
    def last_action(self, opponent: str) -> Optional[Action]:
        if opponent in self.opponent_actions and self.opponent_actions[opponent]:
            return self.opponent_actions[opponent][-1]
        return None
    
    def was_betrayed_by(self, opponent: str) -> bool:
        return opponent in self.betrayed_by


class Strategy:
    """Base strategy class"""
    name: str = "Base"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        raise NotImplementedError


class Cooperator(Strategy):
    """Always cooperates. Trusting and naive."""
    name = "Cooperator"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        return Action.COOPERATE


class Defector(Strategy):
    """Always defects. Pure self-interest."""
    name = "Defector"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        return Action.DEFECT


class TitForTat(Strategy):
    """
    Starts cooperating, then mirrors opponent's last move.
    The most successful IPD strategy - nice, retaliatory, forgiving.
    """
    name = "TitForTat"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        last = state.last_action(opponent)
        if last is None:
            return Action.COOPERATE  # Start nice
        return last  # Mirror their last move


class Grudger(Strategy):
    """
    Cooperates until betrayed, then always defects against that opponent.
    Unforgiving but never initiates defection.
    """
    name = "Grudger"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        if state.was_betrayed_by(opponent):
            return Action.DEFECT  # Never forgive
        return Action.COOPERATE


class Random(Strategy):
    """50/50 random choice. Unpredictable chaos agent."""
    name = "Random"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        return random.choice([Action.COOPERATE, Action.DEFECT])


class Pavlov(Strategy):
    """
    Win-stay, lose-shift. If last outcome was good (both cooperate or
    I defected and they cooperated), repeat. Otherwise switch.
    """
    name = "Pavlov"

    def __init__(self):
        self.last_my_action: Dict[str, Action] = {}

    def decide(self, opponent: str, state: StrategyState) -> Action:
        last_their = state.last_action(opponent)
        last_mine = self.last_my_action.get(opponent)

        if last_their is None or last_mine is None:
            action = Action.COOPERATE  # Start nice
        elif last_their == Action.COOPERATE:
            action = last_mine  # Stay with what worked
        else:
            # Switch
            action = Action.DEFECT if last_mine == Action.COOPERATE else Action.COOPERATE

        # Record my action for next time
        self.last_my_action[opponent] = action
        return action


class SuspiciousTitForTat(Strategy):
    """Like TitForTat but starts with defection. Tests waters first."""
    name = "SuspiciousTitForTat"
    
    def decide(self, opponent: str, state: StrategyState) -> Action:
        last = state.last_action(opponent)
        if last is None:
            return Action.DEFECT  # Start suspicious
        return last


# Strategy registry
STRATEGIES = {
    "Cooperator": Cooperator(),
    "Defector": Defector(),
    "TitForTat": TitForTat(),
    "Grudger": Grudger(),
    "Random": Random(),
    "Pavlov": Pavlov(),
    "SuspiciousTitForTat": SuspiciousTitForTat(),
}


def get_strategy(name: str) -> Strategy:
    return STRATEGIES.get(name, Cooperator())


# Payoff matrix for Prisoner's Dilemma
def calculate_payoff(action_a: Action, action_b: Action, stake: int = 100) -> tuple:
    """
    Returns (reward_a, reward_b) based on actions.
    
    Both Cooperate: (150, 150) - mutual benefit
    Both Defect: (50, 50) - mutual harm
    A Defects, B Cooperates: (200, 0) - A exploits B
    A Cooperates, B Defects: (0, 200) - B exploits A
    """
    if action_a == Action.COOPERATE and action_b == Action.COOPERATE:
        return (stake * 3 // 2, stake * 3 // 2)
    elif action_a == Action.DEFECT and action_b == Action.DEFECT:
        return (stake // 2, stake // 2)
    elif action_a == Action.DEFECT and action_b == Action.COOPERATE:
        return (stake * 2, 0)
    else:  # A cooperates, B defects
        return (0, stake * 2)
