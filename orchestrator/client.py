"""
AEZ Evolution Solana Client

Interacts with the deployed program to:
- Create genomes
- Spawn agents  
- Create and resolve commitments
- Run evolution cycles
"""

import asyncio
import json
import hashlib
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Optional: Use solana-py if available
try:
    from solana.rpc.async_api import AsyncClient
    from solana.transaction import Transaction
    from solana.keypair import Keypair
    from solana.publickey import PublicKey
    SOLANA_PY_AVAILABLE = True
except ImportError:
    SOLANA_PY_AVAILABLE = False
    print("Warning: solana-py not installed. Using subprocess calls to solana CLI.")


class Strategy(Enum):
    COOPERATOR = 0
    DEFECTOR = 1
    TIT_FOR_TAT = 2
    GRUDGER = 3
    RANDOM = 4


class Action(Enum):
    COOPERATE = 0
    DEFECT = 1


@dataclass
class Genome:
    pubkey: str
    name: str
    strategy: Strategy
    generation: int
    total_spawned: int


@dataclass
class Agent:
    pubkey: str
    genome: str
    compute_balance: int
    fitness_score: int
    interactions: int
    cooperations: int
    defections: int
    alive: bool


@dataclass
class Commitment:
    pubkey: str
    agent_a: str
    agent_b: str
    compute_stake: int
    resolved: bool


class AEZClient:
    """Client for interacting with AEZ Evolution on Solana devnet"""
    
    PROGRAM_ID = "GYYRqgHqsYQuYfZNCsQRKCxpJdy9gRS5A6aAK9fDCY7g"
    
    def __init__(self, rpc_url: str = "https://api.devnet.solana.com", keypair_path: Optional[str] = None):
        self.rpc_url = rpc_url
        self.keypair_path = keypair_path or os.path.expanduser("~/.config/solana/id.json")
        
        if SOLANA_PY_AVAILABLE:
            self.client = AsyncClient(rpc_url)
            self._load_keypair()
        else:
            self.client = None
            self.keypair = None
    
    def _load_keypair(self):
        """Load keypair from file"""
        try:
            with open(self.keypair_path) as f:
                secret = json.load(f)
            self.keypair = Keypair.from_secret_key(bytes(secret))
            print(f"Loaded keypair: {self.keypair.public_key}")
        except Exception as e:
            print(f"Could not load keypair: {e}")
            self.keypair = None
    
    async def get_balance(self) -> int:
        """Get SOL balance"""
        if not SOLANA_PY_AVAILABLE:
            import subprocess
            result = subprocess.run(
                ["solana", "balance", "--url", self.rpc_url],
                capture_output=True, text=True
            )
            return float(result.stdout.strip().split()[0])
        
        response = await self.client.get_balance(self.keypair.public_key)
        return response['result']['value'] / 1e9
    
    def derive_genome_pda(self, authority: str, name: str) -> str:
        """Derive PDA for genome account"""
        # Seeds: ["genome", authority, name]
        seeds = [b"genome", bytes.fromhex(authority.replace("0x", "")), name.encode()]
        # This would need actual PDA derivation - simplified for now
        return f"genome_{name}"
    
    def derive_agent_pda(self, genome: str, spawn_index: int) -> str:
        """Derive PDA for agent account"""
        # Seeds: ["agent", genome, spawn_index]
        return f"agent_{genome}_{spawn_index}"
    
    async def create_genome(self, name: str, strategy: Strategy, mutation_rate: int = 5) -> str:
        """Create a new genome"""
        print(f"Creating genome: {name} ({strategy.name})")
        # TODO: Build and send transaction
        return f"genome_{name}"
    
    async def spawn_agent(self, genome: str, initial_compute: int = 1000) -> str:
        """Spawn a new agent from genome"""
        print(f"Spawning agent from {genome} with {initial_compute} compute")
        # TODO: Build and send transaction
        return f"agent_{genome}"
    
    async def create_commitment(self, agent_a: str, agent_b: str, stake: int = 100) -> str:
        """Create commitment between two agents"""
        print(f"Creating commitment: {agent_a} vs {agent_b}, stake={stake}")
        # TODO: Build and send transaction
        return f"commitment_{agent_a}_{agent_b}"
    
    def compute_action_hash(self, action: Action, nonce: bytes) -> bytes:
        """Compute commitment hash for action"""
        data = bytes([action.value]) + nonce
        return hashlib.sha256(data).digest()
    
    async def submit_action(self, commitment: str, action: Action, is_agent_a: bool) -> bytes:
        """Submit hashed action (commit phase)"""
        nonce = os.urandom(32)
        commit_hash = self.compute_action_hash(action, nonce)
        print(f"Submitting action hash for {'A' if is_agent_a else 'B'}")
        # TODO: Build and send transaction
        return nonce
    
    async def reveal_action(self, commitment: str, action: Action, nonce: bytes, is_agent_a: bool):
        """Reveal action (reveal phase)"""
        print(f"Revealing action: {action.name} for {'A' if is_agent_a else 'B'}")
        # TODO: Build and send transaction
    
    async def resolve_commitment(self, commitment: str, agent_a: str, agent_b: str):
        """Resolve commitment and distribute rewards"""
        print(f"Resolving commitment {commitment}")
        # TODO: Build and send transaction
    
    async def get_agent(self, pubkey: str) -> Optional[Agent]:
        """Fetch agent account data"""
        # TODO: Fetch and deserialize account
        return None
    
    async def get_genome(self, pubkey: str) -> Optional[Genome]:
        """Fetch genome account data"""
        # TODO: Fetch and deserialize account
        return None
    
    async def close(self):
        """Close client connection"""
        if self.client:
            await self.client.close()


async def test_connection():
    """Test connection to devnet"""
    client = AEZClient()
    balance = await client.get_balance()
    print(f"Balance: {balance} SOL")
    await client.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
