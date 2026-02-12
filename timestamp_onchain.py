#!/usr/bin/env python3
"""
AEZ Evolution — On-Chain Timestamp

Hashes the entire repository and writes it as a Solana memo transaction.
Immutable proof that this exact codebase existed at this timestamp.

Copyright (c) 2026 SolisHQ. MIT License.
"""

import hashlib
import os
import subprocess
import json
import sys
from datetime import datetime, timezone

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to include in the hash (source code only, no build artifacts)
EXTENSIONS = {'.py', '.rs', '.js', '.html', '.css', '.toml', '.yaml', '.yml', '.md', '.json', '.txt'}
EXCLUDE_DIRS = {'target', 'node_modules', '__pycache__', '.git', '.anchor', 'test-ledger', 'venv'}
EXCLUDE_FILES = {'PATENT-APPLICATION.md', 'PATENT-APPLICATION.pdf', 'COLOSSEUM-SUBMISSION.md',
                 'pitch-video.mp4', 'package-lock.json', 'Cargo.lock'}


def hash_repo():
    """SHA256 hash of all source files, sorted for determinism."""
    hasher = hashlib.sha256()
    files_hashed = []

    for root, dirs, files in os.walk(REPO_DIR):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for fname in sorted(files):
            if fname in EXCLUDE_FILES:
                continue
            ext = os.path.splitext(fname)[1]
            if ext not in EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, REPO_DIR)

            with open(fpath, 'rb') as f:
                content = f.read()

            # Hash: "relative/path\0content"
            hasher.update(relpath.encode())
            hasher.update(b'\0')
            hasher.update(content)
            files_hashed.append(relpath)

    return hasher.hexdigest(), files_hashed


def send_memo(memo_text):
    """Send a memo transaction on Solana devnet."""
    # Use solana CLI to transfer 0 SOL to self with memo
    wallet = subprocess.run(
        ['solana', 'address'],
        capture_output=True, text=True
    ).stdout.strip()

    # solana transfer supports --with-memo
    result = subprocess.run(
        ['solana', 'transfer', '--allow-unfunded-recipient',
         wallet, '0', '--with-memo', memo_text],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  Error: {result.stderr.strip()}")
        return None

    # Extract signature from output
    output = result.stdout.strip()
    # The signature is usually the last line or in the output
    for line in output.split('\n'):
        if 'Signature' in line or len(line) > 80:
            sig = line.split()[-1] if line.split() else line
            return sig

    return output


def main():
    print("\n  AEZ Evolution — On-Chain Timestamp")
    print("  " + "=" * 50)

    # Step 1: Hash the repo
    print("  Hashing repository...")
    repo_hash, files = hash_repo()
    print(f"  Files hashed: {len(files)}")
    print(f"  SHA256: {repo_hash}")

    # Step 2: Build memo
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    memo = f"AEZ-EVOLUTION|{repo_hash[:32]}|{now}|solishq"

    print(f"\n  Memo: {memo}")
    print(f"  Memo length: {len(memo)} bytes")

    # Step 3: Check wallet
    wallet = subprocess.run(['solana', 'address'], capture_output=True, text=True).stdout.strip()
    balance = subprocess.run(['solana', 'balance'], capture_output=True, text=True).stdout.strip()
    print(f"\n  Wallet: {wallet}")
    print(f"  Balance: {balance}")

    # Step 4: Send transaction
    print(f"\n  Sending memo transaction to Solana devnet...")
    sig = send_memo(memo)

    if sig:
        print(f"\n  Transaction signature: {sig}")
        print(f"  Explorer: https://explorer.solana.com/tx/{sig}?cluster=devnet")

        # Step 5: Save proof
        proof = {
            'timestamp': now,
            'repo_hash': repo_hash,
            'files_count': len(files),
            'memo': memo,
            'wallet': wallet,
            'transaction': sig,
            'network': 'devnet',
            'explorer': f'https://explorer.solana.com/tx/{sig}?cluster=devnet',
            'program_id': 'GYYRqgHqsYQuYfZNCsQRKCxpJdy9gRS5A6aAK9fDCY7g',
        }

        proof_path = os.path.join(REPO_DIR, 'PROOF-OF-EXISTENCE.json')
        with open(proof_path, 'w') as f:
            json.dump(proof, f, indent=2)

        print(f"  Proof saved: {proof_path}")
    else:
        print("  Transaction failed.")
        sys.exit(1)

    print("  " + "=" * 50)
    print(f"\n  This proves the AEZ Evolution codebase ({len(files)} files)")
    print(f"  existed at {now}, signed by wallet {wallet[:12]}...")
    print(f"  Hash: {repo_hash[:16]}... stored immutably on Solana.\n")


if __name__ == '__main__':
    main()
