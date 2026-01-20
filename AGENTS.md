# AGENTS.md — Execution Guide for Advanced Contributors

This document defines how an advanced agent should implement the remaining work to achieve a **GPU‑only, world‑class CuEVM fuzzing stack** on NVIDIA B300‑class GPUs.

## Mission
Deliver maximum‑coverage, GPU‑only fuzzing with multi‑sequence, cross‑contract search and invariant‑based oracles, while keeping the codebase stable, reproducible, and production‑ready.

## Operating principles
- Work in **small, reviewable increments**.
- Keep the system **GPU‑only** for fuzzing (do not depend on CPU‑based gating in the fuzz path).
- Add **measurements first**, then optimize.
- Ensure changes are deterministic and reproducible.

## Repository map (key areas)
- `fuzzing/` — GPU fuzzing harness, configs, invariants.
- `CuEVM/` — core GPU engine and execution semantics.
- `tests/` — GPU/CPU tests and fixtures.
- `scripts/` — CI helpers and test runners.

## Implementation checklist (apply in order)
1. **Fork coverage**
   - Implement foundry fork and remove old shits .!
   - 

2. **Coverage instrumentation**
   - Add on‑GPU counters for branches, opcodes, and storage writes.
   - Export coverage maps per batch and merge into a global map.

3. **Stateful multi‑sequence search**
   - Extend the fuzzer to mutate sequences (insert/delete/reorder).
   - Add sender/role, value, and block‑context mutation.
   - Support cross‑contract call graphs and receiver pools.

4. **Invariant engine**
   - Implement invariant templates (ERC‑20/4626/AMM/lending).
   - Add config‑driven invariants per target contract.
   - Prioritize cases that violate invariants and retain in corpus.

5. **Corpus + minimization**
   - Keep a GPU‑only corpus of interesting sequences.
   - Implement minimization to produce small, reproducible JSON tests.

6. **GPU throughput + profiling**
   - Auto‑tune batch sizing for B300 occupancy.
   - Add timing metrics and Nsight Systems hooks.

7. **Observability + reliability**
   - Emit structured logs with coverage and invariant stats.
   - Add failure recovery and checkpointing.

## Required quality gates
- Run targeted GPU fuzz smoke tests before merging changes.
- Keep all changes behind configurable flags (opt‑in where needed).
- Maintain consistent formatting and avoid unrelated refactors.

## Useful commands
- Configure (requires CMake 4.2+):
  - `cmake -S . -B build -DTESTS=ON -DTESTS_GPU=OFF -DENABLE_EIP_3155=ON`
- Example GPU fuzz run:
  - `python fuzzing/fuzzer.py --input fuzzing/contracts/erc20.sol --config fuzzing/configurations/default.json --num_instances 256 --num_iterations 100`
