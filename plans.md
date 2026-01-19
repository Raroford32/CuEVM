# Plan: World‑class GPU‑only CuEVM fuzzing on NVIDIA B300

This plan lists **remaining work** needed to make CuEVM a production‑grade, GPU‑only fuzzer with maximum coverage, correctness, and throughput on B300‑class GPUs.

## 1) Engine + fork coverage (correctness foundation)
- [ ] Implement Osaka / Fulu‑Osaka (Fusaka) fork support in CuEVM (opcodes, precompiles, fork rules, and block context fields).
- [ ] Add fork selection in GPU runner config so fuzzing uses the intended fork rules without CPU gating.
- [ ] Expand EIP‑3155 trace coverage to include all fork‑specific opcodes.
- [ ] Add regression GPU tests for new fork behavior using focused JSON fixtures.

## 2) Coverage instrumentation + metrics
- [ ] Add on‑GPU coverage counters (branch + opcode + storage write sites).
- [ ] Export coverage summaries per batch to disk (JSON/CSV) for corpus management.
- [ ] Implement a coverage map merge step to guide next‑input selection.
- [ ] Track per‑contract and per‑function coverage for multi‑contract targets.

## 3) Stateful, multi‑sequence fuzzing (core search)
- [ ] Add sequence‑aware mutation operators (reorder, insert, delete, splice).
- [ ] Persist and replay sequences with deterministic seeds (GPU‑only).
- [ ] Add block‑context mutation (timestamp, number, basefee, prevRandao).
- [ ] Add sender/role mutation and value mutation per transaction.
- [ ] Introduce cross‑contract call graph awareness to drive inter‑contract sequences.

## 4) Invariants + oracles (signal, not noise)
- [ ] Expand invariant DSL: balance conservation, storage relations, access control, ERC‑4626/AMM/lending templates.
- [ ] Add invariant packs per protocol class with configuration templates.
- [ ] Implement invariant‑guided prioritization (keep cases that violate invariants).
- [ ] Add runtime assertions for invariants in Solidity (optional, but GPU‑only ingestion).

## 5) Corpus + minimization (production‑grade outputs)
- [ ] Maintain a GPU‑only corpus of “interesting” seeds (coverage increase or invariant violation).
- [ ] Implement delta‑debug minimization for tx sequences.
- [ ] Generate reproducible JSON test cases from minimized sequences.
- [ ] Track unique bug signatures and avoid duplicates.

## 6) GPU throughput + batch sizing
- [ ] Auto‑tune `num_instances` and `sequence_length` for B300 occupancy.
- [ ] Add batch‑level timers and throughput metrics (tx/s, sequences/s).
- [ ] Add Nsight Systems profile hooks for GPU bottleneck analysis.
- [ ] Introduce pinned memory pools for large batch I/O (where applicable).

## 7) Reliability + observability
- [ ] Add GPU health checks and hard failure handling (OOM, illegal instruction).
- [ ] Emit structured logs per batch with coverage and invariant stats.
- [ ] Add DCGM exporter and Prometheus/Grafana dashboards for GPU metrics.
- [ ] Add crash‑safe checkpointing of corpus and failing sequences.

## 8) CI + release hardening
- [ ] Add CI workflow for GPU fuzz smoke tests (short runs).
- [ ] Add nightly long‑run GPU fuzz jobs with artifact upload.
- [ ] Pin container base and toolchain versions (NGC + CUDA + CMake).
- [ ] Document reproducible release builds with B300 target settings.

## 9) Security + governance
- [ ] Threat‑model fuzz runner inputs and harden file handling.
- [ ] Add fuzzing sandbox / resource limits for untrusted targets.
- [ ] Add upgrade checklist for dependencies and GPU drivers.
