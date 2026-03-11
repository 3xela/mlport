# CUDA Kernel Learning Project Guide (with ChatGPT as a coach)

Goal: build the ability to write solid CUDA kernels “off the dome” — like you write Python — by internalizing both **CUDA syntax** and the **performance thought process**.

This guide defines:
- what to build
- how to practice
- how to use ChatGPT without outsourcing the thinking
- what “good” looks like at each stage

---

## 0) How we’ll work together (contract)

You will drive the kernel design. I will act like:
- a compiler + profiler interpreter (when you paste outputs)
- a reviewer (for correctness + performance issues)
- a coach (asking the right questions, giving partial hints)

### Default interaction mode: “hint-first”
When you ask for help, I’ll respond in this order unless you explicitly ask for code:
1) clarify the target + constraints
2) ask 2–5 guiding questions
3) give a minimal hint / pattern
4) only then, provide code snippets if you request them

### Escalation ladder (you choose how much help)
When you get stuck, tell me which level you want:
- **L1:** questions only (no hints)
- **L2:** conceptual hints (no code)
- **L3:** pseudocode / skeleton (no full implementation)
- **L4:** partial code (core loop only)
- **L5:** full code (you’ll still write the next kernel alone)

---

## 1) Tooling + setup checklist

### Required
- `nvcc` (CUDA toolkit)
- a GPU (you have NVIDIA; great)
- a C++ build system (CMake is fine)

### Strongly recommended
- Nsight Compute (`ncu`)
- Nsight Systems (`nsys`)
- `clang-format` (optional but helps)

### Project structure (suggested)
- `kernels/`
  - `src/` (CUDA/C++)
  - `include/`
  - `tests/`
  - `bench/`
  - `scripts/`
  - `notes/`

### Baseline build targets
- `kernels_test` (correctness)
- `kernels_bench` (performance)
- `kernels_profile` (runs a fixed case for profiling)

---

## 2) The kernel thought process (the “stack” you’re internalizing)

Every kernel design is the same loop:

### Step A — Define the contract
- input/output shapes
- dtype
- layout (contiguous? strides?)
- numerical tolerance
- deterministic? (optional)
- target GPU (compute capability matters)

### Step B — Choose a parallelization strategy
- What is the *unit of work* per thread?
- per warp?
- per block?
- How do blocks tile the problem?

### Step C — Plan memory movement (this is 80% of performance)
- What loads come from global memory?
- Can you coalesce them?
- Can you reuse them?
- Should you stage in shared memory?
- Can you keep hot values in registers?
- Are you reading/writing each element more than once?

### Step D — Implement correctness-first
- simple indexing
- bounds checks
- reference CPU version
- unit test

### Step E — Optimize one lever at a time
Pick one:
- coalescing
- reduce global traffic (fuse / reuse)
- tiling into shared memory
- warp primitives (shuffle reductions)
- reduce sync / divergence
- manage register pressure / occupancy

### Step F — Measure and explain
For every optimization attempt, you should be able to answer:
- what bottleneck it targets
- why it should help
- what metric proved it helped (or didn’t)

---

## 3) The “big ideas” you must internalize

### Execution model
- Threads execute in **warps of 32**
- Warps are scheduled onto SMs
- Blocks are the unit of **shared memory** + synchronization
- Divergence happens **within a warp** (branches)

### Memory hierarchy (fast → slow)
- registers (per thread)
- shared memory (per block)
- L2 cache (device-wide)
- global memory (HBM/GDDR)

### Coalescing
The warp should ideally access adjacent addresses.
If your warp loads look “strided”, performance often collapses.

### Occupancy vs register pressure
More threads isn’t always faster.
Too many registers → spills → slow.
Too much shared memory per block → fewer resident blocks → lower occupancy.

### Roofline intuition
You’re either:
- **bandwidth-bound** (common for elementwise ops)
- **compute-bound** (common for heavy math / tensor cores)
- **latency-bound** (common when memory access is irregular)

---

## 4) Learning roadmap (kernels in sequence)

You’ll implement each kernel in 3 tiers:
- **Tier 0:** naive correct
- **Tier 1:** “professional baseline” (coalesced, sane block sizes, reduced overhead)
- **Tier 2:** optimized (tiling / warp tricks / fusion)

### Kernel 1 — Vector add / scale
Skills: indexing, grid-stride loops, coalesced loads/stores, benchmarking harness

Deliverables:
- CPU reference
- CUDA kernel + launch config
- benchmark vs simple baseline
- write down achieved GB/s

---

### Kernel 2 — Reduction (sum / max)
Skills: shared memory, synchronization, bank conflicts, numerical stability for sum, warp reductions

Deliverables:
- block reduction
- multi-stage reduction (block results → final)
- compare shared-reduction vs warp-shuffle version

---

### Kernel 3 — Softmax
Skills: row-wise reduction, stable softmax (max-subtraction), fusing passes

Deliverables:
- naive (3 passes: max, sum exp, normalize)
- fused-ish (fewer global reads)
- profile stalls + memory throughput

---

### Kernel 4 — LayerNorm
Skills: two moments (mean/var), reductions, vectorized loads, numerical stability

Deliverables:
- per-row LN
- compare two-pass vs fused approach
- report registers/thread + occupancy

---

### Kernel 5 — Tiled Matmul
Skills: shared memory tiling, blocking, reuse, loop unrolling, optionally tensor cores later

Deliverables:
- naive matmul
- tiled matmul (shared A/B tiles)
- correctness + GFLOPs estimate and measured

---

### Kernel 6 — Attention microkernel (toy)
Skills: tiling + reduction + numerics + reuse (FlashAttention-style ideas)

Deliverables:
- implement a simplified “block attention” for small shapes
- show stable softmax inside a tile

---

## 5) Benchmarks and profiling: required habits

### Benchmark rules
- warmup iterations
- multiple timed iterations
- `cudaEvent` timing (not CPU wall time)
- fixed problem sizes + sweep sizes
- report: time, throughput (GB/s or TFLOPs), and variance

### Profiling rules
- use Nsight Compute on the smallest case that still shows the pattern
- look for:
  - global load/store efficiency
  - achieved memory bandwidth
  - occupancy
  - warp stall reasons
  - register spills (local memory)

When you paste profiling output, include:
- GPU model
- kernel launch config
- problem size
- key metrics you’re staring at

---

## 6) Code review checklist (what “good” looks like)

### Correctness
- bounds checks correct
- no race conditions
- stable numerics where needed
- tests cover edge sizes (not multiples of blockDim)

### Performance hygiene
- grid-stride loops where appropriate
- avoid unnecessary sync
- minimize divergent branches
- coalesced access (most important early)
- avoid re-reading global memory repeatedly

### Maintainability (even for kernels)
- clear naming of indices (row, col, lane, warp, block)
- constants for tile sizes
- simple launch wrappers
- separate: kernel code vs host harness

---

## 7) Daily/weekly practice routine

### Daily (30–90 min)
1) implement a small kernel change
2) benchmark before/after
3) write 5–10 lines of notes: what changed and why

### Weekly (2–4 hrs)
1) one kernel milestone (e.g., Tier 1 → Tier 2)
2) one profiling deep dive
3) one “postmortem”: what surprised you?

---

## 8) “Explain it back” prompts (to force internalization)

After finishing each kernel, write answers to:
- What is the unit of parallel work?
- What is the bottleneck and how do you know?
- What memory accesses are coalesced? Which aren’t?
- Where do you synchronize, and why?
- What is the next optimization lever you didn’t try?

If you can answer those cleanly, you’re learning the right thing.

---

## 9) How to ask ChatGPT for help (templates)

### Design review (no code)
“Here’s my kernel goal + constraints. Ask me the questions I should answer before coding.”

### Debugging
“Here’s the code + the incorrect output. Help me isolate the bug with hypotheses and minimal hints.”

### Performance
“Here’s benchmark + Nsight metrics. Identify the likely bottleneck and suggest 2–3 targeted experiments.”

### Code review
“Review for: coalescing, divergence, shared memory correctness, register pressure. Don’t rewrite everything.”

---

## 10) Milestones (what mastery looks like)

### Milestone A — “I can write correct kernels quickly”
- vector ops + reductions without looking things up much
- can map shapes to thread/block indices naturally

### Milestone B — “I can optimize deliberately”
- can predict whether you’re bandwidth-bound or compute-bound
- can propose 2–3 optimizations and verify them with metrics

### Milestone C — “I can write ML-adjacent kernels”
- stable softmax / layernorm
- tiled matmul
- toy attention block

---

## 11) Next action (pick one)
Choose your starting kernel and scope:
- **K1:** vector add/scale (recommended)
- **K2:** reduction (if you want the “real” CUDA feeling immediately)

When you start, send me:
- the exact operation signature (inputs, outputs, shapes)
- a tiny test case (like n=256) + expected output
- your intended launch config guess (block size)

I’ll respond in L1/L2 mode by default and push you to articulate the mapping first.