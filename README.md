# DGX Spark — Notes from the Workshop

Working notes from running large language models on NVIDIA DGX Sparks. Everything here was tested on real hardware and documented honestly — including what didn't work.

This is a fast-moving space. We've explored some of the pieces to this puzzle, certainly not all of them. What follows is what we've found so far.

## Inference Engines

### Atlas *(active — contributing upstream)*

[Atlas](https://github.com/crate-ai/atlas) is a pure Rust inference engine built with an AI-first philosophy. We're actively contributing Step 3.7 Flash NVFP4 support — Blackwell kernel targets, model architecture, weight loading, and expert parallelism for DGX Spark's sm_121a.

- **[PR #119](https://github.com/crate-ai/atlas/pull/119)** — Step 3.7 Flash NVFP4 support (work in progress)
- **[Our Atlas fork](https://github.com/marksunner/atlas)** — Development branch

This is where most of our current energy is focused. Atlas's Rust + CUDA approach is compelling for DGX Spark — native Blackwell support, no Python overhead, and a design philosophy that aligns with how we think about inference.

### vLLM

Upstream vLLM does not support multi-node tensor parallelism on DGX Spark out of the box — its V1 engine hardcodes a loopback address for the NCCL rendezvous endpoint. StepFun's vLLM fork adds Step 3.7 model support, and with two source patches for multi-node NCCL plus a Ray executor, dual-Spark TP=2 works. The setup is documented in detail, including what fails and why.

- **[Step 3.7 Flash — Dual Spark](https://github.com/marksunner/dgx-spark-step37-dual)** — NVFP4 across two Sparks via patched StepFun vLLM + Ray TP=2. RoCE RDMA config, 262K context, 18.5 tok/s. Docker device permissions, NCCL environment variables, patches, and everything that didn't work.
- **[DeepSeek V4 Flash — TP Benchmark](https://github.com/marksunner/dgx-spark-vllm-tp-benchmark)** — 284B MoE dual-Spark benchmark, 12.4 tok/s.

### llama.cpp / Ollama

The simplest path to running large models on a single Spark. GGUF quantization, minimal setup, good throughput.

- **[Step 3.7 Flash — Single Spark](https://github.com/marksunner/dgx-spark-step37-flash)** — Q4_K_S GGUF, 27 tok/s, 128K context, vision via mmproj.
- **[Single-Spark Agent Stack](https://github.com/marksunner/dgx-spark-single-stack)** — Qwen 122B (hybrid INT4+FP8) + Hermes agent + Honcho memory + monitoring. A complete autonomous agent on one box, 41–47 tok/s.

### ds4 (DwarfStar)

antirez's inference engine, designed for distributed serving.

- **[DeepSeek V4 Flash — ds4 Benchmark](https://github.com/marksunner/dgx-spark-ds4-benchmark)** — 284B MoE dual-Spark benchmark, 11.4 tok/s.

## Step 3.7 Flash — One Spark or Two?

This is probably the most common question for anyone arriving with a DGX Spark and an interest in Step 3.7. Here's what we've found:

| | One Spark | Two Sparks |
|---|---|---|
| **Guide** | [Single Spark](https://github.com/marksunner/dgx-spark-step37-flash) | [Dual Spark](https://github.com/marksunner/dgx-spark-step37-dual) |
| **Engine** | llama.cpp (Q4_K_S GGUF) | vLLM (NVFP4, StepFun fork) |
| **Throughput** | ~27 tok/s | ~18.5 tok/s (with RoCE) |
| **Context** | 128K | **262K** |
| **Quantization** | Q4_K_S (more aggressive) | NVFP4 (closer to BF16) |
| **Vision** | Via mmproj | Native (1.8B encoder) |
| **Complexity** | Low | High |

The single-Spark option is faster and simpler. The dual-Spark option unlocks the model's full 262K native context — useful for large codebases, long documents, or extended conversations. Throughput is roughly equivalent; context window is the practical differentiator.

The NVFP4 model weights (~121 GB) simply don't fit on a single Spark with room for KV cache. That's the physics of it.

## Things We've Learned Along the Way

- **GPU and CPU don't compete on DGX Spark.** Inference engines claim GPU memory; agent frameworks run on CPU/system RAM. They coexist on one box without contention.

- **RoCE is worth the effort for multi-node.** RDMA over the QSFP link gave us 46% throughput improvement over TCP for tensor-parallel workloads. The setup requires specific NCCL environment variables and Docker device cgroup permissions — documented in the [dual-Spark guide](https://github.com/marksunner/dgx-spark-step37-dual).

- **Docker `-v` is not `--device`.** Volume-mounting `/dev/infiniband` makes RDMA device files visible inside the container but Docker's device cgroup blocks actual access. You need `--device=/dev/infiniband/uverbsX` for each device. This cost us a morning so it doesn't have to cost you one.

## Hardware

All testing on NVIDIA DGX Sparks:
- **SoC:** GB10 Grace Blackwell (sm_121a)
- **Memory:** 128 GB unified (CPU+GPU shared)
- **Interconnect:** 200 Gbps QSFP-DD (ConnectX-7, RoCE capable)

## Acknowledgements

- **Nic (albond)** — hybrid INT4+FP8 recipe and OOM guardrails
- **AEON-7** — [vLLM Ultimate DGX Spark container](https://github.com/AEON-7/vllm-ultimate-dgx-spark) and sm_121a reference
- **StepFun** — Step 3.7 Flash model and vLLM fork
- **antirez** — ds4 inference engine
- **AzeezIsh and Thomas Braun** — Atlas creators, responsive and encouraging
- The **DGX Spark community**

## License

MIT
