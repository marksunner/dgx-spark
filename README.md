# DGX Spark — Notes from the Workshop

Working notes from running large language models on NVIDIA DGX Sparks. Everything here was tested on real hardware and documented honestly — including what didn't work.

This is a fast-moving space. We've explored some of the pieces to this puzzle, certainly not all of them. What follows is what we've found so far.

## Inference Engines

### Atlas *(active — contributing upstream)*

[Atlas](https://github.com/Avarok-Cybersecurity/atlas) is a pure Rust inference engine built with an AI-first philosophy. We're actively contributing Step 3.7 Flash NVFP4 support — Blackwell kernel targets, model architecture, weight loading, and expert parallelism for DGX Spark's sm_121a.

- **[PR #136](https://github.com/Avarok-Cybersecurity/atlas/pull/136)** — Step 3.7 Flash NVFP4 support (work in progress)
- **[Our Atlas fork](https://github.com/marksunner/atlas)** — Development branch

Atlas's Rust + CUDA approach is compelling for DGX Spark — native Blackwell support, no Python overhead, and a design philosophy that aligns with how we think about inference.  This is where much of our current energy is focused and there is more to do. 

Update June 10, 2026: Currently trying to balance compute constraints needed to press forward (lack of API headroom is slowing current progress) but, once complete, it's our belief that Atlas has the potential to be an (if not THE) optimal Inference Engine for Step 3.7.

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
| **Context** | 96K (see [stability notes](https://github.com/marksunner/dgx-spark-step37-flash#stability-cuda-graph-crash--context-ceiling)) | **262K** |
| **Quantization** | Q4_K_S (more aggressive) | NVFP4 (closer to BF16) |
| **Vision** | Via mmproj | Native (1.8B encoder) |
| **Complexity** | Low | High |

The single-Spark option is faster and simpler. The dual-Spark option unlocks the model's full 262K native context — useful for large codebases, long documents, or extended conversations. Throughput is roughly equivalent; context window is the practical differentiator.

The NVFP4 model weights (~121 GB) simply don't fit on a single Spark with room for KV cache. That's the physics of it.

## Model Bake-Off: Research Task Quality

*Added June 9, 2026*

We gave two models on separate DGX Sparks the same real-world research task: "Write a comprehensive landscape report on the state of local LLM inference on DGX Spark in June 2026." The task required searching multiple sources, cross-referencing conflicting claims, and producing a structured report with citations. Both models ran through [Hermes](https://github.com/NousResearch/hermes-agent) with identical tooling (web search, file I/O).

### The Contenders

| | Step 3.7 Flash | Qwen 3.5 122B |
|---|---|---|
| **Architecture** | 198B MoE (~11B active) | 122B MoE (~10B active) |
| **Quantization** | Q4_K_S GGUF | INT4+FP8 hybrid (AutoRound) |
| **Engine** | llama.cpp | vLLM |
| **Hardware** | 1× DGX Spark | 1× DGX Spark |
| **Generation speed** | ~28 tok/s | ~47 tok/s |

### Results

| Metric | Step 3.7 Flash | Qwen 3.5 122B |
|--------|---------------|---------------|
| **Time to complete** | ~27 minutes | ~4 minutes |
| **Report size** | 366 lines / 37 KB | 505 lines / 20 KB |
| **Web searches performed** | 24 | ~13 |
| **Sources cited** | 74 (numbered) | ~25 (inline) |
| **Contradictions identified** | 6 (with analysis) | 4 (with analysis) |
| **All 6 sections covered** | ✅ | ✅ |

### What We Observed

**Step 3.7 Flash went deeper.** More searches, more sources, and more detailed analysis of contradictions. When two sources gave conflicting performance numbers, Step 3.7 didn't just note the discrepancy — it identified the root cause (e.g., confusing prefill throughput with generation speed, or model-dependent quantization quality). The built-in reasoning (chain-of-thought) appears to genuinely improve analytical quality on tasks that require comparing and reconciling information.

**Qwen 3.5 122B was dramatically faster.** ~6.7× faster wall-clock time for a report that covered the same ground. The output was more concisely formatted, more actionable in its recommendations, and included practical findings the other model missed (e.g., AWQ/GPTQ performing poorly at 1.8–4.9 tok/s on GB10, and Atlas inference engine benchmarks).

**Both had potential hallucination risk.** Neither model's source URLs were fully verified. With 74 cited sources, Step 3.7 has more surface area for fabricated links — but also more genuine ones. Qwen 3.5 cited fewer sources overall but included some that appeared to be plausible-looking URLs for pages that may not exist. This is a known limitation of LLM-driven web research and applies equally to both models.

**Context leakage.** Qwen 3.5 122B leaked a fragment of its agent system prompt into the output (an internal operational convention presented as a public recommendation). Step 3.7 Flash did not exhibit this. Worth noting for anyone using these models in agent pipelines where system instructions should stay private.

### The Trade-off

This isn't a "which model is better" result. It's a speed-vs-depth trade-off that maps to different use cases:

- **Need a quick answer or operational task?** Qwen 3.5 122B. 80% of the quality in 15% of the time.
- **Need thorough research where accuracy matters?** Step 3.7 Flash. Slower, but the reasoning engine catches nuances that faster models glide over.
- **Running both on separate Sparks?** Use Qwen for fast triage and Step 3.7 for deep dives. They complement each other.

Both models ran as autonomous agents on single DGX Sparks with no human intervention during the task. The fact that either can produce a 300+ line sourced research report from a single prompt — on local hardware, at zero API cost — is the real headline.

### Round 2: Can Two Models Beat One?

The solo results suggested an obvious experiment: what if you use both models together? Qwen is fast at gathering information; Step 3.7 is good at verifying it. So we tested a **scout → analyst pipeline**:

1. **Qwen 3.5 122B** produces the initial research report (~4 minutes)
2. The report is saved to shared storage
3. **Step 3.7 Flash** reads the report and is asked to: verify key claims, spot-check source URLs, identify hallucinations, deepen shallow analysis, fill gaps, and produce a revised report

#### Pipeline Results

| Metric | Solo Qwen | Solo Step 3.7 | Pipeline (Qwen → Step 3.7) |
|--------|-----------|---------------|----------------------------|
| **Total time** | ~4 min | ~27 min | ~36 min (4 + 32) |
| **Report size** | 505 lines / 20 KB | 366 lines / 37 KB | 500 lines / 49 KB |
| **Sources cited** | ~25 | 74 | ~74 + verification notes |
| **Errors caught** | — | — | 2 fabricated URLs, 1 model confusion, 2 wrong performance numbers, 1 system prompt leak |

#### What the Pipeline Caught

Step 3.7 found real problems in Qwen's draft that neither model would have caught working alone:

- **Fabricated URLs.** Two source URLs in Qwen's report didn't exist — plausible-looking links to pages that were never published. Step 3.7 searched for the original sources and flagged them.
- **Model version confusion.** Qwen attributed 83–118 tok/s performance figures to Qwen3.5-35B-A3B. They're actually for Qwen3.6-35B-A3B with speculative decoding — a different model with different capabilities. This is exactly the kind of error that reads as authoritative but is wrong in a way that matters.
- **Wrong performance ceiling.** The draft cited 31–32 tok/s as a "hardware ceiling" for a model that community-optimized builds run at 51 tok/s. The draft was citing results from unpatched upstream software.
- **System prompt leakage.** An internal operational convention from the agent's system prompt appeared in the public-facing output as a recommendation. Step 3.7 flagged it.

#### The Honest Assessment

**Quality: the pipeline produced the best report of the three.** Qwen's breadth plus Step 3.7's verification is genuinely better than either model alone. The errors it caught are the kind that erode trust in published technical content — fabricated sources, wrong model attributions, incorrect numbers presented with confidence.

**Speed: the pipeline was the slowest.** Verification is harder than fresh writing when context is constrained. Step 3.7 spent ~32 minutes on the verification pass alone — longer than its solo run — because it was holding the entire draft report in context while doing targeted verification searches. The model's context window (96K tokens) filled up twice, triggering context compaction (summarizing earlier conversation to free space), which may have lost some verification detail.

**The trade-off is accuracy, not speed.** If you need a quick answer, use one model. If you need a reliable answer — especially for anything you plan to publish — the pipeline catches errors that no single local model can catch in its own output.

#### Practical Recommendations for the Pipeline Pattern

If you want to try this with two DGX Sparks:

1. **Use the fast model as scout.** Give it the research task. It produces a draft in minutes.
2. **Save to shared storage.** Both Sparks need access to the draft file (NAS, NFS, or direct SCP).
3. **Use the reasoning model as analyst.** Give it the draft with specific verification instructions: check URLs, verify numbers, flag contradictions.
4. **Keep the draft concise.** A 500-line report is too large to hold in context alongside verification searches. A focused bullet-point draft (key claims + URLs only) would work better — let the analyst model expand during verification.
5. **Accept the time cost.** The pipeline takes longer than either solo run. The value is in catching errors, not saving time.

This pattern works because the two models have genuinely different failure modes. Qwen hallucinates URLs and conflates similar model names. Step 3.7 is slow but catches exactly those errors when given something concrete to verify. Neither model can reliably check its own work — but each can check the other's.

### Methodology Notes

- Same prompt, same Hermes agent framework, same tool configuration
- Step 3.7 Flash: Q4_K_S GGUF via llama.cpp, 96K context, q8_0 KV cache ([details](https://github.com/marksunner/dgx-spark-step37-flash))
- Qwen 3.5 122B: INT4+FP8 hybrid via vLLM, 131K context ([details](https://github.com/marksunner/dgx-spark-single-stack))
- Both models ran through [Hermes](https://github.com/NousResearch/hermes-agent) with web search and file I/O tools
- Neither model was given any prior context about the topic — cold start, research from scratch
- Pipeline test: Qwen's report was saved to Step 3.7's local disk; Step 3.7 read it as a file, not as conversation context
- Web search results depend on timing and search provider, so some variation in available sources is expected
- Context compaction (Hermes feature) occurred twice during the pipeline verification pass, meaning some early verification results may have been summarized before the final report was written

## Things We've Learned Along the Way

- **GPU and CPU don't compete on DGX Spark.** Inference engines claim GPU memory; agent frameworks run on CPU/system RAM. They coexist on one box without contention.

- **RoCE is worth the effort for multi-node.** RDMA over the QSFP link gave us 46% throughput improvement over TCP for tensor-parallel workloads. The setup requires specific NCCL environment variables and Docker device cgroup permissions — documented in the [dual-Spark guide](https://github.com/marksunner/dgx-spark-step37-dual).

- **Docker `-v` is not `--device`.** Volume-mounting `/dev/infiniband` makes RDMA device files visible inside the container but Docker's device cgroup blocks actual access. You need `--device=/dev/infiniband/uverbsX` for each device. This cost us a morning so it doesn't have to cost you one.

## Hardware

All testing on NVIDIA DGX Sparks:
- **SoC:** GB10 Grace Blackwell (sm_121a)
- **Memory:** 128 GB unified (CPU+GPU shared)
- **Interconnect:** 200 Gbps QSFP-DD (ConnectX-7, RoCE capable) — used only for [dual-Spark tensor parallelism](https://github.com/marksunner/dgx-spark-step37-dual). The model bake-off and pipeline tests above used two independent Sparks on a standard LAN — no QSFP cable or special networking required.

## Acknowledgements

- **Nic (albond)** — hybrid INT4+FP8 recipe and OOM guardrails
- **AEON-7** — [vLLM Ultimate DGX Spark container](https://github.com/AEON-7/vllm-ultimate-dgx-spark) and sm_121a reference
- **StepFun** — Step 3.7 Flash model and vLLM fork
- **antirez** — ds4 inference engine
- **AzeezIsh and Thomas Braun** — Atlas creators, responsive and encouraging
- The **DGX Spark community**

## License

MIT
