# DGX Spark — Notes from the Workshop

Working notes from running large language models on NVIDIA DGX Sparks. Everything here was tested on real hardware and documented honestly — including what didn't work, and what the community has since done better.

This is a fast-moving space. We've explored some of the pieces to this puzzle, certainly not all of them. What follows is what we've found so far.

## Inference Engines

### GLM 5.2 on 4× Sparks ✨ *NEW*

**[GLM 5.2 Quad-Spark Deployment Guide](https://github.com/marksunner/dgx-spark-glm52)** — Z.ai's 671B-parameter reasoning model running across four DGX Sparks with all 256 experts active, 200K context, and MTP speculative decoding (~26 tok/s).

The complete honest journey: unboxing virgin hardware, provisioning four nodes, building the QSFP fabric, the custom vLLM container, and the four bugs we found and fixed along the way (KV cache shape mismatch, Docker build-cache serving wrong commits, `__pycache__` bytecode staleness, DeepGEMM warmup failures).

Companion guide: **[What Is Fabric?](https://github.com/marksunner/dgx-spark-glm52/blob/main/what-is-fabric.md)** — a standalone walkthrough for turning a MikroTik CRS812 from a stock Ethernet switch into a lossless RoCE fabric. Friendly to first-timers ("first the earth cooled…") and reusable for any cluster project.

Built on tonyd2wild's [QuantTrio recipe](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark) — full attribution throughout.

### Atlas *(contributions merged upstream)*

[Atlas](https://github.com/Avarok-Cybersecurity/atlas) is a pure Rust inference engine built with an AI-first philosophy. We contributed Step 3.7 Flash NVFP4 support — Blackwell kernel targets, model architecture, weight loading, and expert parallelism for DGX Spark's sm_121a — and then spent several weeks debugging generation quality on dual Sparks (EP=2), which turned into ~15 additional fixes and one big finding.

- **[PR #136](https://github.com/Avarok-Cybersecurity/atlas/pull/136)** — Step 3.7 Flash NVFP4 support (**merged**)
- **[Issue #184](https://github.com/Avarok-Cybersecurity/atlas/issues/184)** — extended findings: the full quality investigation, quantization A/B, fix list, and open trails
- **[`step37-flash-fixes` branch](https://github.com/marksunner/atlas/tree/step37-flash-fixes)** — all post-#136 fixes as a single squashed commit on current main, diff-able and cherry-pickable
- **[Atlas Step 3.7 artifacts repo](https://github.com/marksunner/dgx-spark-step37-flash-atlas)** — test harnesses, raw A/B transcripts, fix→file map, and the validated dual-Spark launch recipe

The headline from the extended work: after fixing everything else (missing BOS token, unparsed `rope_scaling`, dropped `swiglu_limits` clamping, sliding-window decode masking, tool-call handling, and more — full list in Issue #184), the remaining ~30–40% repetition rate in long generations tracked the **NVFP4 4-bit expert path**. Serving the official **FP8 checkpoint** through a new fused-expert loader eliminated it: 0 catastrophic draws in 22 FP8 runs vs 2/5 on NVFP4, same engine, same settings. FP8 fits on 2× Sparks at EP=2 (~103 GB/rank) and decodes at ~20 tok/s with a reliably clean envelope out to ~70K context. The honest caveat: the A/B doesn't separate checkpoint quantization damage from a possible issue in Atlas's own NVFP4 expert kernels — that disambiguation is one of the open trails.

Atlas's Rust + CUDA approach is compelling for DGX Spark — native Blackwell support, no Python overhead, and a design philosophy that aligns with how we think about inference. We're stepping back from active development on this front, but everything we learned is public in the links above and cherry-pickable without contacting us.

### vLLM

Upstream vLLM does not support multi-node tensor parallelism on DGX Spark out of the box — its V1 engine hardcodes a loopback address for the NCCL rendezvous endpoint. StepFun's vLLM fork adds Step 3.7 model support, and with two source patches for multi-node NCCL plus a Ray executor, dual-Spark TP=2 works. The setup is documented in detail, including what fails and why.

- **[Step 3.7 Flash — Dual Spark](https://github.com/marksunner/dgx-spark-step37-dual)** — NVFP4 across two Sparks via patched StepFun vLLM + Ray TP=2. RoCE RDMA config, 262K context, 18.5 tok/s. Docker device permissions, NCCL environment variables, patches, and everything that didn't work.
- **[DeepSeek V4 Flash — TP Benchmark](https://github.com/marksunner/dgx-spark-vllm-tp-benchmark)** — 284B MoE dual-Spark benchmark, 12.4 tok/s. **⚠️ Superseded — see below.**

**Our DeepSeek V4 Flash benchmark is outdated.** It was an honest snapshot of early-2026 tooling (12.4 tok/s, 4K context tested), but the community has moved far past it. The current standard for DeepSeek V4 Flash on dual Sparks is **[tonyd2wild/deepseek-v4-flash-2x-spark-1m](https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m)**: **45.5 tok/s decode at a true 1M-token context** with MTP speculative decoding, ~800–900 tok/s prefill on real 400K–800K prompts, clean tool calling, running a 24/7 production agent fleet from a pre-built Docker image. If you're here for DeepSeek V4 on two Sparks, start there — our repo remains up for historical reference only.

### llama.cpp / Ollama

The simplest path to running large models on a single Spark. GGUF quantization, minimal setup, good throughput.

- **[Step 3.7 Flash — Single Spark](https://github.com/marksunner/dgx-spark-step37-flash)** — Q4_K_S GGUF, 27 tok/s, 128K context, vision via mmproj.
- **[Single-Spark Agent Stack](https://github.com/marksunner/dgx-spark-single-stack)** — Qwen 122B (hybrid INT4+FP8) + Hermes agent + Honcho memory + monitoring. A complete autonomous agent on one box, 41–47 tok/s.

### ds4 (DwarfStar)

antirez's inference engine, designed for distributed serving.

**⚠️ Superseded** — see [tonyd2wild/deepseek-v4-flash-2x-spark-1m](https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m) for the current community standard (45 tok/s, 1M context).

- **[DeepSeek V4 Flash — ds4 Benchmark](https://github.com/marksunner/dgx-spark-ds4-benchmark)** — 284B MoE dual-Spark benchmark, 11.4 tok/s. ⚠️ *Superseded — see the [1M-context recipe](https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m) above for current performance (45 tok/s with MTP).*

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

> **New third option — Atlas with the FP8 checkpoint (dual Spark).** Following the [Issue #184](https://github.com/Avarok-Cybersecurity/atlas/issues/184) investigation, Atlas now serves the official FP8 checkpoint at EP=2 across two Sparks: ~20 tok/s decode, zero repetition failures in our 22-run battery, with a validated working envelope of ~70K context (needle retrieval passes at 70–78K, fails at 85K+ — well short of the checkpoint's declared 256K, but clean where it works). Launch recipe and harnesses in the [artifacts repo](https://github.com/marksunner/dgx-spark-step37-flash-atlas). Pick it if you want Rust-native serving with the best generation quality we've measured on dual Sparks; pick vLLM if you need the full context window.

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

- **GLM 5.2 fits on 4× Sparks with room to spare.** The 405 GB QuantTrio Int4-Int8Mix checkpoint splits to ~98 GiB per node at TP=4, leaving ~30 GiB free for KV cache, CUDA graphs, and OS — enough for 200K context. That's with all 256 experts alive and no pruning.
- **RoCE is worth the effort for multi-node.** RDMA over the QSFP link gave us 46% throughput improvement over TCP for tensor-parallel workloads. The setup requires specific NCCL environment variables and Docker device cgroup permissions — documented in the [dual-Spark guide](https://github.com/marksunner/dgx-spark-step37-dual).

- **Docker `-v` is not `--device`.** Volume-mounting `/dev/infiniband` makes RDMA device files visible inside the container but Docker's device cgroup blocks actual access. You need `--device=/dev/infiniband/uverbsX` for each device. This cost us a morning so it doesn't have to cost you one.

- **Two-site bugs exist and will cost you days.** The KV cache shape mismatch that bit us hardest appeared in TWO different files (indexer.py and flashmla_sparse.py) with different propagation paths for the cache dtype string. The fix was identical in both places — checking `head_size == 576` instead of trusting `cache_dtype_str` — but finding both sites required reading vLLM's batch reshape pipeline. If your shape error looks familiar and "already fixed", check the `__pycache__` folder.
- **4-bit quantization quality is tensor-dependent, not just engine-dependent.** The same Step 3.7 model that runs cleanly as Q4_K_S on llama.cpp looped badly as NVFP4 on Atlas until the experts were served in FP8 — after every engine bug had been fixed. If a quantized model misbehaves, suspect *which tensors* are quantized before blaming the engine or the model. Details in [Atlas Issue #184](https://github.com/Avarok-Cybersecurity/atlas/issues/184).

## Hardware

All testing on NVIDIA DGX Sparks:
- **SoC:** GB10 Grace Blackwell (sm_121a)
- **Memory:** 128 GB unified (CPU+GPU shared)
- **Interconnect:** 200 Gbps QSFP-DD (ConnectX-7, RoCE capable) — used only for [dual-Spark tensor parallelism](https://github.com/marksunner/dgx-spark-step37-dual). The model bake-off and pipeline tests above used two independent Sparks on a standard LAN — no QSFP cable or special networking required.

## Acknowledgements

- **Nic (albond)** — hybrid INT4+FP8 recipe and OOM guardrails
- **AEON-7** — [vLLM Ultimate DGX Spark container](https://github.com/AEON-7/vllm-ultimate-dgx-spark) and sm_121a reference
- **tonyd2wild** — the [1M-context DeepSeek V4 dual-Spark recipe](https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m) that superseded our benchmark
- **StepFun** — Step 3.7 Flash model and vLLM fork
- **antirez** — ds4 inference engine
- **AzeezIsh and Thomas Braun** — Atlas creators, responsive and encouraging
- The **DGX Spark community**

## License

MIT
