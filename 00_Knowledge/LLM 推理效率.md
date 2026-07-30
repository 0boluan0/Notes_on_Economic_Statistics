---
aliases:
  - "LLM Inference Efficiency"
  - "LLM Inference Optimization"
  - "大模型推理优化"
status: source-checked
---

# LLM 推理效率
<!-- bilingual-en:start -->
*LLM Inference Efficiency*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在显存、吞吐和响应时间约束下，把已经训练好的模型高效地服务给用户。
> **具体锚点：** 长提示先并行做 prefill，随后逐 token decoding；前者常受算力限制，后者频繁读取权重和 KV cache，瓶颈不同。
> **核心难点：** 参数量、首 token 延迟、每 token 延迟、并发吞吐和显存不能用一个数字概括；优化一种指标可能伤害另一种。
> **为什么重要：** 同一模型在不同 batch、上下文长度和硬件上的实际成本可相差很大。
> **继续：** 先测量 prefill/decode 和真实流量分布，再选 batching、量化、attention kernel 或 speculative decoding。
> <!-- bilingual-en:start -->
> **Problem addressed:** Serve a trained model efficiently under memory, throughput, and response-time constraints.
> **Concrete anchor:** A long prompt is processed in a parallel prefill, followed by token-by-token decoding. Prefill is often compute-bound, while decoding repeatedly reads weights and the KV cache, so their bottlenecks differ.
> **Central difficulty:** Parameter count, time to first token, time per output token, concurrent throughput, and memory cannot be summarized by one number; improving one can damage another.
> **Why it matters:** The same model can have radically different practical cost under different batches, context lengths, and hardware.
> **Continue with:** Measure prefill and decode under the real traffic distribution before selecting batching, quantization, attention kernels, or speculative decoding.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
> - The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
> <!-- bilingual-en:end -->

## Prefill 与 decoding
<!-- bilingual-en:start -->
*Prefill and Decoding*
<!-- bilingual-en:end -->

prefill 对整个提示构建各层 KV，容易获得较高并行度；decoding 每步只生成一个或少数 token，却要读取大量权重和历史 KV。报告端到端延迟时至少拆 TTFT（time to first token）、TPOT（time per output token）和吞吐。
<!-- bilingual-en:start -->
Prefill builds the layer-wise keys and values for the full prompt and exposes substantial parallelism. Decoding produces one or a few tokens per step while repeatedly reading model weights and historical KV state. End-to-end latency should therefore be separated into time to first token, time per output token, and throughput.
<!-- bilingual-en:end -->

对请求长度 $L_{in}$、输出长度 $L_{out}$，用户感知时间粗略是 $T_{prefill}(L_{in})+\sum_{t=1}^{L_{out}}T_{decode}(L_{in}+t)$，还要加排队与网络时间。只测 tokens/s 会隐藏首 token 等待，只测 TTFT 又会隐藏长回答的流式速度。
<!-- bilingual-en:start -->
For input length $L_{in}$ and output length $L_{out}$, user-perceived time is roughly $T_{prefill}(L_{in})+\sum_{t=1}^{L_{out}}T_{decode}(L_{in}+t)$ plus queueing and network time. Measuring only tokens per second hides the wait for the first token, while measuring only TTFT hides streaming speed for long responses.
<!-- bilingual-en:end -->

prefill 的大矩阵乘在足够长度和 batch 下通常能提高算力利用率；decode 的矩阵形状较窄，且每步需把权重与缓存搬入计算单元，常更受内存带宽限制。该判断应由目标硬件上的 profiler 验证，而不是作为所有模型的固定定律。
<!-- bilingual-en:start -->
The large matrix multiplications in prefill can often reach high compute utilization at sufficient sequence length and batch size. Decode uses narrower matrices and moves weights and cache data at every step, making it commonly more memory-bandwidth-bound. A profiler on the target hardware must verify this diagnosis rather than treating it as a universal law.
<!-- bilingual-en:end -->

## KV cache
<!-- bilingual-en:start -->
*KV Cache*
<!-- bilingual-en:end -->

self-attention 生成时缓存历史 key/value，避免每步重算前缀。缓存大致随层数、序列长度、batch、KV head 和精度增长。MQA/GQA 减少 KV heads，paged attention 改善不等长请求的显存管理，但都不改变必须验证质量和调度的事实。
<!-- bilingual-en:start -->
During autoregressive generation, self-attention caches historical keys and values to avoid recomputing the prefix at every step. Cache size grows roughly with layers, sequence length, batch, KV heads, and numerical precision. MQA and GQA reduce the number of KV heads, while paged attention improves memory management for variable-length requests; neither removes the need to verify quality and scheduling.
<!-- bilingual-en:end -->

忽略对齐与元数据，一份 cache 的字节数约为 $2\times n_{layers}\times B\times L\times n_{kvheads}\times d_{head}\times bytes$；开头的 2 代表 K 与 V。普通 multi-head attention 常有 $n_{kvheads}=n_{heads}$，GQA/MQA 通过共享 KV 降低该项。
<!-- bilingual-en:start -->
Ignoring alignment and metadata, cache bytes are approximately $2\times n_{layers}\times B\times L\times n_{kvheads}\times d_{head}\times bytes$; the leading two represents keys and values. Standard multi-head attention often has $n_{kvheads}=n_{heads}$, while GQA and MQA share KV state and reduce that factor.
<!-- bilingual-en:end -->

paged attention 把逻辑连续的每个请求 cache 映射到非连续物理块，减少预留最大长度造成的内部碎片，也方便共享前缀与动态调度。它解决内存管理，不会改变 attention 随已生成上下文增长的读取工作量。
<!-- bilingual-en:start -->
Paged attention maps each request's logically contiguous cache onto noncontiguous physical blocks, reducing internal fragmentation from reserving maximum length and supporting prefix sharing and dynamic scheduling. It solves memory management; it does not remove the attention reads that grow with generated context.
<!-- bilingual-en:end -->

## Worked example：估算 KV cache
<!-- bilingual-en:start -->
*Worked Example: Estimate KV-Cache Memory*
<!-- bilingual-en:end -->

设 32 层、batch 8、上下文 4,096、8 个 KV heads、head dimension 128、BF16 两字节，则 cache 约为 $2\times32\times8\times4096\times8\times128\times2\approx4$ GiB。若把上下文翻倍且并发不变，cache 也近似翻倍；模型权重没有变化。
<!-- bilingual-en:start -->
For 32 layers, batch eight, context 4,096, eight KV heads, head dimension 128, and two-byte BF16, cache size is about $2\times32\times8\times4096\times8\times128\times2\approx4$ GiB. Doubling context at fixed concurrency approximately doubles cache memory even though model weights do not change.
<!-- bilingual-en:end -->

这项估算可先判断 OOM 来自权重还是请求状态。若权重 14 GiB、可用显存 24 GiB，除 cache 外还要为临时 workspace、CUDA graph 和碎片留余量，不能把剩余 10 GiB 全部承诺给并发请求。
<!-- bilingual-en:start -->
This estimate distinguishes weight memory from request-state memory before an out-of-memory failure. If weights occupy 14 GiB on a 24 GiB device, temporary workspaces, CUDA graphs, and fragmentation still require headroom; the remaining 10 GiB cannot all be promised to concurrent requests.
<!-- bilingual-en:end -->

## Batching 与 serving
<!-- bilingual-en:start -->
*Batching and Serving*
<!-- bilingual-en:end -->

continuous batching 在请求到达/完成时动态填槽，提高设备利用率；代价是排队与尾延迟。吞吐优先的离线任务和低延迟交互服务应使用不同 batch 策略，不能只报满载峰值。
<!-- bilingual-en:start -->
Continuous batching fills slots dynamically as requests arrive and finish, improving device utilization at the cost of queueing and tail latency. Throughput-oriented offline jobs and low-latency interactive services need different batching policies; saturated peak throughput alone is not representative.
<!-- bilingual-en:end -->

调度器还要决定 prefill 是否打断 decode、长请求是否饿死短请求、最大 batch token、cache eviction 与优先级。chunked prefill 可把大提示切成小块，减少阻塞已有 decode，却可能增加调度和 kernel 开销。
<!-- bilingual-en:start -->
The scheduler must also decide whether prefill can interrupt decode, whether long requests starve short ones, the maximum tokens per batch, cache eviction, and priorities. Chunked prefill divides a large prompt to reduce blocking of active decodes, but can add scheduling and kernel overhead.
<!-- bilingual-en:end -->

## 量化
<!-- bilingual-en:start -->
*Quantization*
<!-- bilingual-en:end -->

权重或激活从 FP16/BF16 降到 INT8/INT4 可减显存与带宽。PTQ 直接校准，QAT 在训练中适应。收益依赖硬件 kernel、量化粒度和异常通道；用目标任务验证困惑度、准确率和生成行为。
<!-- bilingual-en:start -->
Reducing weights or activations from FP16 or BF16 to INT8 or INT4 can lower memory and bandwidth. Post-training quantization calibrates an existing model, while quantization-aware training adapts during training. Gains depend on hardware kernels, quantization granularity, and outlier channels; perplexity, accuracy, and generation behavior must be checked on target tasks.
<!-- bilingual-en:end -->

按组 scale、zero point、权重-only 或权重—激活联合量化会产生不同 kernel 与误差。文件更小不保证更快：若硬件要在运行时昂贵反量化，或缺少对应低精度矩阵 kernel，吞吐可能没有收益。
<!-- bilingual-en:start -->
Group-wise scales, zero points, weight-only quantization, and joint weight–activation quantization produce different kernels and errors. A smaller file does not guarantee faster execution: expensive runtime dequantization or missing low-precision matrix kernels can erase throughput gains.
<!-- bilingual-en:end -->

## 高效 attention 与 speculative decoding
<!-- bilingual-en:start -->
*Efficient Attention and Speculative Decoding*
<!-- bilingual-en:end -->

FlashAttention 通过 IO-aware 分块精确计算 attention，减少高带宽内存读写，并非近似 attention。speculative decoding 用小 draft 模型提出多个 token，再由目标模型并行验收；分布保持正确时，速度取决于接受率和两个模型的成本比。
<!-- bilingual-en:start -->
FlashAttention computes exact attention with I/O-aware tiling that reduces high-bandwidth-memory traffic; it is not approximate attention. Speculative decoding lets a small draft model propose several tokens and the target model verify them in parallel. When the target distribution is preserved, speed depends on acceptance rate and the cost ratio between the two models.
<!-- bilingual-en:end -->

FlashAttention 改善的是 attention kernel 的内存访问，不能消除 decoder 其余权重读取；短上下文时收益也可能被固定开销掩盖。speculative decoding 若 draft 太弱则接受率低，若太强又接近目标成本，因此要在目标提示与采样设置下联合调参。
<!-- bilingual-en:start -->
FlashAttention improves memory access inside the attention kernel; it does not remove weight reads in the rest of the decoder, and fixed overhead can hide gains at short context. In speculative decoding, a weak draft yields low acceptance while an expensive draft approaches target cost, so both must be tuned on the target prompts and sampling settings.
<!-- bilingual-en:end -->

## 测量口径
<!-- bilingual-en:start -->
*Measurement Protocol*
<!-- bilingual-en:end -->

固定模型版本、精度、硬件、输入/输出长度分布和并发；同时报告 TTFT、TPOT、tokens/s、峰值显存、能耗/成本和质量变化。一次短提示的 microbenchmark 不能代表生产流量。
<!-- bilingual-en:start -->
Fix model version, precision, hardware, input and output length distributions, and concurrency. Report TTFT, TPOT, tokens per second, peak memory, energy or cost, and quality change together. A single short-prompt microbenchmark does not represent production traffic.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- OOM 随并发或长度出现：先用 KV 公式估算，再检查碎片、预留块、workspace 与 cache eviction。
  <!-- bilingual-en:start -->
  Out-of-memory failures appear with concurrency or length: estimate KV memory first, then inspect fragmentation, reserved blocks, workspaces, and cache eviction.
  <!-- bilingual-en:end -->
- 平均延迟正常但 p99 很差：按请求长度、排队时间、prefill 干扰和 cache miss 分层，不要只看均值。
  <!-- bilingual-en:start -->
  Mean latency is acceptable but p99 is poor: stratify by request length, queueing time, prefill interference, and cache misses rather than relying on the mean.
  <!-- bilingual-en:end -->
- INT4 模型反而更慢：确认实际 kernel、batch/shape、反量化和 CPU–GPU 传输，避免把文件大小当运行带宽。
  <!-- bilingual-en:start -->
  An INT4 model is slower: verify the actual kernel, batch and shapes, dequantization, and CPU–GPU transfers instead of treating file size as runtime bandwidth.
  <!-- bilingual-en:end -->
- speculative decoding 没加速：分别测 draft 成本、target verification 成本、接受 token 数和调度开销。
  <!-- bilingual-en:start -->
  Speculative decoding gives no speedup: separately measure draft cost, target-verification cost, accepted-token count, and scheduling overhead.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 为什么长上下文会显著增加 KV cache？
<!-- bilingual-en:start -->
*Why does long context substantially increase the KV cache?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 每层要为每个历史位置保存 key/value，缓存随序列长度和并发近似线性增长。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Every layer stores keys and values for each historical position, so cache grows approximately linearly with sequence length and concurrency.
<!-- bilingual-en:end -->

### FlashAttention 是否通过近似 softmax 换速度？
<!-- bilingual-en:start -->
*Does FlashAttention gain speed by approximating softmax?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不是；它重排分块与内存访问来精确计算 attention，主要减少 HBM IO。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. It reorganizes tiling and memory access to compute exact attention, mainly reducing high-bandwidth-memory I/O.
<!-- bilingual-en:end -->

### 一种 INT4 方案吞吐更高，能否直接说更优？
<!-- bilingual-en:start -->
*Can an INT4 method be called better merely because its throughput is higher?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能；还要在目标硬件和流量下比较延迟、显存、成本及质量退化。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. Latency, memory, cost, and quality degradation must also be compared on the target hardware and traffic.
<!-- bilingual-en:end -->

### 怎样判断一次优化改善了 prefill 还是 decode？
<!-- bilingual-en:start -->
*How can you tell whether an optimization improved prefill or decode?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 分别固定输入长度、输出长度和并发，报告 TTFT 与 TPOT，并用 profiler 检查算力、带宽、attention 和权重读取时间。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Control input length, output length, and concurrency separately; report TTFT and TPOT and profile compute, bandwidth, attention, and weight-reading time.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
  <!-- bilingual-en:start -->
  [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
  <!-- bilingual-en:end -->
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
  <!-- bilingual-en:start -->
  The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
  <!-- bilingual-en:end -->
- [Dao et al. (2022), FlashAttention](https://arxiv.org/abs/2205.14135)：核验 IO-aware exact attention、分块机制与内存读写分析。
  <!-- bilingual-en:start -->
  [Dao et al. (2022), FlashAttention](https://arxiv.org/abs/2205.14135) verifies I/O-aware exact attention, tiling, and memory-traffic analysis.
  <!-- bilingual-en:end -->
- [Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：核验 paged KV-cache 管理、continuous batching 与 serving 指标。
  <!-- bilingual-en:start -->
  [Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) verifies paged KV-cache management, continuous batching, and serving measurements.
  <!-- bilingual-en:end -->
- [Leviathan, Kalman, and Matias (2023), Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：核验保持目标分布的 speculative decoding 与接受率机制。
  <!-- bilingual-en:start -->
  [Leviathan, Kalman, and Matias (2023), Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) verifies speculative decoding that preserves the target distribution and its acceptance mechanism.
  <!-- bilingual-en:end -->
