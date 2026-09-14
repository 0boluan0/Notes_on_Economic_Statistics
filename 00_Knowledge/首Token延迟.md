---
aliases:
  - 首 token 延迟是请求起点到首个输出 token 被观测到的时间差
  - Time to first token is the interval from a specified request start to the first observed output token
  - TTFT
student_os: knowledge-atom
atom_id: LLM-INF-003
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 首 token 延迟是请求起点到首个输出 token 被观测到的时间差

<!-- bilingual-en:start -->
*Time to first token is the interval from a specified request start to the first observed output token*
<!-- bilingual-en:end -->

**首 token 延迟**（time to first token，TTFT）衡量请求开始后，观察者等待第一个输出 token 的时间。必须同时说明请求起点与观测位置：客户端发送到收到首输出，和服务端接收到产生首输出，是不同区间。

<!-- bilingual-en:start -->
**Time to first token** (TTFT) measures how long an observer waits for the first output token after a request starts. Both the starting event and observation point must be specified: client send-to-receive and server receive-to-generate are different intervals.
<!-- bilingual-en:end -->

$$
\mathrm{TTFT}_{\mathrm{client}}
=t_{\mathrm{first\ token\ received}}-t_{\mathrm{request\ sent}}.
$$

这个客户端区间覆盖实际发生在两事件之间的网络传输、前处理、排队、[[预填充]]、首次选择及返回处理。因而 TTFT 不能直接当作 prefill kernel 的运行时间。

<!-- bilingual-en:start -->
The client interval includes the transmission, preprocessing, queueing, [[预填充|prefill]], first-token selection, and response handling that occur between those events. TTFT therefore cannot be treated directly as prefill-kernel runtime.
<!-- bilingual-en:end -->

## 同一请求的三个时间差

<!-- bilingual-en:start -->
*Three intervals from the same request*
<!-- bilingual-en:end -->

构造一条所有时刻已对齐的时间线：客户端在 0 ms 发送；服务端在 20 ms 接收；排队到 60 ms 开始预填充；100 ms 产生首 token；120 ms 客户端收到。则客户端 TTFT 为 120 ms，从服务端接收计为 80 ms，而例中预填充到首 token 的计算片段为 40 ms。这些数字描述不同事件区间，没有矛盾。

<!-- bilingual-en:start -->
In a constructed timeline with aligned timestamps, the client sends at 0 ms, the server receives at 20 ms, prefill starts at 60 ms, the first token is generated at 100 ms, and the client receives it at 120 ms. Client TTFT is 120 ms, the server receive-to-generate interval is 80 ms, and the illustrated prefill-to-first-token computation takes 40 ms. These describe different intervals.
<!-- bilingual-en:end -->

真实测量应在同一计时域计算时间差，或先解决跨机器时钟对齐；不能把两个未对齐的时间戳直接相减。vLLM V1 的文档示例以开始 tokenization 的 `arrival_time` 为服务端起点，因此即使指标也叫 TTFT，它仍不是客户端发送起算的值。

<!-- bilingual-en:start -->
Real measurements should calculate intervals within one timing domain or first establish cross-machine clock alignment. Unaligned timestamps cannot be subtracted directly. The documented vLLM V1 example starts its server-side interval at `arrival_time`, when tokenization begins, so its TTFT differs from a client send-based measurement.
<!-- bilingual-en:end -->

## 首响应未必包含首 token

<!-- bilingual-en:start -->
*The first response may contain no output token*
<!-- bilingual-en:end -->

流式接口可能先返回角色、空文本或其他元数据。测量时须声明是否忽略这些帧，以及首输出如何按[[Token口径]]识别。GenAI-Perf 文档把 TTFT 描述为发送请求到接收首响应；使用其结果时需保留工具的实际响应处理口径。后续流畅度看[[Token间隔]]，完整等待看[[请求完成延迟]]。

<!-- bilingual-en:start -->
A stream may first return a role, empty text, or other metadata. Specify whether such frames are ignored and how the first output is identified under the declared [[Token口径|token-counting convention]]. GenAI-Perf describes TTFT as request send to first response receipt; its actual response-handling convention must accompany reported results. [[Token间隔|Inter-token latency]] measures subsequent pacing, while [[请求完成延迟|request completion latency]] measures the full wait.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NVIDIA GenAI-Perf, Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html#metrics)：支持客户端发送与首响应接收的 TTFT 口径。
  <!-- bilingual-en:start -->
  Defines TTFT using client request sending and first-response receipt.
  <!-- bilingual-en:end -->
- [vLLM, Metrics](https://docs.vllm.ai/en/latest/design/metrics/)，Engine Core Events、Frontend Stats Collection：支持服务端排队/调度/token 事件以及从 tokenization 开始计算的 TTFT 例子。时间线数字为自构示例。
  <!-- bilingual-en:start -->
  Documents server events and the TTFT example beginning at tokenization. Numerical timestamps are constructed for this note.
  <!-- bilingual-en:end -->
