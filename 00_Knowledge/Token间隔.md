---
aliases:
  - Token 间隔是同一请求相邻输出 token 在同一观测点出现的时间差
  - Inter-token latency is the interval between adjacent output tokens observed at the same point for one request
  - ITL
student_os: knowledge-atom
atom_id: LLM-INF-004
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# Token 间隔是同一请求相邻输出 token 在同一观测点出现的时间差

<!-- bilingual-en:start -->
*Inter-token latency is the interval between adjacent output tokens observed at the same point for one request*
<!-- bilingual-en:end -->

**Token 间隔**（inter-token latency，ITL）度量同一请求的两个相邻输出之间等了多久。若第 $i$ 个 token 在同一观测点的出现时刻为 $t_i$，则

<!-- bilingual-en:start -->
**Inter-token latency** (ITL) measures the wait between adjacent outputs of one request. If token $i$ appears at time $t_i$ at the same observation point, then
<!-- bilingual-en:end -->

$$
\mathrm{ITL}_i=t_i-t_{i-1},\qquad i=2,\ldots,N.
$$

这是每个间隔一个值。[[平均输出Token时间]]则把一条请求的多个间隔合成一个平均值；两者的样本单位不同。第一个 token 之前的等待由[[首Token延迟]]记录。

<!-- bilingual-en:start -->
This produces one value per gap. [[平均输出Token时间|Time per output token]] instead averages several gaps into one value per request, so the sampling units differ. [[首Token延迟|Time to first token]] records the initial wait.
<!-- bilingual-en:end -->

## 30 ms 与 60 ms 不能合成没有停顿的体验

<!-- bilingual-en:start -->
*A 30 ms gap and a 60 ms gap remain different waits*
<!-- bilingual-en:end -->

若三个 token 分别在 120、150、210 ms 到达客户端，则两个 ITL 为 30、60 ms。平均虽为 45 ms，第二段实际等待仍是第一段的两倍。间隔可能包含[[增量解码]]计算、其他请求造成的调度间歇、抢占与网络缓冲，因此不能由客户端 ITL 单独反推出一个 GPU kernel 的耗时。

<!-- bilingual-en:start -->
If three tokens arrive at the client at 120, 150, and 210 ms, the gaps are 30 and 60 ms. Their 45 ms average does not erase the longer second wait. A gap can include [[增量解码|incremental decoding]], scheduling gaps caused by other requests, preemption, and network buffering. Client ITL alone cannot isolate GPU-kernel runtime.
<!-- bilingual-en:end -->

## 多 token 一起到达时只能声明估计口径

<!-- bilingual-en:start -->
*Bundled delivery requires an explicit estimation convention*
<!-- bilingual-en:end -->

若第一响应在 120 ms 含一个 token，第二响应在 210 ms 含两个新 token，客户端只知道两个响应相隔 90 ms。GenAI-Perf 对后一个响应采用 $90/2=45$ ms/token 的归一化间隔；这不能确定服务器上两个 token 的实际生成时点，也不表示其中一段真实间隔必为零。

<!-- bilingual-en:start -->
Suppose the first response contains one token at 120 ms and the second contains two new tokens at 210 ms. The client observes a 90 ms response gap. GenAI-Perf normalizes this by the second response's two tokens, giving 45 ms/token. This does not recover their server generation times or establish a zero gap between them.
<!-- bilingual-en:end -->

报告 ITL 的均值或分位数时，需同时保留客户端/服务端观测位置、逐 token/逐响应单位及[[Token口径|新增 token 计数]]规则。response 归一化 ITL 的 p99、逐 token ITL 的 p99、每请求平均值的 p99 是不同统计对象。

<!-- bilingual-en:start -->
ITL means and percentiles need the observation point, token-versus-response sampling unit, and [[Token口径|new-token counting rule]]. The p99 of response-normalized gaps, individual-token gaps, and request averages describes different statistical objects.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NVIDIA GenAI-Perf, Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html#metrics)：支持相邻响应间隔除以后一个响应新 token 数的工具定义。90/2 示例据此独立计算。
  <!-- bilingual-en:start -->
  Specifies normalization of response gaps by the later response's new-token count. The 90/2 example follows that rule.
  <!-- bilingual-en:end -->
- [vLLM, Metrics](https://docs.vllm.ai/en/latest/design/metrics/)，Interval Calculations、Interval Calculations vs Preemptions：支持服务端相邻 NEW_TOKENS 事件与抢占进入间隔的口径。
  <!-- bilingual-en:start -->
  Documents intervals between server NEW_TOKENS events and the inclusion of preemption in those intervals.
  <!-- bilingual-en:end -->
