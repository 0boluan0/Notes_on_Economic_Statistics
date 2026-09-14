---
aliases:
  - 平均输出 token 时间用首 token 之后的生成时段除以后续输出 token 数
  - Time per output token divides the interval after the first token by the number of subsequent output tokens
  - TPOT
student_os: knowledge-atom
atom_id: LLM-INF-005
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 平均输出 token 时间用首 token 之后的生成时段除以后续输出 token 数

<!-- bilingual-en:start -->
*Time per output token divides the interval after the first token by the number of subsequent output tokens*
<!-- bilingual-en:end -->

**平均输出 token 时间**（time per output token，TPOT）在本卡采用每请求的流式平均口径：从首 token 到末 token 的经过时间，除以首 token 之后的输出数。它回答“开始输出后，平均等多久得到下一个 token”。

<!-- bilingual-en:start -->
This note defines **time per output token** (TPOT) as a per-request streaming average: elapsed time from the first to the final token divided by the number of outputs after the first. It asks how long the next token takes on average once output has begun.
<!-- bilingual-en:end -->

设同一观测点记录 $N>1$ 个输出的时刻 $t_1,\ldots,t_N$。按[[Token间隔]]定义，时间差相加后中间时刻抵消，因此

<!-- bilingual-en:start -->
Let $t_1,\ldots,t_N$ record $N>1$ outputs at one observation point. Summing the [[Token间隔|inter-token intervals]] cancels intermediate timestamps, giving
<!-- bilingual-en:end -->

$$
\mathrm{TPOT}
=\frac{t_N-t_1}{N-1}
=\frac{1}{N-1}\sum_{i=2}^{N}\mathrm{ITL}_i.
$$

首 token 已由[[预填充]]预测，所以分母是后续的 $N-1$ 个 token。[[首Token延迟]]没有被平均到这段流式速度里。若 $N=1$，没有后续间隔，数学上的 TPOT 未定义；工具采用排除、N/A 或约定为零时应注明。

<!-- bilingual-en:start -->
The first token is already predicted by [[预填充|prefill]], leaving $N-1$ subsequent outputs in the denominator. [[首Token延迟|Time to first token]] is excluded from this streaming-speed average. When $N=1$, there is no subsequent gap and mathematical TPOT is undefined; any tool convention that excludes it, returns N/A, or assigns zero must be stated.
<!-- bilingual-en:end -->

## 每请求平均与每间隔平均的权重不同

<!-- bilingual-en:start -->
*Request averages and pooled gaps use different weights*
<!-- bilingual-en:end -->

一个请求在 120、150、210 ms 得到三个输出，TPOT 为 $(210-120)/2=45$ ms。再考虑两条不同请求：A 只有一个 100 ms 间隔；B 有九个 10 ms 间隔。两请求 TPOT 分别是 100 和 10 ms，所以等权平均为 55 ms；把十个 ITL 合并后平均却是

<!-- bilingual-en:start -->
A request with outputs at 120, 150, and 210 ms has TPOT $(210-120)/2=45$ ms. Now consider request A with one 100 ms gap and request B with nine 10 ms gaps. Their TPOT values are 100 and 10 ms, giving a request-weighted mean of 55 ms. Pooling all ten ITLs instead gives
<!-- bilingual-en:end -->

$$
\frac{100+9\times10}{10}=19\ \mathrm{ms}.
$$

因此，平均 TPOT 不等于合并所有 token 间隔后的平均 ITL，分位数也不能直接互换。

<!-- bilingual-en:start -->
Mean request TPOT therefore differs from the mean of pooled ITLs, and their percentiles are not interchangeable either.
<!-- bilingual-en:end -->

## 工具的结束边界可能不同

<!-- bilingual-en:start -->
*Tools may use a different ending event*
<!-- bilingual-en:end -->

vLLM v0.8.5 的 serving benchmark 用 `(latency - ttft) / (output_len - 1)` 计算 TPOT。如果其中 latency 直到最终响应才停止，而最后 token 更早到达，结果也会摊入末 token 后的处理尾差。阅读结果时应核对[[请求完成延迟|完成事件]]；有些文档还直接用 TPOT 指 ITL，不能仅凭缩写认定统计口径相同。

<!-- bilingual-en:start -->
The vLLM v0.8.5 serving benchmark computes TPOT as `(latency - ttft) / (output_len - 1)`. If latency ends at the final response after the last token has arrived, that trailing interval is also averaged in. Check the [[请求完成延迟|completion event]] when interpreting results. Some documentation also uses TPOT for ITL, so the abbreviation alone does not establish a shared convention.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [vLLM v0.8.5, benchmark_serving.py](https://github.com/vllm-project/vllm/blob/v0.8.5/benchmarks/benchmark_serving.py#L125-L224)，`calculate_metrics`：核对 $N-1$ 分母、每请求 TPOT 与合并 ITL 的不同列表；$N\le1$ 不进入 TPOT 统计列表，在 goodput 判断中另约定为零。本卡的时间差恒等式与55/19算例独立推导。
  <!-- bilingual-en:start -->
  The implementation verifies the $N-1$ denominator and distinct request-TPOT and pooled-ITL lists. Requests with $N\le1$ are excluded from the TPOT statistics list and assigned zero for goodput evaluation. The identity and 55/19 example are derived in this note.
  <!-- bilingual-en:end -->
