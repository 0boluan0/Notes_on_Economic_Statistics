---
aliases:
  - FlashAttention-2 是通过调整分块并行与局部计算改善 GPU 执行的精确注意力算法
  - FlashAttention-2 is an exact attention algorithm that improves GPU execution through block parallelism and local computation changes
  - FlashAttention-2
student_os: knowledge-atom
atom_id: LLM-INF-038
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# FlashAttention-2 是通过调整分块并行与局部计算改善 GPU 执行的精确注意力算法
<!-- bilingual-en:start -->
*FlashAttention-2 is an exact attention algorithm that improves GPU execution through block parallelism and local computation changes*
<!-- bilingual-en:end -->

FlashAttention-2 保留 [[FlashAttention|分块精确注意力]]，进一步调整 GPU 上的工作划分和归一化计算。它针对的是“已经少搬了数据，但 GPU 仍没有高效执行”的问题：计算块可能太少，块内协作也可能产生多余的共享内存读写。

<!-- bilingual-en:start -->
FlashAttention-2 retains [[FlashAttention|tiled exact attention]] while changing GPU work partitioning and normalization. It addresses inefficiency that remains after reducing device-memory traffic, including too few parallel blocks and unnecessary communication within a block.
<!-- bilingual-en:end -->

前向计算中，一组 query 的输出不依赖另一组 query 的输出；每行仍需遍历自己的全部可见 K/V。因此可以让不同 GPU thread block 各负责一段 query 行，除了 batch 和 head 之外，再利用序列方向提供并行工作。thread block 是 GPU 的协作执行单位，不是 [[PagedAttention|KV 的物理存储块]]。

<!-- bilingual-en:start -->
In the forward pass, one group of query outputs does not depend on another. Each row still traverses all its visible K/V, allowing separate GPU thread blocks to own query-row ranges. This adds sequence parallelism beyond batch and heads. A thread block is an execution unit, not a [[PagedAttention|physical KV storage block]].
<!-- bilingual-en:end -->

在一个 thread block 内，论文也让不同 warp（较小的线程协作组）分别负责不同 query 行，并共享访问 K/V。这样每个 warp 得到自己完整的输出行，不必像分割 key 范围那样，先算同一行的多份局部输出再跨 warp 合并。原文的这一“无需合并”针对该前向划分，不代表整个算法或反向传播没有同步。

<!-- bilingual-en:start -->
Within a thread block, warps own different query rows while accessing shared K/V. Each warp produces its own complete output rows rather than partial results for the same row that require cross-warp reduction. This communication saving concerns the stated forward partition; it does not remove all synchronization from the algorithm or backward pass.
<!-- bilingual-en:end -->

例如四行 query 对四列 key：按 query 行分给两组，每组各算两行对四列的读取，并直接交付自己的两行输出。若按 key 列分给两组，两组都会得到四行的局部贡献，之后还要按 [[在线Softmax合并|共同归一化规则]] 合并。这是依赖关系示意，不是实际内核尺寸或速度测试。

<!-- bilingual-en:start -->
For four query rows and four keys, two workers split by query can each produce two complete output rows. Split by keys instead, both produce partial contributions for all four rows that require [[在线Softmax合并|consistent normalization and merging]]. This illustrates dependencies, not a real kernel size or benchmark.
<!-- bilingual-en:end -->

算法还减少非矩阵乘法工作：保留未归一化累计输出，到最后再除以归一化总和，而非每一步重复除法。延后的是除以总和；运行最大值增大时，旧累计输出仍须按 $\exp(m_{\rm old}-m_{\rm new})$ 缩放到共同尺度，见 [[在线Softmax合并]]。执行单元、head 宽度、序列长度和并行任务数仍影响收益；“第二版”不等于所有设备、所有单 token decode 都有固定提速，见 [[推理性能可比性]]。

<!-- bilingual-en:start -->
The algorithm also reduces non-matmul work by retaining unnormalized accumulated output until final normalization. Only division by the total is postponed: when the running maximum increases, the old accumulator must still be rescaled by $\exp(m_{\rm old}-m_{\rm new})$, as derived in [[在线Softmax合并|online softmax merging]]. Execution units, head dimensions, sequence length, and available parallel work still determine the benefit. A new algorithm version does not imply a fixed speedup for every device or single-token decode shape; use [[推理性能可比性|matched performance comparisons]].
<!-- bilingual-en:end -->

## 来源与核验

- [Dao (2023), *FlashAttention-2*](https://arxiv.org/html/2307.08691v1)，§3.1.1、§3.2、§3.3 及 Figures 2–3：支持延后归一化、按 query 行安排前向 thread block、warp 内工作划分及其同步边界。四行例子是这些依赖关系的缩小示意。
- [Dao et al. (2022), *FlashAttention*](https://arxiv.org/html/2205.14135v2)，§3.1：支持共同指数尺度的重标定；具体代数见 [[在线Softmax合并]]。FA2 v1 Algorithm 1 第 10 行的逆因子与这一恒等式不一致，本卡使用可直接推导核验的缩放因子。

<!-- bilingual-en:start -->
The cited sections and figures support delayed normalization, query-row forward parallelism, warp partitioning, and synchronization scope. FlashAttention §3.1 supports rescaling to a common exponential scale; the derivation is in [[在线Softmax合并|online softmax merging]]. The inverse factor printed in FA2 v1 Algorithm 1, line 10 is inconsistent with that identity; this card uses the directly verifiable scaling. The four-row example is a reduced dependency illustration.
<!-- bilingual-en:end -->
