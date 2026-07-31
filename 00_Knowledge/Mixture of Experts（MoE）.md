---
aliases:
  - "Mixture of Experts"
  - "Sparse Mixture of Experts"
  - "专家混合模型"
status: source-checked
---

# Mixture of Experts（MoE）
<!-- bilingual-en:start -->
*Mixture of Experts (MoE)*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在不让每个 token 经过全部参数的前提下扩大模型容量，使总参数增长与单 token 计算部分解耦。
> **具体锚点：** 一个 token 可能只被 router 送到 64 个 expert 中的两个；其余 expert 对该 token 不计算。
> **核心难点：** 稀疏激活节省的是算术计算，不会自动消除路由不均、容量溢出、跨设备通信、训练不稳定和大权重存储。
> **为什么重要：** MoE 模型的“总参数”“激活参数”“每 token FLOPs”是三种不同口径，不能互换。
> **继续：** 先理解 top-k routing 与负载均衡，再读 [[大模型分布式训练]] 的 expert parallel；预算比较见 [[Scaling laws 与计算最优训练]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Expand model capacity without sending every token through every parameter, partially decoupling total parameter count from per-token computation.
> **Concrete anchor:** A router may send one token to only two of 64 experts; the other experts perform no computation for that token.
> **Central difficulty:** Sparse activation saves arithmetic but does not automatically remove routing imbalance, capacity overflow, cross-device communication, training instability, or storage of the full weight set.
> **Why it matters:** Total parameters, active parameters, and per-token FLOPs are distinct quantities for an MoE model and cannot be interchanged.
> **Continue with:** Understand top-k routing and load balancing below, then use [[大模型分布式训练|Distributed Training for Large Models]] for expert parallelism and [[Scaling laws 与计算最优训练|Scaling Laws and Compute-Optimal Training]] for budget comparison.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - Switch Transformer 与稀疏门控 expert 原论文：核验 top-k routing、capacity、负载均衡和条件计算。
> <!-- bilingual-en:start -->
> - The Switch Transformer and sparsely gated expert papers verify top-k routing, capacity, load balancing, and conditional computation.
> <!-- bilingual-en:end -->

## 稀疏路由机制
<!-- bilingual-en:start -->
*Sparse Routing Mechanism*
<!-- bilingual-en:end -->

Mixture of Experts 用 router 为每个 token 选择少数前馈 expert。总参数可很大而每 token 只激活一部分。主要问题是负载均衡、expert capacity、token dropping、跨设备 all-to-all 通信与训练稳定性；“总参数”尤其不能直接代表单次推理计算。
<!-- bilingual-en:start -->
A Mixture of Experts uses a router to select a small number of feed-forward experts for each token. Total parameters can be large while each token activates only a subset. Major problems include load balancing, expert capacity, token dropping, cross-device all-to-all communication, and training stability; total parameters especially do not directly represent inference computation per token.
<!-- bilingual-en:end -->

若 hidden state 为 $h$，router 产生 logits $Wh$，再用 softmax 得到 expert 权重并选择 top-$k$。输出可写成 $y=\sum_{i\in\operatorname{TopK}(h)}g_i(h)E_i(h)$。router 与 expert 一起训练，因此“专家分工”是涌现结果，不是预先固定的学科标签。
<!-- bilingual-en:start -->
Given hidden state $h$, a router produces logits $Wh$, applies softmax to obtain expert weights, and selects the top $k$. The output can be written $y=\sum_{i\in\operatorname{TopK}(h)}g_i(h)E_i(h)$. Router and experts train jointly, so specialization emerges rather than following predetermined subject labels.
<!-- bilingual-en:end -->

## 容量、负载均衡与通信
<!-- bilingual-en:start -->
*Capacity, Load Balancing, and Communication*
<!-- bilingual-en:end -->

每个 expert 在一个 batch 中可接收的 token 数常由 capacity factor 限制。若 router 把太多 token 送往同一 expert，溢出 token 可能被丢弃、绕过或排队；辅助负载均衡损失鼓励利用更均匀，但过强会妨碍有意义的专门化。
<!-- bilingual-en:start -->
An expert's token capacity within a batch is often limited by a capacity factor. If the router sends too many tokens to one expert, overflow tokens may be dropped, bypassed, or queued. An auxiliary load-balancing loss encourages more even use, but excessive pressure can obstruct meaningful specialization.
<!-- bilingual-en:end -->

expert 分散在不同设备时，每层都可能先 all-to-all 发送 token，再把 expert 输出送回原设备。低 batch、不均路由或慢网络会让通信压过节省的矩阵计算；因此理论 active FLOPs 低不保证端到端吞吐高。
<!-- bilingual-en:start -->
When experts reside on different devices, each layer may use all-to-all communication to send tokens to experts and return outputs. Small batches, imbalanced routing, or a slow network can make communication dominate the saved matrix computation, so low theoretical active FLOPs do not guarantee high end-to-end throughput.
<!-- bilingual-en:end -->

## Worked example：容量溢出
<!-- bilingual-en:start -->
*Worked Example: Capacity Overflow*
<!-- bilingual-en:end -->

设一个 batch 有 1,024 个 token、16 个 expert、top-1 routing。均匀负载期望每个 expert 64 个 token；capacity factor 为 1.25 时容量是 80。若某 expert 收到 140 个 token，就有 60 个超出容量。此时平均利用率看似足够，局部热点仍会丢信息或制造延迟。
<!-- bilingual-en:start -->
Suppose a batch contains 1,024 tokens, 16 experts, and top-1 routing. Uniform load would assign 64 tokens per expert; a capacity factor of 1.25 gives capacity 80. If one expert receives 140 tokens, 60 exceed capacity. Average system utilization may look adequate while a local hotspot still drops information or creates delay.
<!-- bilingual-en:end -->

将 top-1 改为 top-2 可提高冗余与表达能力，却大约增加 expert 计算和通信，并使容量核算按两次派发进行。选择 $k$ 是质量、稳定性和系统成本的权衡，不是越大越好。
<!-- bilingual-en:start -->
Changing top-1 to top-2 can improve redundancy and expressiveness, but roughly increases expert computation and communication and counts each token twice for capacity. Choosing $k$ trades quality and stability against system cost; larger is not automatically better.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 某些 expert 几乎不用：查看 router 概率、token 分布和辅助损失，区分健康专门化与 expert collapse。
  <!-- bilingual-en:start -->
  Some experts are almost unused: inspect router probabilities, token distribution, and auxiliary loss to distinguish healthy specialization from expert collapse.
  <!-- bilingual-en:end -->
- 训练 loss 抖动并出现 token drop：检查 capacity factor、batch 组成、router 数值范围和负载均衡，而不是只降学习率。
  <!-- bilingual-en:start -->
  Training loss oscillates with token drops: inspect capacity factor, batch composition, router numerical range, and load balancing rather than only lowering the learning rate.
  <!-- bilingual-en:end -->
- active FLOPs 低但延迟高：分解 all-to-all 时间、kernel 小矩阵效率、权重加载和路由开销。
  <!-- bilingual-en:start -->
  Active FLOPs are low but latency is high: decompose all-to-all time, small-matrix kernel efficiency, weight loading, and routing overhead.
  <!-- bilingual-en:end -->
- 用总参数与 dense 模型横向比较：同时报告 active parameters、每 token FLOPs、显存、硬件拓扑和质量。
  <!-- bilingual-en:start -->
  Total parameters are compared directly with a dense model: report active parameters, per-token FLOPs, memory, hardware topology, and quality as well.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### MoE 的“参数多、计算省”具体是什么意思？
<!-- bilingual-en:start -->
*What exactly does “more parameters, less computation” mean for MoE?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 总 expert 参数很多，但 router 对每个 token 只激活少数 expert；代价转向路由、容量和通信。
> <!-- bilingual-en:start -->
> The total expert parameter set is large, but the router activates only a few experts for each token; costs shift toward routing, capacity, and communication.
> <!-- bilingual-en:end -->

### 为什么负载均衡损失不能无限增大？
<!-- bilingual-en:start -->
*Why can the load-balancing loss not be increased without limit?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 强制完全均匀会压制由数据产生的有意义专门化，而且辅助目标可能与主语言建模目标冲突。
> <!-- bilingual-en:start -->
> Enforcing perfect uniformity suppresses meaningful data-driven specialization, and the auxiliary objective can conflict with the main language-modeling objective.
> <!-- bilingual-en:end -->

### 哪些指标能区分算术节省与系统真正加速？
<!-- bilingual-en:start -->
*Which metrics distinguish arithmetic savings from genuine system speedup?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 同时报 active FLOPs、路由分布、token drop、all-to-all 时间、设备利用率、端到端吞吐/延迟和质量。
> <!-- bilingual-en:start -->
> Report active FLOPs, routing distribution, token drops, all-to-all time, device utilization, end-to-end throughput or latency, and quality together.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [Fedus, Zoph, and Shazeer (2022), Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)：核验 top-1 routing、capacity、负载均衡、训练稳定性与稀疏计算的实验依据。
  <!-- bilingual-en:start -->
  [Fedus, Zoph, and Shazeer (2022), Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html) verifies top-1 routing, capacity, load balancing, training stability, and the experimental basis for sparse computation.
  <!-- bilingual-en:end -->
- [Shazeer et al. (2017), Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)：核验稀疏门控 expert、条件计算与负载均衡的早期机制。
  <!-- bilingual-en:start -->
  [Shazeer et al. (2017), Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) verifies the early mechanism of sparsely gated experts, conditional computation, and load balancing.
  <!-- bilingual-en:end -->
