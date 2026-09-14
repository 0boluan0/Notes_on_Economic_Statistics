---
aliases:
  - "Dense 与 MoE 公平比较必须对齐训练数据计算硬件与质量目标"
  - Fair dense versus MoE comparison aligns data compute hardware and quality targets
  - Dense 与 MoE 公平比较
student_os: knowledge-atom
atom_id: LLM-MOE-015
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# Dense 与 MoE 公平比较必须对齐训练数据计算硬件与质量目标

<!-- bilingual-en:start -->
*A fair dense-versus-MoE comparison aligns training data, compute, hardware, and quality targets*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> dense 与 MoE 没有脱离目标的单一“公平规模”。样本效率问题应对齐数据分布和 token/训练进度；计算效率问题固定总训练 FLOPs，但可以让模型—数据分配成为研究对象；墙钟问题固定硬件、拓扑和时间；部署问题则固定质量门槛与 serving 约束。精度、batch、实现成熟度和评测协议都要报告，但不是每种研究问题都要把所有变量同时设成相等。
>
> <!-- bilingual-en:start -->
> There is no goal-independent, uniquely “fair size” for comparing dense and MoE models. A sample-efficiency question aligns the data distribution and token budget or training progress; a compute-efficiency question fixes total training FLOPs while allowing model–data allocation to be studied; a wall-clock question fixes hardware, topology, and time; a deployment question fixes a quality threshold and serving constraints. Precision, batch, implementation maturity, and evaluation protocol must be reported, but not every question holds every variable equal at once.
> <!-- bilingual-en:end -->

## 自然解释

若研究问题是“相同 token 能学多好”，先固定数据组成与 token/step 口径；若问“同样算术预算能达到多好”，做 total-FLOP-matched 比较，并允许不同模型选择不同 token 数；若问“这套集群一周能训练到什么质量”，看固定硬件下的 wall-clock 与 time-to-quality；若问“线上能否部署”，还要约束系统总权重、单设备显存、并发、延迟和通信。这些实验可能给出不同赢家，却并不互相矛盾。

<!-- bilingual-en:start -->
For “how much can each model learn from the same number of tokens?”, first match the data mixture and the token or step convention. For “how much quality can the same arithmetic budget buy?”, match total training FLOPs while allowing models to choose different token counts. For “what quality can this cluster reach in a week?”, measure wall-clock and time to quality on fixed hardware. Deployability additionally constrains system-wide weights, per-device memory, concurrency, latency, and communication. These experiments can name different winners without contradiction.
<!-- bilingual-en:end -->

MoE 侧还应报告 expert 数、$k$、capacity factor、drop 或 dropless policy、负载损失和路由布局；两侧都要报告训练 token 与数据组成、tokenizer、序列长度、优化配方和任务质量。只把“1T MoE”与“70B dense”两个 marketing 数字并列，无法判断容量、训练投入或部署成本中的任何一个。

<!-- bilingual-en:start -->
The MoE side should additionally report expert count, $k$, capacity factor, drop or dropless policy, load loss, and routing layout. Both sides need training tokens and data mixture, tokenizer, sequence length, optimisation recipe, and task quality. Juxtaposing “1T MoE” and “70B dense” marketing figures answers none of capacity, training investment, or deployment cost by itself.
<!-- bilingual-en:end -->

Unified Scaling Laws 的实验把不同 $N$ 与 $E$ 的模型都训练到 130B token，再拟合 routed model 的缩放关系。它支持“在这个固定 token 预算下，base model size 与 expert 数是不同缩放轴”，却没有同时求解 compute-optimal 的模型规模—token 分配；不能把这组固定数据量实验直接当成 Chinchilla 式计算最优结论。

<!-- bilingual-en:start -->
The Unified Scaling Laws experiments trained models with different $N$ and $E$ for a fixed 130B tokens before fitting routed-model scaling. This supports treating base-model size and expert count as separate scaling axes at that token budget, but it does not jointly solve compute-optimal allocation between model size and tokens. The fixed-data experiments therefore cannot be read as a Chinchilla-style compute-optimal result.
<!-- bilingual-en:end -->

> [!warning] 边界
> matched FLOPs 也不是天然完美：不同 kernel 和通信使同 FLOPs 的时间不同；matched wall-clock 又会混入实现优化程度。公平不是找一个神奇数字，而是让比较条件与实际决策问题一致，并公开剩余差异。
>
> <!-- bilingual-en:start -->
> FLOP matching is not inherently perfect because kernels and communication give equal FLOPs different runtimes; wall-clock matching can confound implementation maturity. Fairness means matching the comparison to the real decision and disclosing remaining differences, not finding one magical scalar.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 一篇论文说 MoE 在“相同计算”下优于 dense。阅读方法部分至少要确认哪三件事？
>
> **答案：** “计算”是每 token FLOPs、总训练 FLOPs 还是设备时间；token/数据是固定条件还是允许优化的变量；质量评测、硬件、batch、精度、路由与通信实现是否与该研究问题可比。

## 来源与核验

- Clark et al. (2022), [*Unified Scaling Laws for Routed Language Models*](https://proceedings.mlr.press/v162/clark22a.html)，§4.4–5：在固定 130B token 的实验上下文中建模 base model size、expert count、每输入计算与总参数，并明确其系数和边界依赖这一 token 数；该论文不提供 compute-optimal token 分配。
- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，表 1 与 §3：展示 FLOP-matched、相同硬件、实际速度与质量门槛等不同比较口径。
