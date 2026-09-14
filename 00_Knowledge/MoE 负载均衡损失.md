---
aliases:
  - "负载均衡辅助损失鼓励使用分散但不保证语义专门化"
  - Load-balancing auxiliary losses encourage distributed use but do not guarantee semantic specialisation
  - MoE 负载均衡损失
student_os: knowledge-atom
atom_id: LLM-MOE-009
atom_type: optimization-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 负载均衡辅助损失鼓励使用分散但不保证语义专门化

<!-- bilingual-en:start -->
*Load-balancing auxiliary losses encourage distributed use but do not guarantee semantic specialisation*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> MoE 常在主任务损失之外加入辅助项，惩罚 router 把概率或 assignment 长期集中到少数 expert。它提供“多用几台 expert”的训练压力，却只是负载代理目标：不能保证每个 batch 精确均匀，也不能证明 expert 学到了互补、可解释的分工。
>
> <!-- bilingual-en:start -->
> MoE systems often add an auxiliary term to the main task loss to discourage router probabilities or assignments from concentrating on a few experts. It provides training pressure to use more experts, but remains a proxy for load: it guarantees neither exact balance in every batch nor complementary, interpretable expert specialisation.
> <!-- bilingual-en:end -->

## 自然解释

Switch Transformer 对每层统计 $f_i$（实际送往 expert $i$ 的 token 比例）与 $P_i$（router 分给该 expert 的平均概率），使用 $\alpha E\sum_i f_iP_i$。理想均匀时二者都约为 $1/E$；其中 $P_i$ 可微，因此辅助项能推动 router 改变概率，即使离散 assignment 本身不可微。

<!-- bilingual-en:start -->
Switch Transformer measures $f_i$, the fraction of tokens dispatched to expert $i$, and $P_i$, the average router probability allocated to it, using $\alpha E\sum_i f_iP_i$. Under ideal uniform use, both are about $1/E$. Since $P_i$ is differentiable, the auxiliary term can move router probabilities even though discrete assignments are not.
<!-- bilingual-en:end -->

系数 $\alpha$ 太小，可能不足以阻止热点；太大，则辅助目标会压过语言建模或其他主任务。更重要的是，“均匀”只约束使用量，不规定每个 expert 应该学什么。正确评估应把主任务质量、负载分布、overflow 和[[MoE 专家分工涌现|专门化证据]]分开报告。

<!-- bilingual-en:start -->
If coefficient $\alpha$ is too small, it may not prevent hotspots; if too large, the auxiliary objective can overwhelm language modelling or another primary task. More importantly, balance constrains usage volume, not what an expert should learn. Evaluation should separately report task quality, load distribution, overflow, and [[MoE 专家分工涌现|evidence of specialisation]].
<!-- bilingual-en:end -->

> [!warning] 边界
> 辅助损失的具体公式并不统一。Shazeer 的早期设计区分 importance 与 load，Switch 使用简化点积，其他路由器可能采用完全不同的约束；不能只说“有 load-balancing loss”就假定行为相同。
>
> <!-- bilingual-en:start -->
> The auxiliary formula is not universal. Shazeer's early design separates importance and load, Switch uses a simplified dot product, and other routers may impose different constraints. Merely stating “a load-balancing loss is present” does not make their behaviour equivalent.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 所有 expert 的 token 数完全相等，能否据此断言它们已经形成最佳分工？
>
> **答案：** 不能。相等只说明使用量；expert 可能学到重复功能，而且辅助系数若大到压过主任务会损害质量。无论哪一种，都不能由 token 计数推出互补分工。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，式 (4)–(6)：定义 $f_i$、$P_i$、辅助损失及系数与主目标的取舍。
- Shazeer et al. (2017), [*Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538)，§4 与 Appendix A：给出 importance 与 load 两类均衡目标，并把它们定位为缓解路由不均的辅助项。
- Zoph et al. (2022), [*ST-MoE*](https://arxiv.org/abs/2202.08906)，Appendix A：明确辅助系数需足以促进均衡、又不能大到压过主交叉熵目标。
