---
aliases:
  - "在至多 I(1) 系统中 Pi 的秩区分无协整约化秩与水平平稳"
  - VECM Pi rank cases
  - Pi 秩的三种情形
student_os: knowledge-atom
atom_id: TS-CI-017
atom_set: cointegration-error-correction
atom_type: classification
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR到VECM重参数化]]"
  - "[[协整]]"
related:
  - "[[αβ分解非唯一性]]"
  - "[[标准协整流程边界]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# 在至多 I(1) 系统中 Pi 的秩区分无协整约化秩与水平平稳
<!-- bilingual-en:start -->
*In a system that is at most I(1), the rank of Pi separates no cointegration, reduced rank, and stationary levels*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 对满足标准正则条件且变量至多为 $I(1)$ 的 $n$ 维 VECM，$\operatorname{rank}(\Pi)$ 有三种意义：0 表示没有水平误差修正；$0<r<n$ 表示有 $r$ 条协整关系；$r=n$ 表示水平向量本身平稳。

- $r=0$：$\Pi=0$，一阶差分模型不遗漏任何线性水平约束；系统有 $n$ 个随机趋势。
- $0<r<n$：$\Pi$ 约化秩，可写成 $\alpha\beta'$；系统有 $r$ 条稳定组合和 $n-r$ 个共同随机趋势。
- $r=n$：$\Pi$ 可逆，水平 $x_t$ 为 $I(0)$；此时不应继续把所有分量称为 $I(1)$ 协整变量。

这张分类卡的前提不能省。若系统含 $I(2)$、爆炸根、季节根、退化创新或未建模确定性趋势，单看 $\Pi$ 的普通秩不足以套用上述解释。

> [!question]- 自检
> Johansen 程序选出满秩 $r=n$，最直接的建模含义是什么？
>
> **答案：** 在当前规格和标准前提下，水平向量应作为平稳 VAR 处理，而不是非平凡协整 VECM。

## 来源与核验

- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对三种秩情形。
- [[01_Math/06_时间序列分析/07_协整和误差修正模型.md]]：对照课程的秩解释。
