---
aliases:
  - "VECM 中 beta 张成协整空间而 alpha 记录各变量调整载荷"
  - Alpha and beta in a VECM
  - VECM 调整矩阵与协整空间
student_os: knowledge-atom
atom_id: TS-CI-008
atom_set: cointegration-error-correction
atom_type: parameter-meaning
status: source-checked
mastery_state: unassessed
requires:
  - "[[ECM长短期结构]]"
  - "[[协整秩与共同趋势]]"
related:
  - "[[αβ分解非唯一性]]"
  - "[[弱外生性与调整载荷]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# VECM 中 beta 张成协整空间而 alpha 记录各变量调整载荷
<!-- bilingual-en:start -->
*In a VECM, beta spans the cointegration space while alpha records each variable's adjustment loadings*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 当 $\Pi=\alpha\beta'$ 且秩为 $r$ 时，$\beta'x_{t-1}$ 给出 $r$ 个长期偏离，$\alpha$ 的每一行说明相应变量的差分方程怎样响应这些偏离。

若 $x_t$ 有 $n$ 个分量，则 $\alpha$ 与 $\beta$ 都是 $n\times r$。$\beta$ 的列不是“每个变量的调整速度”，而是形成平稳组合的系数；$\alpha$ 的列也不是新的长期关系，而是每一条关系进入各方程的载荷。两者相乘后，$\Pi x_{t-1}=\alpha(\beta'x_{t-1})$ 才是水平信息对本期变化的完整作用。

解释时应按顺序问：这条 $\beta$ 在当前归一化下测量什么偏离？哪个 $\alpha_{ij}$ 把该偏离送入第 $i$ 个变量的方程？其符号是否产生修正？短期差分项会不会同时推动变量？把 $\alpha$ 和 $\beta$ 混成一组“长期系数”，会同时丢掉关系与调整两个层次。

> [!question]- 自检
> 在三变量、两条协整关系的 VECM 中，$\alpha$ 的第二行表示什么？
>
> **答案：** 第二个变量的差分方程对两条长期偏离各自承担多大调整载荷。

## 来源与核验

- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对约化秩分解与参数角色。
- [[01_Math/06_时间序列分析/07_协整和误差修正模型.md]]：对照课程的 $\Pi=\alpha\beta'$ 表示。
