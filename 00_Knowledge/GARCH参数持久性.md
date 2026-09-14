---
aliases:
  - "alpha 与 beta 分别控制冲击反应和方差递推而其和刻画二阶持久性"
  - GARCH alpha beta persistence
  - GARCH news and persistence parameters
  - GARCH 持久性
student_os: knowledge-atom
atom_id: TS-VOL-007
atom_set: conditional-volatility
atom_type: parameter-interpretation
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH有限方差条件]]"
related:
  - "[[GARCH多步方差预测]]"
  - "[[方差断点伪GARCH持久性]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# alpha 与 beta 分别控制冲击反应和方差递推而其和刻画二阶持久性
<!-- bilingual-en:start -->
*Alpha and beta govern shock response and variance carryover, while their sum measures second-moment persistence*
<!-- bilingual-en:end -->

> [!summary] 原子解释
> 在 GARCH(1,1) 中，$\alpha$ 决定最新平方创新 $\varepsilon_t^2$ 对下一期方差的即时更新强度，$\beta$ 决定上一期条件方差 $h_t$ 被带入下一期的程度。令 $\rho=\alpha+\beta$；在有限二阶矩条件下，$\rho$ 决定**条件期望中的方差缺口**以多快速度衰减。

$\alpha$ 大不等于“总持久性一定大”，$\beta$ 大也不等于“冲击反应一定强”。两个模型可以有相同 $\rho$，却一个对新消息反应快、另一个主要延续旧方差，路径形状并不相同。

$\rho$ 还是模型内、二阶口径的持久性摘要。它不是自然界的不可变常数：样本频率、均值规格、创新分布、异常值与结构突变都会改变估计。把 $\hat\rho$ 接近一直接叫作“真实长期记忆”会越过模型证据。

> [!question]- 自检
> 模型 A 为 $(\alpha,\beta)=(0.20,0.75)$，模型 B 为 $(0.05,0.90)$。它们有相同的 $\rho$，含义是否完全相同？
>
> **答案：** 不同。两者期望方差缺口衰减率相同，但 A 对新平方冲击更敏感，B 更依赖已有方差状态。

## 来源与核验

- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：核对 GARCH 递推与二阶持久性。
- [Nelson (1990)](https://doi.org/10.1017/S0266466600005296)：核对不同“persistence”定义并不等价的边界。
