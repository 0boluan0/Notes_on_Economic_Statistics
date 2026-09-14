---
aliases:
  - "线性 ARCH-GARCH 的非负参数是保证方差非负的常用充分条件"
  - GARCH positivity constraints
  - ARCH positivity restrictions
  - 条件方差正性约束
student_os: knowledge-atom
atom_id: TS-VOL-004
atom_set: conditional-volatility
atom_type: condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH(p,q)模型]]"
related:
  - "[[GJR-GARCH模型]]"
  - "[[EGARCH模型]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# 线性 ARCH-GARCH 的非负参数是保证方差非负的常用充分条件
<!-- bilingual-en:start -->
*Nonnegative coefficients are standard sufficient positivity restrictions for linear ARCH-GARCH models*
<!-- bilingual-en:end -->

> [!summary] 原子条件
> 对线性 ARCH/GARCH 方差递推，$\omega>0$、$\alpha_i\ge0$、$\beta_j\ge0$ 可保证任意历史下 $h_t>0$。这些是简单、可实施的**充分条件**；它们不是所有非线性波动模型共同的必要条件，也不是平稳性条件。

若只取 $\omega=0$，其余系数非负通常只能保证 $h_t\ge0$。例如零初值配合连续零冲击会把递推留在 $h_t=0$，此时 $z_t=\varepsilon_t/\sqrt{h_t}$ 无法定义；因此“非负”与“严格为正”不能混写。$\omega<0$ 也并非仅靠其他系数非负就能对任意历史保证合法方差。

三类限制必须分开：

- **正性：** 每条允许的路径上 $h_t$ 不能为负；
- **严格平稳：** 整个联合分布对时间平移不变；
- **矩存在：** 例如 $E(\varepsilon_t^2)<\infty$。

EGARCH 递推的是 $\log h_t$，指数化后自然为正，因此其有符号系数不必全非负。[[GJR-GARCH模型|GJR-GARCH]] 中负冲击的平方系数是 $\alpha+\gamma$，所以常用正性条件允许 $\gamma<0$，只要 $\alpha+\gamma\ge0$。直接把标准 GARCH 的参数框复制给扩展模型会误删合法规格。

> [!question]- 自检
> 一个 GARCH(1,1) 满足 $\omega>0,\alpha,\beta\ge0$，能否仅据此宣布它有有限长期方差？
>
> **答案：** 不能。正性只保证 $h_t$ 不为负；有限二阶矩还要求 $\alpha+\beta<1$（在标准化冲击口径下）。

## 来源与核验

- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：核对线性 GARCH 的标准非负参数约束。
- [Nelson (1991)](https://doi.org/10.2307/2938260)：核对 EGARCH 通过对数方差避免线性正性约束的设计。
- [Glosten, Jagannathan & Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)：核对符号相关平方项的模型边界。
