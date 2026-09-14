---
aliases:
  - "GARCH(1,1) 的平方创新具有 ARMA(1,1) 型二阶表示"
  - ARMA representation of squared GARCH innovations
  - GARCH squared-error ARMA representation
  - 平方创新 ARMA 表示
student_os: knowledge-atom
atom_id: TS-VOL-008
atom_set: conditional-volatility
atom_type: representation
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH(p,q)模型]]"
  - "[[ARMA(p,q)模型]]"
related:
  - "[[McLeod-Li检验]]"
  - "[[波动率聚集]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH(1,1) 的平方创新具有 ARMA(1,1) 型二阶表示
<!-- bilingual-en:start -->
*Squared GARCH(1,1) innovations have an ARMA(1,1)-type second-order representation*
<!-- bilingual-en:end -->

> [!summary] 原子表示
> 定义方差预测误差 $v_t=\varepsilon_t^2-h_t$。由 $E(v_t\mid\mathcal F_{t-1})=0$ 及 GARCH(1,1) 递推可得
> $$\varepsilon_t^2=\omega+(\alpha+\beta)\varepsilon_{t-1}^2+v_t-\beta v_{t-1}.$$
> 因而在相应二阶矩存在时，平方创新表现为 ARMA(1,1) 型动态；这解释了为什么原创新可不相关，而平方创新仍持续相关。

这个表示首先是代数恒等式；只要 $E(\varepsilon_t^2)<\infty$，$v_t$ 可积且满足鞅差条件。若要进一步把 $\varepsilon_t^2$ 当作具有有限方差与 ACF 的二阶 ARMA 型过程，还需 $E(\varepsilon_t^4)<\infty$。它不应被夸大为“$\varepsilon_t^2$ 是 Gaussian ARMA”：$v_t$ 通常既非同方差也非正态，高阶分布与普通线性 ARMA 不同。

该表示还说明 $\alpha+\beta$ 出现在平方过程的 AR 系数中，而 $-\beta$ 出现在 MA 项中。若 $\varepsilon_t$ 不具有限四阶矩，平方过程的 ACF 与协方差平稳解释就失去基础，即便递推恒等式形式仍能写出。

> [!question]- 自检
> 看到平方残差 ACF 拖尾，能否据此唯一识别出 GARCH(1,1)？
>
> **答案：** 不能。拖尾只与某些 GARCH 动态相容；其他非线性模型、结构突变或高阶模型也可产生类似样本图形。

## 来源与核验

- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：核对 GARCH 平方过程的 ARMA 型自相关结构。
- [[01_Math/06_时间序列分析/lecture.pdf#page=165|课程讲义 pp. 165–167]]：核对以平方残差相关诊断条件异方差的课程路线。
