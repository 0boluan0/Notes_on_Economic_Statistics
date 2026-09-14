---
aliases:
  - "只有当变量在 LPM 中以单一未变换主效应进入时，其系数才直接是一单位变化的概率斜率"
  - Interpreting LPM coefficients
student_os: knowledge-atom
atom_id: ECON-BIN-003
atom_set: binary-outcome-models
atom_type: interpretation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[LPM的OLS估计]]"
related:
  - "[[类别连续交互]]"
  - "[[线性组合推断]]"
leads_to:
  - "[[非线性边际效应]]"
  - "[[二元离散效应]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 只有当变量在 LPM 中以单一未变换主效应进入时，其系数才直接是一单位变化的概率斜率

<!-- bilingual-en:start -->
*An LPM coefficient directly gives the probability slope for a one-unit change only when the variable enters as a single, untransformed main effect.*
<!-- bilingual-en:end -->

> [!summary] 解释边界
> 在 $p(X)=\beta_0+\beta_jX_j+X_{-j}'\gamma$ 中，$X_j$ 只以一个未变换、无交互的主效应进入，才有
> $$
> \frac{\partial p(X)}{\partial X_j}=\beta_j.
> $$
> 一旦同一变量还出现在平方项、对数、样条或交互项中，必须对完整预测式求导或计算目标对比，不能只读一个系数。
> <!-- bilingual-en:start -->
> In $p(X)=\beta_0+\beta_jX_j+X_{-j}'\gamma$, the derivative shown above equals $\beta_j$ only when $X_j$ enters solely as one untransformed main effect with no interaction. Once the same variable also appears in a squared term, logarithm, spline, or interaction, one must differentiate the complete prediction equation or calculate the relevant contrast rather than read a single coefficient in isolation.
> <!-- bilingual-en:end -->

若 $X_j$ 连续且满足上述简单规格，$\beta_j=0.04$ 表示 $X_j$ 增加一单位时预测概率改变 0.04，即 4 个百分点，不是“增加 4%”。若 $D$ 是 0/1 变量，同样简单的 LPM 中

<!-- bilingual-en:start -->
If $X_j$ is continuous and satisfies the simple specification above, $\beta_j=0.04$ means that a one-unit increase in $X_j$ changes the predicted probability by 0.04, or 4 percentage points—not that it “increases by 4%.” If $D$ is a 0/1 variable, the following equality holds in the same simple LPM.
<!-- bilingual-en:end -->

$$
p(D=1,X_{-D})-p(D=0,X_{-D})=\beta_D.
$$

有交互时，单个系数只属于某个基准条件。例如

<!-- bilingual-en:start -->
With an interaction, an individual coefficient applies only under a particular reference condition. For example, consider
<!-- bilingual-en:end -->

$$
p=\beta_0+\beta_1X+\beta_2D+\beta_3XD
$$

中，$D=0$ 时 $X$ 的斜率是 $\beta_1$，$D=1$ 时是 $\beta_1+\beta_3$；给定 $X=x$ 的组差是 $\beta_2+x\beta_3$。同一变量被多次编码时，logit/probit 的系数符号也要按这条完整效应原则解释。

<!-- bilingual-en:start -->
In this model, the slope of $X$ is $\beta_1$ when $D=0$ and $\beta_1+\beta_3$ when $D=1$; at $X=x$, the difference between the two groups is $\beta_2+x\beta_3$. When the same variable is encoded in more than one term, the signs of logit or probit coefficients must likewise be interpreted through this full-effect principle.
<!-- bilingual-en:end -->

> [!question]- 自检
> 在 $p=0.2+0.03X-0.001X^2$ 中，能否把 0.03 直接称为 $X$ 的概率斜率？
>
> **答案：** 不能。完整斜率是 $0.03-0.002X$，会随 $X$ 改变。
> <!-- bilingual-en:start -->
> In $p=0.2+0.03X-0.001X^2$, can 0.03 be described directly as the probability slope of $X$?
>
> **Answer:** No. The full slope is $0.03-0.002X$, which varies with $X$.
> <!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §§7.4–7.5：核对 LPM 的百分点解释，并结合多项式和交互规格读取完整部分效应。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §§7.4–7.5: verifies the LPM's percentage-point interpretation and how to read full partial effects in polynomial and interaction specifications.
<!-- bilingual-en:end -->
- [[类别连续交互]]：复用主效应、交互系数与组别条件斜率之间的通用关系。
<!-- bilingual-en:start -->
- [[类别连续交互|Indicator-by-continuous interactions]]: supplies the general relationship among main effects, interaction coefficients, and group-specific slopes.
<!-- bilingual-en:end -->
- [[线性组合推断]]：复用对 $\beta_1+\beta_3$ 等完整效应计算标准误的方法。
<!-- bilingual-en:start -->
- [[线性组合推断|Inference for linear combinations]]: supplies the method for calculating standard errors for full effects such as $\beta_1+\beta_3$.
<!-- bilingual-en:end -->
