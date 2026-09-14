---
aliases:
  - "单工具单内生变量且工具矩条件有效时，工具变量系数等于工具对结果的约化式效应除以工具对处理的第一阶段效应"
  - "Single-instrument covariance-ratio estimator"
  - "Wald 比率与协方差比率"
student_os: knowledge-atom
atom_id: ECON-IV-008
atom_set: endogeneity-iv-gmm
atom_type: formula
status: source-checked
mastery_state: unassessed
part_of:
  - "[[内生性与工具变量.canvas|内生性与工具变量]]"
requires:
  - "[[工具变量有效条件]]"
  - "[[识别阶数与秩]]"
related:
  - "[[工具外生与排除]]"
  - "[[定居者死亡率工具变量]]"
---

# 单工具单内生变量且工具矩条件有效时，工具变量系数等于工具对结果的约化式效应除以工具对处理的第一阶段效应
<!-- bilingual-en:start -->
*With one instrument and one endogenous regressor under valid instrument moments, the IV coefficient is the reduced-form effect of the instrument on the outcome divided by its first-stage effect on treatment*
<!-- bilingual-en:end -->

> [!summary] 原子公式
> 对含外生控制 $W$ 的标量结构方程
> $$
> Y=\beta X+W'\gamma+u,
> $$
> 令 $\tilde Y,\tilde X,\tilde Z$ 分别为把 $Y,X,Z$ 对 $(1,W)$ 做总体线性投影后的残差。若 $E[\tilde Zu]=0$ 且 $E[\tilde Z\tilde X]\neq0$，则总体矩条件给出
> $$
> \beta_{IV}
> =\frac{E[\tilde Z\tilde Y]}{E[\tilde Z\tilde X]}
> =\frac{\operatorname{Cov}(\tilde Z,\tilde Y)}{\operatorname{Cov}(\tilde Z,\tilde X)}.
> $$
> 分子是工具对结果的约化式关系，分母是工具对内生回归量的第一阶段关系；样本估计量用对应样本矩替换总体矩。

## 从矩条件推导

残差化后的有效工具矩要求
$$
E[\tilde Z(\tilde Y-\beta\tilde X)]=0.
$$
展开并在 $E[\tilde Z\tilde X]\neq0$ 时求解：
$$
E[\tilde Z\tilde Y]-\beta E[\tilde Z\tilde X]=0
\quad\Longrightarrow\quad
\beta=\frac{E[\tilde Z\tilde Y]}{E[\tilde Z\tilde X]}.
$$
工具乘以任意非零常数会同时缩放分子与分母，因此不改变比率；这不表示弱第一阶段无关，因为接近零的分母会放大抽样噪声和任何有效性偏离。

## 二元工具的 Wald 形式

若 $Z\in\{0,1\}$，且除截距外没有其他控制变量，则
$$
\beta_W=
\frac{E[Y\mid Z=1]-E[Y\mid Z=0]}
{E[X\mid Z=1]-E[X\mid Z=0]}.
$$
分子是 intention-to-treat 型结果差，分母是工具引起的处理接受差。连续工具的协方差比率与这一差分比率承担同一逻辑。

上述残差化不是可选步骤：有已含外生控制 $W$ 时，不能忽略它们后直接套原始协方差比。用 $\tilde Y,\tilde X,\tilde Z$ 计算的比率，与一个排除工具、一个内生回归量的恰好识别 2SLS 点估计一致。

## 解释边界

- 若 $\operatorname{Cov}(\tilde Z,\tilde X)=0$，比率未定义；若它很小，常规正态近似和置信区间可能严重失真。
- 若真实结果式为 $Y=\beta X+W'\gamma+\delta Z+u$，则
  $$
  \frac{\operatorname{Cov}(\tilde Z,\tilde Y)}{\operatorname{Cov}(\tilde Z,\tilde X)}
  =\beta
  +\delta\frac{\operatorname{Var}(\tilde Z)}{\operatorname{Cov}(\tilde Z,\tilde X)}
  +\frac{\operatorname{Cov}(\tilde Z,u)}{\operatorname{Cov}(\tilde Z,\tilde X)}.
  $$
  只有继续维持 $\operatorname{Cov}(\tilde Z,u)=0$ 时，才只多出直接效应的 $\delta$ 项；两种失效都可能被弱第一阶段分母放大，不能混成同一项。
- 在同质线性效应模型中，比率识别共同 $\beta$。二元处理效应异质时，只有再加工具独立性、排除、非零第一阶段和单调性等条件，Wald 比率才具有 compliers 的 LATE 解释；它不自动是 ATE。
- 比率的标准误不是“分子标准误除以分母标准误”。必须使用匹配的 IV/2SLS 联合方差公式。

> [!question]- 自检
> 工具使处理率提高 0.20，使平均结果提高 0.06。若所需同质线性工具条件成立，Wald 比率是多少？
>
> **答案：** $0.06/0.20=0.30$。这个数的因果含义仍依赖外生性、排除和目标效应假设，而不只依赖算术。

## 来源与核验

- [MIT OpenCourseWare 14.310x, Lecture 21](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week10_lec21.pdf)：核对二元工具的结果差/处理差 Wald 比率、first stage、reduced form 及异质效应下的 LATE 边界。
- [MIT OpenCourseWare 14.03, Lecture 13](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/0f7ffdf98f7dc053c20b02dbda223d14_MIT14_03F16_lec13.pdf)：核对第一阶段、约化式和排除限制在应用 IV 中的角色。
- [Stata `ivregress` 官方手册](https://www.stata.com/manuals/rivregress.pdf)：核对单工具恰好识别点估计与 2SLS 的一般矩阵实现，并确认推断不能由手工第二阶段 OLS 标准误替代。
