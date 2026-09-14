---
aliases:
  - "随机效应 GLS 通过按单位规模准去均值结合组内与组间信息"
  - Random-effects GLS
  - RE quasi-demeaning
student_os: knowledge-atom
atom_id: ECON-PANEL-010
atom_set: panel-data
atom_type: method
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[随机效应]]"
  - "[[随机效应正交假设]]"
  - "[[非平衡面板均值]]"
contrasts_with:
  - "[[组内变换]]"
---

# 随机效应 GLS 通过按单位规模准去均值结合组内与组间信息

<!-- bilingual-en:start -->
*Random-effects GLS combines within and between information by quasi-demeaning according to panel size*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在标准随机截距协方差结构下，$\operatorname{Var}(c_i)=\sigma_c^2$、$\operatorname{Var}(u_{it})=\sigma_u^2$、$\operatorname{Cov}(c_i,u_{it})=0$，且 $t\ne s$ 时 $\operatorname{Cov}(u_{it},u_{is})=0$。此时 RE 的 GLS 变换写成
> $$
> z_{it}^*=z_{it}-\theta_i\bar z_i,
> \qquad
> \theta_i=1-\sqrt{\frac{\sigma_u^2}{\sigma_u^2+T_i\sigma_c^2}}.
> $$
> 它只减去单位均值的一部分，所以同时保留组内与组间信息；非平衡面板的 $\theta_i$ 随 $T_i$ 改变。
>
> <!-- bilingual-en:start -->
> Under the standard random-intercept covariance structure, $\operatorname{Var}(c_i)=\sigma_c^2$, $\operatorname{Var}(u_{it})=\sigma_u^2$, $\operatorname{Cov}(c_i,u_{it})=0$, and $\operatorname{Cov}(u_{it},u_{is})=0$ for $t\ne s$. RE then transforms $z_{it}$ into $z_{it}^*=z_{it}-\theta_i\bar z_i$, where $\theta_i=1-\sqrt{\sigma_u^2/(\sigma_u^2+T_i\sigma_c^2)}$. Partial demeaning retains both within and between information, and $\theta_i$ varies with $T_i$ in an unbalanced panel.
> <!-- bilingual-en:end -->

当 $\sigma_c^2=0$ 时，$\theta_i=0$，变换退化为 pooled OLS；当 $\sigma_c^2>0$ 且单位效应相对重要或 $T_i$ 增大时，$\theta_i$ 接近 1，权重更接近 within。时间不变回归量在 quasi-demeaning 后乘以 $1-\theta_i$，因此仍可估计。
<!-- bilingual-en:start -->
When $\sigma_c^2=0$, $\theta_i=0$ and the transformation reduces to pooled OLS. As the unit effect becomes more important or $T_i$ grows, $\theta_i$ approaches one and the transform approaches within demeaning. A time-invariant regressor remains as $(1-\theta_i)z_i$ and can therefore be estimated.
<!-- bilingual-en:end -->

GLS 权重只在 [[随机效应正交假设]] 和方差结构可信时带来通常的效率收益。若 $c_i$ 与 $X_i$ 相关，精细估计 $\theta_i$ 也不会修复系数不一致；方差建模不是内生性修复。
<!-- bilingual-en:start -->
The GLS weighting delivers its usual efficiency gain only when RE orthogonality and the covariance model are credible. If $c_i$ correlates with $X_i$, precise estimation of $\theta_i$ does not restore consistency; covariance modelling is not an endogeneity correction.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么 RE 不是“FE 加一点噪声”？
>
> **答案：** RE 用由方差分量和 $T_i$ 决定的准去均值变换，主动保留组间信息；这要求组间部分也满足额外正交限制。

## 来源与核验

- StataCorp, [`xtreg` manual](https://www.stata.com/manuals/xtxtreg.pdf), Methods and formulas for `xtreg, re`：给出 $T_i$、$\bar z_i$ 与 $\theta_i$ 的准去均值公式，以及 transformed constant $1-\theta_i$。
- [[02_Economy/01_Econometrics/太白金星v计量.pdf]] 第 15 章 PDF 页 170–172：核对课程由复合误差协方差进入 FGLS 的推导顺序。
