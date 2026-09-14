---
aliases:
  - "结构 VAR 用可解释冲击与同期关系说明系统的经济传导"
  - Structural VAR
  - SVAR
  - Structural vector autoregression
student_os: knowledge-atom
atom_id: TS-VAR-023
atom_set: vector-autoregression
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
  - "[[简约型VAR创新]]"
related:
  - "[[简约型创新不是结构冲击]]"
  - "[[SVAR识别条件]]"
  - "[[结构脉冲响应]]"
  - "[[预测误差方差分解]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 结构 VAR 用可解释冲击与同期关系说明系统的经济传导
<!-- bilingual-en:start -->
*A structural VAR uses interpretable shocks and contemporaneous relations to explain the system's economic transmission*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 结构向量自回归（SVAR）在简约型 VAR 的联合动态之上，另外指定当期关系与一组具有经济含义的结构冲击。它的目标不只是预测 $y_t$，而是把观测到的新息分解成可命名的冲击，再追踪这些冲击如何经动态系统传播。

<!-- bilingual-en:start -->
> [!summary] What it is
> A structural vector autoregression augments the joint dynamics of a reduced-form VAR with contemporaneous relations and economically interpretable structural shocks. Its aim is not merely to forecast $y_t$, but to decompose observed news into named shocks and trace their propagation through the dynamic system.
<!-- bilingual-en:end -->

一种常见写法是
$$
B_0y_t=b+B_1y_{t-1}+\cdots+B_py_{t-p}+\varepsilon_t,
\qquad E(\varepsilon_t\varepsilon_t')=I_K,
$$
其中 $B_0$ 编码当期关系，$\varepsilon_t$ 是已定标的候选结构冲击。若 $B_0$ 非奇异，左乘 $B_0^{-1}$ 得到简约型
$$
y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+u_t,
\qquad u_t=B_0^{-1}\varepsilon_t.
$$
后一个模型可以从数据估计，但它通常只确定 $\Sigma_u=B_0^{-1}(B_0^{-1})'$，不唯一确定 $B_0$。

<!-- bilingual-en:start -->
A common representation is $B_0y_t=b+\sum_{i=1}^pB_iy_{t-i}+\varepsilon_t$ with normalized structural shocks. If $B_0$ is nonsingular, premultiplication yields the reduced form with $u_t=B_0^{-1}\varepsilon_t$. The reduced form is estimable from the data, but its covariance generally identifies only $B_0^{-1}(B_0^{-1})'$, not the structural impact matrix itself.
<!-- bilingual-en:end -->

因此，“结构”不是指方程很多、变量都叫内生变量，也不是在估计后给残差贴上名字。它指的是：同期映射、冲击尺度与经济标签由一组可明说、可辩护的限制所支持。只有 [[SVAR识别条件|识别条件]] 确实选出了所需方向，[[结构脉冲响应|结构 IRF]] 和结构 FEVD 才能承担相应的经济解释。

<!-- bilingual-en:start -->
"Structural" does not mean merely having many equations or calling every variable endogenous, nor does it arise by attaching labels to estimated residuals. It means that the contemporaneous map, shock scale, and economic labels are supported by explicit, defensible restrictions. Structural IRFs and FEVDs inherit economic meaning only after those restrictions actually identify the required shock directions.
<!-- bilingual-en:end -->

> [!example] 同一个简约型，不同结构故事
> 利率、通胀与产出的简约型 VAR 可以产生稳定预测，却同时兼容“利率当期不响应产出冲击”和反向排序等多种递归结构。这些结构共享同一个简约型拟合，但会产生不同的政策冲击响应。

> [!question]- 自检
> 把多个宏观变量放入同一个 VAR，并对创新做 Cholesky 分解，是否就自动得到了经济上有效的 SVAR？
>
> **答案：** 没有。Cholesky 给出候选递归映射；变量排序所隐含的同期零限制还必须由制度时序或经济理论支持。

## 来源与核验

- [Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*](https://doi.org/10.1017/9781108164818)，第 1–3 章：核对结构形式、简约形式与冲击映射。
- [Kilian & Lütkepohl, Cambridge 节选](https://assets.cambridge.org/97811071/96575/excerpt/9781107196575_excerpt.pdf)：核对“结构参数可从简约型与限制恢复”的识别定义，以及结构结论对限制可信度的依赖。
- [Stock & Watson (2001), *Vector Autoregressions*](https://www.aeaweb.org/articles?id=10.1257/jep.15.4.101)：核对简约型、递归型与结构型 VAR 在数据描述、预测和结构推断中的不同任务。
- [[01_Math/06_时间序列分析/lecture.pdf#page=219|课程讲义 pp. 219–221]]：核对二变量候选结构式及其简约化。
