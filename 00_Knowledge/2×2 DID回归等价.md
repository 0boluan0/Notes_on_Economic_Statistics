---
aliases:
  - "2×2 DID 的回归交互项等于双重差分"
  - DID regression interaction coefficient
  - 回归 DID
  - Two-by-two DID regression
student_os: knowledge-atom
atom_id: ECON-DID-003
atom_type: equivalence
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
  - "[[虚拟变量与交互项.canvas|虚拟变量与交互项]]"
requires:
  - "[[双重差分法]]"
contrasts_with:
  - "[[2×2 DID 识别 ATT]]"
leads_to:
  - "[[错位 DID 的已处理对照]]"
---

# 2×2 DID 的回归交互项等于双重差分

<!-- bilingual-en:start -->
*In a $2\times2$ design, the regression interaction coefficient equals the difference-in-differences*
<!-- bilingual-en:end -->

> [!summary] 代数结论
> 在只有处理组/对照组和处理前/处理后四个单元格的饱和线性模型中，组别与时期的交互项系数恰好等于四格均值的双重差分。这是代数恒等式，不是因果识别的证明。
>
> <!-- bilingual-en:start -->
> In a saturated linear model with treated/comparison groups and pre/post periods, the group-by-period interaction coefficient is exactly the four-cell DID. This is an algebraic identity, not proof of causal identification.
> <!-- bilingual-en:end -->

$Y_{it}=\alpha+\gamma G_i+\lambda Post_t+\tau(G_i\times Post_t)+u_{it}$ 中 $\tau$ 等于 2×2 DID。面板、重复截面都可使用，但抽样和构成假设不同。加入协变量应明确是提高精度、使条件平行趋势更可信，还是改变目标总体。
<!-- bilingual-en:start -->
In $Y_{it}=\alpha+\gamma G_i+\lambda Post_t+\tau(G_i\times Post_t)+u_{it}$, $\tau$ equals the $2\times2$ DID. Both panel data and repeated cross-sections can be used, but they require different sampling and composition assumptions. When adding covariates, state whether they improve precision, make conditional parallel trends more plausible, or change the target population.
<!-- bilingual-en:end -->

把四个条件均值代入模型即可看见这一点：

|  | 处理前 | 处理后 | 前后差 |
|---|---:|---:|---:|
| 对照组 $G=0$ | $\alpha$ | $\alpha+\lambda$ | $\lambda$ |
| 处理组 $G=1$ | $\alpha+\gamma$ | $\alpha+\gamma+\lambda+\tau$ | $\lambda+\tau$ |

两行的前后差再相减，得到 $(\lambda+\tau)-\lambda=\tau$。因此，在最简单的四格设计里，“手算 DID”和“回归交互项”只是同一个比较的两种写法。
<!-- bilingual-en:start -->
Substituting the four conditional means into the model makes the identity visible. The comparison group's change is $\lambda$, and the treated group's change is $\lambda+\tau$; subtracting them gives $\tau$. In the simple four-cell design, a hand-calculated DID and the regression interaction are two representations of the same comparison.
<!-- bilingual-en:end -->

## 最容易混淆的边界

看到回归软件报告 $\hat\tau$，只能说明样本中的四格比较被算出来了。要把它解释成 ATT，还必须另外论证平行趋势、无预期、样本构成和无干扰等条件。反过来，在多期、错位处理或效应异质时，加入单位固定效应和时间固定效应已经不再等同于一个清楚的四格比较；不能把本原子的结论无限外推。
<!-- bilingual-en:start -->
A reported $\hat\tau$ only shows that the sample's four-cell contrast has been computed. Interpreting it as the ATT still requires parallel trends, no anticipation, stable composition, and no interference. Conversely, with many periods, staggered adoption, or heterogeneous effects, unit and time fixed effects no longer represent one transparent four-cell comparison. This result must not be extrapolated beyond its $2\times2$ scope.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 若 $\tau$ 在四格回归中显著，是否已经证明政策有因果作用？为什么？
>
> **答案：** 没有。$\tau$ 等于 DID 是代数事实；DID 等于 ATT 是另一个依赖识别假设的事实。显著性不能补上缺失的反事实论证。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，式 (2.8)–(2.10)：核验两组两期回归交互项与 DID/ATT 的对应关系及其所需假设。
- Wooldridge, *Introductory Econometrics: A Modern Approach*，关于两期面板与政策分析的章节：交叉核验虚拟变量交互项的四格均值解释。
