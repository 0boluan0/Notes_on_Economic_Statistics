---
aliases:
  - "交叉Gamma是价值对两个不同市场因子的混合二阶敏感度"
student_os: knowledge-atom
atom_id: FI-HEDGE-001
atom_type: definition
status: source-checked
requires:
  - "[[Hessian矩阵]]"
  - "[[Greeks]]"
related:
  - "[[Hessian对称条件]]"
leads_to:
  - "[[多因子二阶损益近似]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# 交叉Gamma是价值对两个不同市场因子的混合二阶敏感度
<!-- bilingual-en:start -->
*Cross-gamma is a mixed second-order value sensitivity to two different market factors*
<!-- bilingual-en:end -->

给定估值函数 $V(x_1,\ldots,x_n)$，在其他输入及持仓不变时，交叉 Gamma 衡量因子 $x_i$ 的一阶敏感度怎样随另一个因子 $x_j$ 改变：
<!-- bilingual-en:start -->
For a valuation function $V(x_1,\ldots,x_n)$ with other inputs and holdings fixed, cross-gamma measures how first-order sensitivity to factor $x_i$ changes with another factor $x_j$:
<!-- bilingual-en:end -->

$$\Gamma_{ij}=\frac{\partial}{\partial x_j}\left(\frac{\partial V}{\partial x_i}\right),\qquad i\ne j.$$

它是 [[Hessian矩阵]] 的非对角元素。所需二阶偏导在邻域连续时，[[Hessian对称条件]] 保证 $\Gamma_{ij}=\Gamma_{ji}$。单一因子自身的 Gamma 为零，不代表这些交叉项为零；交叉 Gamma 也不是两个因子的统计相关系数。
<!-- bilingual-en:start -->
It is an off-diagonal entry of the [[Hessian矩阵|Hessian]]. Continuous second partials in a neighbourhood give equality of the two differentiation orders by [[Hessian对称条件|Hessian symmetry]]. Zero own-factor gammas do not imply zero cross terms, and cross-gamma is not a statistical correlation coefficient.
<!-- bilingual-en:end -->

例如持有 $n$ 股外币股票，本币价值为 $V=nSX$，其中 $S$ 是每股外币价格，$X$ 是“本币／外币”汇率。此时 $V_{SS}=V_{XX}=0$，但 $V_{SX}=n$：汇率改变时，股价 Delta $nX$ 也会改变。两个因子同时变化产生的额外项为 $n\Delta S\Delta X$。
<!-- bilingual-en:start -->
For $n$ foreign-currency shares, domestic value is $V=nSX$, where $S$ is foreign price per share and $X$ is domestic currency per unit of foreign currency. Own-factor gammas are zero, but $V_{SX}=n$: FX changes the stock-price delta $nX$. A simultaneous move adds $n\Delta S\Delta X$.
<!-- bilingual-en:end -->

报告时要写明两项因子、单位和扰动规则。不同产品会给混合敏感度专名：例如现货与隐含波动率的 $V_{S\sigma}$ 通常称为 vanna；不应把所有混合导数无区别地只报作“Gamma”。按实际 bump 报告的金额交叉变化，也不等于未经尺度换算的混合偏导。
<!-- bilingual-en:start -->
Report both factors, units, and shock conventions. Some mixed sensitivities have specific names: spot/implied-volatility sensitivity $V_{S\sigma}$ is commonly called vanna. Do not report every mixed derivative as an unidentified “gamma”. A monetary cross-change for specified bumps also differs from an unscaled mixed partial.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [van der Zwaard、Grzelak、Oosterlee，Relevance of Wrong-Way Risk in Funding Valuation Adjustments（2022），印刷第 3 页／含大学封面的 PDF 第 4 页脚注 3](https://research-portal.uu.nl/ws/files/152522438/1_s2.0_S1544612322003166_main.pdf#page=4)：已实际重开大学托管原文并目视该页，核对 cross-gamma 是对两个不同市场输入的二阶偏导；只借用术语定义，不移植该文 FVA 模型。
- [Carr、Wu，Option Profit and Loss Attribution and Pricing（2020），印刷第 2278 页／PDF 第 8 页，式 (2)](https://engineering.nyu.edu/sites/default/files/2020-09/option-profit-carr-jofi-12894.pdf#page=8)：已重开并目视现货／隐含波动率混合项及 vanna 命名。外币股票例子由 $nSX$ 独立求导。
<!-- bilingual-en:start -->
- The university-hosted 2022 paper's printed p. 3 / PDF p. 4, including the repository cover, was reopened and visually checked for footnote 3's mixed-market-input definition of cross-gamma; its FVA model is not imported here.
- Carr and Wu, printed p. 2278 / PDF p. 8, equation (2), was reopened and visually checked for the spot/volatility mixed term and vanna terminology. The foreign-equity example was independently differentiated.
<!-- bilingual-en:end -->
