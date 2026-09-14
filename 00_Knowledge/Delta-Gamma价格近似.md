---
aliases:
  - "Delta-Gamma价格近似用当前一阶与二阶价格导数估计小幅标的变动"
student_os: knowledge-atom
atom_id: FI-GREEK-008
atom_type: method
status: source-checked
requires:
  - "[[Delta]]"
  - "[[Gamma]]"
  - "[[多元Taylor近似]]"
related:
  - "[[多因子二阶损益近似]]"
  - "[[Greeks损益归因]]"
  - "[[估值近似验证]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Delta-Gamma价格近似用当前一阶与二阶价格导数估计小幅标的变动
<!-- bilingual-en:start -->
*The delta–gamma price approximation uses current first and second price derivatives to estimate a small underlying move*
<!-- bilingual-en:end -->

在同一估值时刻，固定波动率、利率和其余输入，只把标的价格从 $S$ 改为 $S+h$。若这条价格方向上的价值函数在基准点邻域为 $C^2$，则 [[多元Taylor近似|Taylor 展开]] 在此方向上给出
<!-- bilingual-en:start -->
At one valuation time, hold volatility, rates, and all other inputs fixed and move only the underlying price from $S$ to $S+h$. If the value function restricted to this price direction is $C^2$ near the baseline, the [[多元Taylor近似|Taylor expansion]] gives:
<!-- bilingual-en:end -->

$$V(S+h)-V(S)=\Delta_Sh+\frac12\Gamma_{SS}h^2+o(h^2).$$

应用时先统一价值单位与价格坐标，再用基准点的 Delta 和 Gamma 计算线性项与二阶修正。$\tfrac12$ 不可漏掉；Gamma 为正时二阶项非负，Gamma 为负时非正，但这只是对线性预测的修正，不单独决定总价格变化的符号。
<!-- bilingual-en:start -->
First align value units and the price coordinate, then use baseline delta and gamma to compute the linear term and quadratic correction. Retain the factor of $\tfrac12$. Positive gamma gives a nonnegative quadratic term and negative gamma a nonpositive one, but this corrects the linear prediction rather than determining the sign of the whole price change.
<!-- bilingual-en:end -->

例如 Delta 为 $0.5$、Gamma 为 $0.04$，标的价格增加 $2$：一阶估计为 $1$，Delta-Gamma 估计为
<!-- bilingual-en:start -->
With delta $0.5$, gamma $0.04$, and a price increase of $2$, the first-order estimate is $1$, while the delta–gamma estimate is:
<!-- bilingual-en:end -->

$$0.5\times2+\frac12\times0.04\times2^2=1.08.$$

可以用局部多项式 $V(S_0+h)=10+0.5h+0.02h^2+0.001h^3$ 检查它的局限：基准 Delta、Gamma 正好是上述数值，但 $h=2$ 的精确增量是 $1.088$，尚有三阶余项 $0.008$。二阶近似没有消除全部误差，也不保证任意有限冲击下总比一阶更准确。
<!-- bilingual-en:start -->
The local polynomial $V(S_0+h)=10+0.5h+0.02h^2+0.001h^3$ illustrates the limitation. Its baseline delta and gamma match the stated values, but the exact increment at $h=2$ is $1.088$, leaving the cubic remainder $0.008$. Second order does not remove all error or guarantee greater accuracy than first order for every finite shock.
<!-- bilingual-en:end -->

若多个输入同时变化，应使用 [[多因子二阶损益近似]]，交叉项与其他平方项也可能是二阶，不能因不是 Gamma 项就称为更高阶。若跨越时间、收付现金流或发生交易，应按 [[Greeks损益归因]] 处理完整损益。大冲击、非光滑边界或目标尾部的可用性，须依 [[估值近似验证]] 与基准全重估比较；增加 Gamma 不是通用精度保证。
<!-- bilingual-en:start -->
If multiple inputs move, use the [[多因子二阶损益近似|multifactor second-order P&L approximation]]: cross terms and other squared terms can also be second order. Changes across time, cash flows, and trades require the fuller convention in [[Greeks损益归因|Greek P&L attribution]]. For large shocks, nonsmooth boundaries, or target-tail use, follow [[估值近似验证|valuation-approximation validation]] against benchmark revaluation; adding gamma is not a universal accuracy guarantee.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[多元Taylor近似]]：复用已核验的 $C^2$ 局部二阶展开及 $o(\|h\|^2)$ 余项，限制到单一价格方向得到本式。
- [NYU Courant，Kohn 与 Allen，Section 5，第 5、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开并目视两页，核对 Delta、Gamma 的导数含义及未及时再平衡时的价格曲率影响。$1.08$ 与多项式精确增量为独立复算，不作为市场观测值。
<!-- bilingual-en:start -->
- The verified [[多元Taylor近似|multivariable Taylor approximation]] supplies the $C^2$ expansion and its $o(\|h\|^2)$ remainder; restricting it to one price coordinate gives the displayed equation.
- [NYU Courant, Kohn and Allen, Section 5, pp. 5 and 8](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), reopened and visually checked, supports the derivative meanings and curvature exposure between hedge adjustments. The estimate of $1.08$ and exact polynomial increment were independently recalculated, not treated as observed market prices.
<!-- bilingual-en:end -->
