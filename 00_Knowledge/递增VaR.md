---
aliases:
  - "在本组采用的口径下，递增 VaR 是有限交易前后组合 VaR 的差"
  - Incremental VaR
student_os: knowledge-atom
atom_id: RM-VAR-028
atom_set: var-es-backtesting
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
related:
  - "[[VaR贡献口径]]"
  - "[[边际VaR]]"
  - "[[成分VaR]]"
  - "[[全微分]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 在本组采用的口径下，递增 VaR 是有限交易前后组合 VaR 的差
<!-- bilingual-en:start -->
*In the convention used here, incremental VaR is the difference between portfolio VaR after and before a finite trade*
<!-- bilingual-en:end -->

> [!summary] 它回答“做完这笔交易，风险实际变多少”
> 对有限头寸变化 $\Delta x$，递增 VaR 必须比较完整的新旧组合，而不是把当前点的局部导数当作精确差值。
>
> <!-- bilingual-en:start -->
> For a finite position change $\Delta x$, incremental VaR compares the complete new and old portfolios. A local derivative is not an exact finite difference.
> <!-- bilingual-en:end -->

$$
\operatorname{IVaR}(\Delta x;x)
=\rho(x+\Delta x)-\rho(x).
$$

只有当 $\Delta x$ 足够小且 $\rho$ 局部平滑时，

$$
\operatorname{IVaR}(\Delta x;x)
\approx\nabla\rho(x)^\top\Delta x.
$$

例如交易前 VaR 为 80，交易后完整重算为 92，则递增 VaR 为 12；无论该交易当前成分 VaR 是多少，有限变化的答案仍由 92−80 给出。

<!-- bilingual-en:start -->
If VaR is 80 before the trade and 92 after full recomputation, incremental VaR is 12. The finite answer is 92−80 regardless of the trade's current component VaR.
<!-- bilingual-en:end -->

> [!question]- 自检
> 删除一个大头寸时，能否用它的边际 VaR 直接作为递增 VaR？
>
> **答案：** 不能。边际 VaR 还要乘变化量且只是局部近似；大变动应重算完整组合。
>
> <!-- bilingual-en:start -->
> **Self-check:** When removing a large position, can its marginal VaR be used directly as incremental VaR?
>
> **Answer:** No. Marginal VaR must be multiplied by the position change and is only a local approximation; a large change requires recomputing the full portfolio.
> <!-- bilingual-en:end -->

## 边界

- 递增 VaR 的符号取决于“新组合减旧组合”的约定。
- 分位排序跳变会使线性近似很差。
- 交易后的相关性、非线性和对冲关系必须重新进入估计。

<!-- bilingual-en:start -->
- The sign follows the “new minus old” convention.
- Quantile-rank jumps can make linear approximation poor.
- Post-trade dependence, nonlinearity, and hedging must enter the recomputation.
<!-- bilingual-en:end -->

## 来源与核验

- Hallerbach, [*Decomposing Portfolio Value-at-Risk: A General Analysis*](https://repub.eur.nl/pub/7723/1999-0342.pdf)，§1 与 §2 式 (6)：核对新增交易对组合 VaR 的增量问题，以及小交易可用 marginal VaR 作一阶近似。
- Tasche, [*Capital Allocation to Business Units and Sub-Portfolios*](https://arxiv.org/abs/0708.2542)：核对有限差分与 Euler 局部分摊的区别。
- RiskMetrics Group, [*Risk Management: A Practical Guide*](https://www.msci.com/documents/10199/3c2dcea9-97be-4fb4-befe-a03b75c885aa)，§1.2：提供会交换 marginal 与 incremental 标签的另一套实务命名；本卡以明示的“新组合减旧组合”公式固定本组口径。
