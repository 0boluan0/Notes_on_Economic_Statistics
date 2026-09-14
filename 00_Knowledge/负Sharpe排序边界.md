---
aliases:
  - 负Sharpe的数值排名不能直接解释为均值—方差优劣
  - Negative Sharpe ratio ranking limitation
student_os: knowledge-atom
atom_id: FI-MV-024
atom_type: boundary
status: source-checked
part_of:
  - "[[均值—方差投资组合理论.canvas]]"
requires:
  - "[[Sharpe比率]]"
related:
  - "[[资本配置线]]"
  - "[[均值方差支配]]"
---

# 负Sharpe的数值排名不能直接解释为均值—方差优劣
<!-- bilingual-en:start -->
*Ranking negative Sharpe ratios numerically does not directly rank mean–variance desirability*
<!-- bilingual-en:end -->

在相同且确定的无风险基准下，如果两组合的平均超额收益都为负，较大的 [[Sharpe比率]] 可能只是分母更大、即波动更高。因而“Sharpe 越高就越好”不能无条件用于原组合之间的 [[均值方差支配|均值—方差比较]]。
<!-- bilingual-en:start -->
Under the same certain risk-free benchmark, a higher [[Sharpe比率|Sharpe ratio]] between portfolios with negative mean excess returns may merely reflect greater volatility in the denominator. Thus “higher Sharpe is better” cannot be applied unconditionally to [[均值方差支配|mean–variance comparisons]] of the original portfolios.
<!-- bilingual-en:end -->

最小反例：两组合的期望收益均为 1%，无风险收益均为 3%；A 的波动率为 10%，B 为 20%。此时 $S_A=-0.20$、$S_B=-0.10$，数值上 B 更高，但 A 在相同期望收益下波动更低。若无风险资产可投资，两者又都被确定收益 3% 的资产支配。
<!-- bilingual-en:start -->
A minimal counterexample has expected return 1% for both portfolios and risk-free return 3%. A has volatility 10%, B has 20%, giving $S_A=-0.20$ and $S_B=-0.10$. B ranks higher numerically, yet A has lower volatility at the same expected return. If the risk-free asset is available, its certain 3% return dominates both.
<!-- bilingual-en:end -->

这不意味着负 Sharpe 没有数学含义。对于同一差额收益 $D$，按固定倍数 $a\ne0$ 放大头寸时，比率满足下式：正倍数保持比率，负倍数翻转符号。但“允许做空整个差额收益策略”改变了可选头寸，不能把这种新策略和原来的多头组合混为同一个对象。
<!-- bilingual-en:start -->
A negative Sharpe ratio still has mathematical meaning. Scaling one differential return $D$ by a fixed $a\ne0$ gives the following relation: positive scaling preserves the ratio and negative scaling reverses its sign. Allowing a short position in the entire differential-return strategy changes the available position; it is not the original long portfolio.
<!-- bilingual-en:end -->

$$
S(aD)=\frac{aE(D)}{|a|\operatorname{sd}(D)}
=\operatorname{sign}(a)S(D).
$$

即使比较的 Sharpe 都为正，单项比率也未必决定它加入既有持仓后的贡献，因为还需要协方差。先明确比较的是原组合、可以调节规模的 [[资本配置线|配置策略]]，还是对既有组合的增量，再解释排名。
<!-- bilingual-en:start -->
Even when all ratios are positive, a standalone ratio need not determine the benefit of adding an asset to existing holdings; covariance also matters. Before interpreting a ranking, distinguish original portfolios, scalable [[资本配置线|allocation strategies]], and additions to an existing portfolio.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Sharpe（1994），The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)，Scale Independence、Adding a Strategy to a Risky Portfolio 和脚注 2：支持规模、头寸方向与既有持仓相关性对使用范围的影响。负值排序反例和 $S(aD)$ 是依据定义的自拟核验，并非原文算例。
<!-- bilingual-en:start -->
- [Sharpe (1994), The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm), Scale Independence, Adding a Strategy to a Risky Portfolio, and endnote 2, supports the roles of scaling, position direction, and correlations with existing holdings. The negative-ranking counterexample and $S(aD)$ are original checks from the definition, not examples quoted from the paper.
<!-- bilingual-en:end -->
