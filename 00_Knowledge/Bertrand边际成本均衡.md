---
aliases:
  - "在同质产品、相同恒定边际成本 c≥0、价格可在 [0,∞) 连续选择、无容量约束、同价均分需求，且市场需求 D(p) 非增并在 c 上方的一个右邻域内严格为正的 Bertrand 双寡头中，唯一纯策略 Nash 均衡是两家都按边际成本定价"
  - Under homogeneous goods, identical constant marginal cost c≥0, a continuous price set [0,∞), no capacity constraints, equal demand splitting at tied prices, and nonincreasing market demand D(p) that is strictly positive in a right-neighborhood above c, the unique pure-strategy Bertrand equilibrium has both firms price at marginal cost
  - Bertrand悖论
student_os: knowledge-atom
atom_id: GT-OLI-007
atom_set: oligopoly-competition
atom_type: equilibrium-result
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bertrand竞争]]"
  - "[[纳什均衡]]"
related:
  - "[[均衡与效率]]"
leads_to:
  - "[[Bertrand基准边界]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在同质产品、相同恒定边际成本 c≥0、价格可在 [0,∞) 连续选择、无容量约束、同价均分需求，且市场需求 D(p) 非增并在 c 上方的一个右邻域内严格为正的 Bertrand 双寡头中，唯一纯策略 Nash 均衡是两家都按边际成本定价
<!-- bilingual-en:start -->
*Under homogeneous goods, identical constant marginal cost $c\ge 0$, a continuous price set $[0,\infty)$, no capacity constraints, equal demand splitting at tied prices, and nonincreasing market demand $D(p)$ that is strictly positive in a right-neighborhood above $c$, the unique pure-strategy Bertrand equilibrium has both firms price at marginal cost*
<!-- bilingual-en:end -->

> [!summary] 两家企业也可能得到竞争价格，但结论靠一整组假设
> 共同价格若高于边际成本，一家可以略微降价并在能供应全部市场时取得所有需求；低于边际成本则卖得越多亏得越多。只有两家都报 $c$ 时，任何单边偏离都不能获得正利润。
>
> <!-- bilingual-en:start -->
> If a common price exceeds marginal cost, either firm can undercut slightly and serve the entire market when capacity is unconstrained. Below marginal cost, additional sales deepen the loss. Only the pair $(c,c)$ leaves no unilateral deviation that earns positive profit.
> <!-- bilingual-en:end -->

## 排除所有其他价格组合

设两家企业边际成本相同且恒为 $c\ge 0$，价格可在 $[0,\infty)$ 连续选择。市场需求 $D(p)\ge 0$ 随价格非增，并在 $c$ 上方的某个右邻域内严格为正。低价企业取得全部需求，同价时各得一半；任一家都没有容量约束，能够独自服务全部市场。

1. **共同价格 $p_1=p_2=p>c$：** 若 $D(p)>0$，取足够小的 $0<\varepsilon<(p-c)/2$。因为 $D$ 非增，$D(p-\varepsilon)\ge D(p)$，降价后的利润严格大于原来平分市场的利润。若 $D(p)=0$，则可改报右邻域内某个 $c+\delta<p$ 且 $D(c+\delta)>0$ 的价格，取得正利润。原组合都不是均衡。
2. **最低价格小于 $c$：** 报最低价的企业承担负单位利润，可以提高价格到不成交或至少不亏的位置；原组合不是均衡。
3. **两家价格不等且最低价高于 $c$：** 高价企业可报一个介于 $c$ 与最低价之间的价格，取得正利润；原组合不是均衡。
4. **一家报 $c$，另一家报更高价格：** 报 $c$ 的企业可略微提高价格但仍低于对手，从零利润变为正利润；原组合不是均衡。
5. **两家都报 $c$：** 降价会产生负利润；涨价会失去全部需求且仍得零利润；保持原价也得零利润。因此没有严格有利的单边偏离。

这些情况穷尽价格关系，所以 $(p_1^*,p_2^*)=(c,c)$ 是唯一纯策略 Nash 均衡。

<!-- bilingual-en:start -->
Every other price pair admits a profitable deviation. At a positive-demand tie above cost, a sufficiently small undercut raises demand weakly and changes the firm's share from one half to all; at a zero-demand tie, the firm can instead quote a positive-demand price just above $c$. A lowest price below cost can be abandoned; unequal prices above cost let the high-price firm undercut profitably; and if one firm charges $c$ while the other charges more, the low-price firm can raise its price slightly while remaining cheaper. At $(c,c)$, cutting price loses money and raising price loses demand without creating positive profit. These cases establish uniqueness.
<!-- bilingual-en:end -->

## “等于边际成本”不是企业的一阶条件

均衡来自相对价格造成的销量跳变和单边偏离，而不是先写 $MR=MC$ 再求一个光滑内点。两家均衡利润为 0 也不表示没有生产者剩余的所有现实含义；若存在固定成本，零经营利润可能不足以覆盖进入成本，市场结构本身还需由进入模型解释。

结论常被称为 Bertrand paradox：仅两家企业就在基准模型中消除加价。它之所以“反直觉”，正说明应继续检查 [[Bertrand基准边界|同质性、容量和需求分配等假设]]，而不是把 $p=c$ 当成所有价格竞争的定义。

<!-- bilingual-en:start -->
The result is produced by discontinuous demand allocation and deviation incentives, not by a smooth $MR=MC$ first-order condition. It is often called the Bertrand paradox because two firms eliminate the markup in the benchmark. The right response is to inspect the assumptions, not to treat $p=c$ as the definition of all price competition.
<!-- bilingual-en:end -->

> [!question]- 自检
> 价格组合 $(c,c+10)$ 中，报 $c$ 的企业已经拿走全部市场但利润为 0。为什么这仍不是均衡？
>
> **答案：** 它可以把价格提高到 $c+\varepsilon<c+10$，仍是低价者并取得全部需求，同时获得正的单位利润；因此存在有利单边偏离。

## 来源与核验

- MIT 14.12, [*Chapter 7: Application—Imperfect Competition*, §7.2](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/a870a72380a584e8d1ffd2b34fa24c9e_MIT14_12F12_chapter7.pdf)：核对同质品 Bertrand 的分段需求与唯一 $p=c$ 均衡证明。
- MIT 15.010/15.011, [*The Basics of Game Theory*, p. 3](https://ocw.mit.edu/courses/15-010-economic-analysis-for-business-decisions-fall-2004/807ba86e100d349ef73c294b9e720931_the_bsc_game_thy.pdf)：交叉核对持续降价直到价格等于边际成本的偏离逻辑。
- [[01_Math/03_game theory/第2次作业.pdf|作业 2，第 2 题]]与[[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#4.2. 纳什均衡|本地课程：Bertrand 均衡]]：支持课程要求证明唯一均衡及其分情况写法。
