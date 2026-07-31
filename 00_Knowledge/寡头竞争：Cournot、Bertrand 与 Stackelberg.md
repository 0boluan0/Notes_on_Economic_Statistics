---
aliases:
  - "Oligopoly Models"
  - "Cournot, Bertrand and Stackelberg Competition"
  - "寡头竞争"
status: source-checked
---

# 寡头竞争：Cournot、Bertrand 与 Stackelberg
<!-- bilingual-en:start -->
*Oligopoly Competition: Cournot, Bertrand, and Stackelberg*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 比较少数企业选择产量、价格或行动时序时，相互最佳反应怎样决定市场价格、总量与利润。
> **具体锚点：** Cournot 企业选择产量时，每家的最优产量取决于对手产量；反应函数的交点就是均衡。
> **核心难点：** Cournot、Bertrand 与 Stackelberg 的结论由战略变量、可观察时序、产品差异和容量约束共同决定，不能互换公式。
> **为什么重要：** 它把市场结构落到可检验的企业行为，并解释为何相似企业数目下价格竞争强度可能完全不同。
> **继续：** 先用 [[纳什均衡与混合策略]] 掌握相互最佳反应；有可信承诺时再用 [[扩展式博弈、逆向归纳与子博弈完美均衡]] 检查先后行动。
> <!-- bilingual-en:start -->
> **What it solves:** Compare how mutual best responses determine market price, total output, and profit when a small number of firms choose quantities, prices, or the timing of moves.
> **Concrete anchor:** In Cournot competition, each firm chooses output in response to its rival's output; the intersection of their reaction functions is the equilibrium.
> **Central difficulty:** Cournot, Bertrand, and Stackelberg conclusions depend jointly on the strategic variable, observable timing, product differentiation, and capacity constraints; their formulas are not interchangeable.
> **Why it matters:** The models translate market structure into testable firm behavior and explain why price competition can differ radically with a similar number of firms.
> **Continue with:** First use [[纳什均衡与混合策略|Nash Equilibrium and Mixed Strategies]] to master mutual best responses; when commitment is credible, use [[扩展式博弈、逆向归纳与子博弈完美均衡|Extensive-form Games, Backward Induction, and Subgame-perfect Equilibrium]] to analyze sequential moves.
> <!-- bilingual-en:end -->

寡头模型回答的是：少数企业相互预见对方行为时，战略变量、行动时序和产品环境怎样决定价格、产量与利润。先判断企业选择的是数量还是价格、同时还是先后行动，再选择模型；不能先背公式再给市场贴标签。
<!-- bilingual-en:start -->
Oligopoly models ask how strategic variables, timing, and product-market conditions determine prices, quantities, and profits when a small number of firms anticipate one another's actions. First determine whether firms choose quantities or prices and whether moves are simultaneous or sequential; do not memorize a formula and then attach a market label afterward.
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/03_game theory/第1次作业.pdf]]—[[01_Math/03_game theory/第3次作业.pdf]]：支持课程覆盖、课堂模型与题目语境。
> - [MIT OCW 14.12 Game Theory lecture notes](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/pages/lecture-notes/)：核验策略式、扩展式、Nash、Bayesian 与精炼概念。
> - [MIT OCW 15.040 Game Theory for Managers](https://ocw.mit.edu/courses/15-040-game-theory-for-managers-spring-2004/pages/lecture-notes/)：核验同时行动、序贯行动、承诺和市场进入的战略逻辑。
> <!-- bilingual-en:start -->
> - [[01_Math/03_game theory/第1次作业.pdf|Game Theory Assignment 1]]—[[01_Math/03_game theory/第3次作业.pdf|Game Theory Assignment 3]]: support the course coverage, classroom models, and problem context.
> - [MIT OCW 14.12 Game Theory lecture notes](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/pages/lecture-notes/): verifies strategic-form and extensive-form games, Nash and Bayesian equilibrium, and refinements.
> - [MIT OCW 15.040 Game Theory for Managers](https://ocw.mit.edu/courses/15-040-game-theory-for-managers-spring-2004/pages/lecture-notes/): verifies the strategic logic of simultaneous moves, sequential moves, commitment, and market entry.
> <!-- bilingual-en:end -->

## 三个模型的共同骨架
<!-- bilingual-en:start -->
*The Common Structure of the Three Models*
<!-- bilingual-en:end -->

Cournot 同时选产量，价格由总量决定；Bertrand 同时选价格，产品同质和容量无限等假设会产生强烈价格竞争；Stackelberg 有先行者，领导者预见跟随者的反应再优化。三者差异来自战略变量、时序、产品差异和容量约束，不应只背均衡公式。
<!-- bilingual-en:start -->
Cournot firms choose quantities simultaneously, with price determined by total output. Bertrand firms choose prices simultaneously; homogeneous products and unlimited capacity can generate extremely aggressive price competition. In Stackelberg competition, a leader moves first and optimizes after anticipating the follower's reaction. The models differ because of the strategic variable, timing, product differentiation, and capacity constraints—not merely because their equilibrium formulas differ.
<!-- bilingual-en:end -->

## Cournot worked example
<!-- bilingual-en:start -->
*Cournot Worked Example*
<!-- bilingual-en:end -->

设逆需求 $P=a-b(q_1+q_2)$，两家边际成本均为 $c<a$。企业 1 的利润为 $[a-b(q_1+q_2)-c]q_1$，一阶条件给出反应函数 $q_1=(a-c-bq_2)/(2b)$；企业 2 对称。联立得到 $q_1^*=q_2^*=(a-c)/(3b)$。这不是把垄断产量机械除以二，而是两条最佳反应同时成立的结果。
<!-- bilingual-en:start -->
Let inverse demand be $P=a-b(q_1+q_2)$ and let both firms have marginal cost $c<a$. Firm 1's profit is $[a-b(q_1+q_2)-c]q_1$, so its first-order condition gives the reaction function $q_1=(a-c-bq_2)/(2b)$; firm 2 is symmetric. Solving the two reaction functions yields $q_1^*=q_2^*=(a-c)/(3b)$. This is not obtained by mechanically dividing monopoly output in half; it follows from satisfying both best-response conditions simultaneously.
<!-- bilingual-en:end -->

## Bertrand 与关键假设
<!-- bilingual-en:start -->
*Bertrand Competition and Its Critical Assumptions*
<!-- bilingual-en:end -->

同质产品、相同恒定边际成本、无容量约束且消费者购买最低价产品时，只要价格高于边际成本，对手就能略微降价夺走市场，因此均衡趋向 $p=c$。产品差异、搜索成本、容量限制、动态互动或成本差异都会改变结论；看到现实价格高于边际成本，不能据此简单宣布 Bertrand 模型“错误”。
<!-- bilingual-en:start -->
With homogeneous products, identical constant marginal costs, no capacity constraints, and consumers buying from the lowest-price seller, any price above marginal cost can be undercut slightly, so equilibrium converges to $p=c$. Product differentiation, search costs, capacity constraints, repeated interaction, or cost asymmetry all change this result. Observing prices above marginal cost does not by itself falsify Bertrand reasoning; it may instead show that its benchmark assumptions do not apply.
<!-- bilingual-en:end -->

## Stackelberg 与承诺
<!-- bilingual-en:start -->
*Stackelberg Competition and Commitment*
<!-- bilingual-en:end -->

领导者先选择可观察且难以撤回的产量，再把跟随者反应函数代入自己的利润最大化。先行动优势来自承诺改变了对手的可行最佳反应，不来自“先走天然更聪明”。如果行动不可观察、容易逆转或企业选择的是战略互补变量，先行动未必有利。
<!-- bilingual-en:start -->
The leader first chooses an observable and difficult-to-reverse quantity, then substitutes the follower's reaction function into its own profit-maximization problem. First-mover advantage comes from commitment changing the rival's feasible best response, not from moving first being inherently smarter. If the action is unobservable, easily reversible, or strategically complementary, moving first need not be advantageous.
<!-- bilingual-en:end -->

## 比较静态与模型边界
<!-- bilingual-en:start -->
*Comparative Statics and Model Boundaries*
<!-- bilingual-en:end -->

参数变化如何移动反应函数，比单个数值均衡更有解释力。结论必须绑定假设：边际成本、需求斜率、进入、容量、重复互动或产品差异改变时，均衡结构可能改变。
<!-- bilingual-en:start -->
How parameter changes shift reaction functions is more informative than a single numerical equilibrium. Every conclusion must remain tied to its assumptions: changing marginal cost, demand slope, entry, capacity, repeated interaction, or product differentiation can alter the equilibrium structure.
<!-- bilingual-en:end -->

## 重复互动入口
<!-- bilingual-en:start -->
*Entry Point to Repeated Interaction*
<!-- bilingual-en:end -->

一次博弈的 Nash 不必是长期唯一结果。重复博弈中，未来惩罚可支持合作，但需要足够耐心、可观察行为和可信的惩罚策略。重复本身不自动产生合作。
<!-- bilingual-en:start -->
The one-shot Nash outcome need not be the only long-run outcome. In a repeated game, future punishment may sustain cooperation, but only when players are sufficiently patient, actions are observable, and punishment strategies are credible. Repetition by itself does not automatically generate cooperation.
<!-- bilingual-en:end -->

## 选择模型与失败诊断
<!-- bilingual-en:start -->
*Model Selection and Failure Diagnosis*
<!-- bilingual-en:end -->

先回答四个问题：企业真正控制什么变量；竞争行动能否被对手观察；容量是否在定价前已锁定；产品和消费者搜索是否造成差异。若解出负产量，应检查非负边界；若 Bertrand 结论与数据不符，应先检查容量、差异化和动态互动；若把 Stackelberg 当作同时行动求解，就会丢掉承诺的全部作用。
<!-- bilingual-en:start -->
Begin with four questions: What variable do firms actually control? Can rivals observe the strategic action? Is capacity committed before pricing? Do product differentiation or consumer search matter? A negative quantity signals that the non-negativity boundary must be checked. If the Bertrand benchmark conflicts with the data, inspect capacity, differentiation, and dynamic interaction first. Solving Stackelberg as a simultaneous-move game removes the entire commitment mechanism.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimal Self-check*
<!-- bilingual-en:end -->

### Cournot 与 Stackelberg 的核心差别是什么？
<!-- bilingual-en:start -->
*What is the central difference between Cournot and Stackelberg competition?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 战略变量都可为产量，但 Stackelberg 有可观察的先后顺序，先行者把跟随者反应函数纳入自己的优化。
> <!-- bilingual-en:start -->
> Both may use quantity as the strategic variable, but Stackelberg has an observable order of moves, so the leader incorporates the follower's reaction function into its own optimization.
> <!-- bilingual-en:end -->

### 同质产品 Bertrand 均衡为什么会逼近边际成本？
<!-- bilingual-en:start -->
*Why does the homogeneous-product Bertrand equilibrium approach marginal cost?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 只要共同价格高于边际成本，一家企业就能略微降价并在无容量约束下夺走全部需求；该偏离直到价格降至边际成本才停止有利。
> <!-- bilingual-en:start -->
> Whenever the common price exceeds marginal cost, one firm can undercut slightly and capture all demand when capacity is unconstrained; the deviation stops being profitable only at marginal cost.
> <!-- bilingual-en:end -->

### 观察到价格高于边际成本，为什么不能直接否定 Bertrand 逻辑？
<!-- bilingual-en:start -->
*Why does observing price above marginal cost not directly refute Bertrand logic?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 现实可能存在产品差异、搜索成本、容量约束、重复互动或成本差异；应先判断基准假设是否成立。
> <!-- bilingual-en:start -->
> Product differentiation, search costs, capacity constraints, repeated interaction, or cost asymmetry may be present, so the benchmark assumptions must be assessed before rejecting the logic.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[01_Math/03_game theory/第1次作业.pdf]]—[[01_Math/03_game theory/第3次作业.pdf]]：支持课程覆盖、课堂模型与题目语境。
- [MIT OCW 14.12 Game Theory lecture notes](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/pages/lecture-notes/)：核验策略式、扩展式、Nash、Bayesian 与精炼概念。
- [MIT OCW 15.040 Game Theory for Managers](https://ocw.mit.edu/courses/15-040-game-theory-for-managers-spring-2004/pages/lecture-notes/)：核验同时与序贯寡头竞争、承诺和进入的分析顺序。
<!-- bilingual-en:start -->
- [[01_Math/03_game theory/第1次作业.pdf|Game Theory Assignment 1]]—[[01_Math/03_game theory/第3次作业.pdf|Game Theory Assignment 3]]: support the course coverage, classroom models, and problem context.
- [MIT OCW 14.12 Game Theory lecture notes](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/pages/lecture-notes/): verifies strategic-form and extensive-form games, Nash and Bayesian equilibrium, and refinements.
- [MIT OCW 15.040 Game Theory for Managers](https://ocw.mit.edu/courses/15-040-game-theory-for-managers-spring-2004/pages/lecture-notes/): verifies the analysis of simultaneous and sequential oligopoly competition, commitment, and entry.
<!-- bilingual-en:end -->
