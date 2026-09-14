---
aliases:
  - "Cournot 竞争把各企业同时选择的产量作为策略，并用总产量决定市场价格，因此每家利润取决于所有企业的产量"
  - Cournot competition is simultaneous quantity competition with price determined by aggregate output
  - 古诺竞争
student_os: knowledge-atom
atom_id: GT-OLI-002
atom_set: oligopoly-competition
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[策略式博弈]]"
related:
  - "[[Bertrand竞争]]"
  - "[[寡头模型选择]]"
leads_to:
  - "[[Cournot反应函数]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# Cournot 竞争把各企业同时选择的产量作为策略，并用总产量决定市场价格，因此每家利润取决于所有企业的产量
<!-- bilingual-en:start -->
*Cournot competition treats firms' simultaneous quantity choices as strategies and lets aggregate output determine market price, so each firm's profit depends on every firm's output*
<!-- bilingual-en:end -->

> [!summary] 竞争的是产量，不是直接报价
> 每家企业在不知道对手本轮实际产量的情况下同时选 $q_i$。市场把 $Q=\sum_i q_i$ 映射为价格 $P(Q)$；企业收入是自身销量乘共同市场价，所以对手增产会通过压低价格改变我的最优产量。
>
> <!-- bilingual-en:start -->
> Each firm chooses $q_i$ simultaneously, without observing its rivals' current choices. The market maps total output $Q=\sum_i q_i$ into a price $P(Q)$. Because a firm's revenue is its own sales times the common market price, rival output changes its optimal quantity through the market-price effect.
> <!-- bilingual-en:end -->

## 把现实叙述写成策略式博弈

对 $n$ 家企业，标准 Cournot 模型至少要写出：

$$
S_i=[0,\infty),\qquad
Q=\sum_{j=1}^n q_j,\qquad
\pi_i(q_i,q_{-i})=q_iP(Q)-C_i(q_i).
$$

策略是企业对本期产量的一次选择；它不是“若对手生产多少，我就生产多少”的整条反应函数。反应函数是从利润最大化推出来的分析对象，见 [[Cournot反应函数]]。

同时行动也不表示企业彼此无知。模型通常假设需求、成本和行动集合为共同知识；企业知道对手会理性响应，只是在作出本轮选择时不能观察对手本轮已经选出的 $q_j$。均衡要让每家的预期与实际选择相容，见 [[纳什均衡]]。

<!-- bilingual-en:start -->
For $n$ firms, the strategic-form primitives are $S_i=[0,\infty)$ and $\pi_i=q_iP(\sum_jq_j)-C_i(q_i)$. A strategy is the current quantity choice, not the entire reaction curve. Simultaneous play does not mean ignorance of demand or costs; it means that current rival quantities are not observed before choosing. Nash equilibrium makes each firm's optimizing conjecture consistent with actual quantities.
<!-- bilingual-en:end -->

## 模型名称不替代边界条件

- 若企业先后选择产量，且后行动者观察到先行量，模型转为 [[Stackelberg竞争]]。
- 若企业选择价格而非产量，进入 [[Bertrand竞争]] 或差异化价格模型。
- 若产能先锁定、随后再定价，需要把两个阶段都保留；不能只凭最终观察到的价格或产量倒推模型。
- 若产品并非同质、各企业面对不同需求或成本，利润函数必须相应改写；“Cournot”不等于“所有企业对称”。

<!-- bilingual-en:start -->
The label does not replace the boundary conditions. Observable sequential quantity choice is Stackelberg; simultaneous price choice is Bertrand or another pricing model; capacity followed by pricing is a multi-stage game; and differentiated products or asymmetric costs require modified payoff functions.
<!-- bilingual-en:end -->

> [!question]- 自检
> 题目说两家企业知道同一条需求曲线，并且“各自决定生产多少，市场随后按总产量出清”。企业的策略变量和价格分别是什么？
>
> **答案：** 策略变量是各自的产量 $q_i$；价格不是企业直接选择的策略，而是由总产量 $Q=q_1+q_2$ 通过逆需求 $P(Q)$ 决定。

## 来源与核验

- MIT 14.126, [*Game Theory, Lecture Notes*, Cournot Duopoly, pp. 53–55](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)：核对策略集合、总供给、利润函数与同时产量选择。
- MIT 15.010/15.011, [*The Basics of Game Theory*, pp. 1–2](https://ocw.mit.edu/courses/15-010-economic-analysis-for-business-decisions-fall-2004/807ba86e100d349ef73c294b9e720931_the_bsc_game_thy.pdf)：核对 Cournot 的同质品、已知需求与同时产量定义。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#3.1. 模型信息|本地课程：线性 Cournot 设置]]与[[01_Math/03_game theory/第1次作业补充.pdf|补充习题 2.2]]：支持课程中的玩家、产量策略、线性需求与 $n$ 企业语境。
