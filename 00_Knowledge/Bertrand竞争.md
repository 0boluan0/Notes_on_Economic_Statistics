---
aliases:
  - "Bertrand 竞争把各企业同时选择的价格作为策略，并由相对价格对应的需求分配规则决定各自销量与利润"
  - Bertrand competition is simultaneous price choice with demand allocated by relative prices
  - 伯特兰竞争
student_os: knowledge-atom
atom_id: GT-OLI-006
atom_set: oligopoly-competition
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[策略式博弈]]"
related:
  - "[[Cournot竞争]]"
  - "[[寡头模型选择]]"
leads_to:
  - "[[Bertrand边际成本均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# Bertrand 竞争把各企业同时选择的价格作为策略，并由相对价格对应的需求分配规则决定各自销量与利润
<!-- bilingual-en:start -->
*Bertrand competition treats firms' simultaneous prices as strategies and uses a demand-allocation rule based on relative prices to determine each firm's sales and profit*
<!-- bilingual-en:end -->

> [!summary] “企业定价”仍不足以定义模型
> Bertrand 的关键不是表面上出现价格，而是价格怎样把消费者分给企业。在同质品基准中，低价企业取得全部需求、同价企业平分需求；这使销量在两家价格相等处跳变，也使普通光滑一阶条件失效。
>
> <!-- bilingual-en:start -->
> Prices alone do not define the game; the demand-allocation rule does. In the homogeneous-good benchmark, the lower-priced firm serves the whole market and equal-priced firms split demand. Sales jump at a price tie, so a smooth first-order condition is generally unavailable.
> <!-- bilingual-en:end -->

## 同质品双寡头的策略与销量

两家企业同时选择 $p_i\in[0,\infty)$。设市场总需求为 $D(p)$。标准同质品规则写成

$$
q_i(p_i,p_j)=
\begin{cases}
D(p_i), & p_i<p_j,\\[2mm]
\tfrac12D(p_i), & p_i=p_j,\\[2mm]
0, & p_i>p_j.
\end{cases}
$$

若成本为 $C_i(q_i)$，利润是

$$
\pi_i(p_i,p_j)=p_iq_i-C_i(q_i).
$$

这里的策略是价格，销量由两家价格共同诱导。企业不能在同一个静态模型里同时把价格和销量都当作彼此独立的策略；若现实中先定容量、后定价，需要建立多阶段博弈。

<!-- bilingual-en:start -->
In the homogeneous-good duopoly, each firm chooses a price. The lower-priced firm serves $D(p_i)$, an equal-priced firm serves half, and the higher-priced firm serves zero. Profit is induced by the price pair and the cost of serving the resulting quantity. Price and quantity are not two independent strategic choices in this one-stage model; capacity followed by pricing is a different, multi-stage game.
<!-- bilingual-en:end -->

## 为什么不能直接求导

当 $p_i$ 从略高于 $p_j$ 降到略低于 $p_j$ 时，企业销量可从 0 跳到全部市场。利润函数在价格相等处通常不连续或不可微，因此“对 $p_i$ 求导并令其等于 0”会漏掉最关键的降价偏离。正确方法是按 $p_i<p_j$、$p_i=p_j$ 与 $p_i>p_j$ 分情况比较利润。

差异化产品会让企业即使价格略高仍保留部分需求，利润函数可能变得光滑；[[Hotelling价格竞争]]就是一个明确例子。这不是在同质品公式上加一项装饰，而是换了需求分配机制。

<!-- bilingual-en:start -->
At a tie, a tiny price change can move a firm from zero sales to the whole market. The key deviation is therefore missed by blindly differentiating. The homogeneous benchmark must be analysed piecewise. Differentiation can smooth demand and preserve positive sales for the higher-priced firm, as in Hotelling; that changes the mechanism rather than merely decorating the same formula.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两家同质品企业报价相同。为什么仅检验该共同价格附近的导数为 0 不能证明均衡？
>
> **答案：** 因为价格相等处的销量分配会跳变；略微降价可能让一家从一半市场变成全部市场。利润在该点通常不可微，必须直接比较这一离散需求变化带来的偏离利润。

## 来源与核验

- MIT 14.12, [*Chapter 7: Application—Imperfect Competition*, §7.2](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/a870a72380a584e8d1ffd2b34fa24c9e_MIT14_12F12_chapter7.pdf)：核对同时定价、低价者取得需求、同价平分及销量的分段定义。
- MIT 15.010/15.011, [*The Basics of Game Theory*, p. 3](https://ocw.mit.edu/courses/15-010-economic-analysis-for-business-decisions-fall-2004/807ba86e100d349ef73c294b9e720931_the_bsc_game_thy.pdf)：核对同质品同时价格竞争的模型定义。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#4.1. 基本信息|本地课程：Bertrand 设置]]与[[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#4.2. 纳什均衡|分段偏离分析]]：核对课程需求分配与不用普通一阶条件的提醒。
