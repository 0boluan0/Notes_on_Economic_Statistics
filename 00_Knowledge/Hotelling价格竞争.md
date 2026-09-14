---
aliases:
  - "Hotelling 价格竞争把企业置于产品空间的位置上，让消费者按标价与距离成本之和选择，并让企业同时决定价格"
  - Hotelling price competition places firms in a product space, lets consumers choose by posted price plus distance cost, and has firms choose prices simultaneously
  - Hotelling空间竞争
student_os: knowledge-atom
atom_id: GT-OLI-009
atom_set: oligopoly-competition
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bertrand竞争]]"
related:
  - "[[Bertrand基准边界]]"
  - "[[寡头模型选择]]"
leads_to:
  - "[[Hotelling需求系统]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# Hotelling 价格竞争把企业置于产品空间的位置上，让消费者按标价与距离成本之和选择，并让企业同时决定价格
<!-- bilingual-en:start -->
*Hotelling price competition places firms in a product space, lets consumers choose by posted price plus distance cost, and has firms choose prices simultaneously*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Hotelling 价格竞争把产品差异表示成“距离”。企业位于产品空间的不同位置并同时定价；消费者比较每家企业的标价与到该位置的距离成本之和。因而离某家企业较近的消费者可能在它价格略高时仍购买，需求不再像同质品 Bertrand 基准那样只由最低标价全部拿走。
>
> <!-- bilingual-en:start -->
> Hotelling price competition represents product differentiation as distance. Firms occupy different positions and set prices simultaneously; consumers compare posted price plus the cost of reaching each position. A nearby consumer may therefore remain with a slightly more expensive firm, unlike the winner-take-all allocation in homogeneous-good Bertrand competition.
> <!-- bilingual-en:end -->

## 模型怎样写

设企业 $i$ 的位置为 $z_i$、价格为 $p_i$，消费者位置为 $x$。消费者从企业 $i$ 购买的广义成本可写为

$$
p_i+T(|x-z_i|),
$$

其中 $T(\cdot)$ 把产品位置差异转成运输、匹配或转换成本。消费者选择广义成本最低的企业；企业再根据这条选择规则诱导出的需求最大化利润。具体需求取决于企业位置、消费者分布、距离成本、保留价值和市场是否覆盖，不能仅凭“Hotelling”一词直接写出一个固定公式。

<!-- bilingual-en:start -->
A consumer at $x$ faces generalized cost $p_i+T(|x-z_i|)$ from firm $i$. Consumer choice induces each firm's demand, and firms choose prices against that demand. Exact quantities depend on locations, the consumer distribution, the distance-cost function, reservation values, and market coverage; the model name alone does not determine one demand formula.
<!-- bilingual-en:end -->

## 与具体线性城市版本分开

最常用的特例把两家企业固定在单位线性城市两端，消费者均匀分布，距离成本线性，并先分析全市场覆盖、无差异消费者位于内部的分支。在这些额外条件下才得到连续的内点需求系统，见 [[Hotelling需求系统]]；给定成本后的利润最优化属于 [[Hotelling反应函数]]，联立反应并比较角点属于 [[Hotelling均衡]]。

若企业也能选择位置、消费者并非均匀分布、距离成本非线性或部分消费者选择不购买，需求与均衡都要重新推导。产品差异能改变 Bertrand 的降价逻辑，但不自动推出某个固定加价。

<!-- bilingual-en:start -->
The familiar unit linear-city case adds endpoint firms, uniform consumers, linear travel cost, market coverage, and an interior split. Those assumptions generate the separate [[Hotelling需求系统|Hotelling demand system]]. Profit optimization and equilibrium are further atoms. Endogenous locations, nonuniform consumers, nonlinear distance costs, or uncovered markets require a new derivation; differentiation alone does not imply one universal markup.
<!-- bilingual-en:end -->

> [!question]- 最小例子
> 两家咖啡店标价相同，但消费者到两家店的距离不同。Hotelling 模型首先用什么量解释消费者选择？
>
> **答案：** 比较“标价 + 距离成本”的广义成本；价格相同并不意味着每个消费者对两家店无差异。

## 来源与核验

- MIT 14.271, [*Static Competition and Models of Differentiation, Part 1*, Hotelling Competition](https://ocw.mit.edu/courses/14-271-industrial-organization-i-fall-2022/mit14_271_f22_lec5slides.pdf)：核对空间位置、消费者距离成本与差异化价格竞争的模型结构。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#5. Hotelling 模型|本地课程：Hotelling 模型]]：核对线性城市作为空间差异化价格竞争的课程入口。
- [[Hotelling需求系统]]拥有端点、均匀消费者、线性距离成本、全覆盖与内点分割下的具体需求公式；本卡只拥有 Hotelling 价格竞争的模型定义。
