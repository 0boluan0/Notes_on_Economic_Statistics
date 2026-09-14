---
aliases:
  - "Samuelson 条件把同一公共品单位上的个人 MRS 纵向相加"
  - "The Samuelson condition vertically sums individual MRS values"
  - "Samuelson 公共品有效条件"
student_os: knowledge-atom
atom_id: PF-PG-004
atom_set: public-goods
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[物品的竞争性与排他性]]"
part_of:
  - "[[公共品、搭便车与 Samuelson 条件.canvas]]"
related:
  - "[[公共品搭便车]]"
  - "[[Samuelson 条件的实现缺口]]"
---

# Samuelson 条件把同一公共品单位上的个人 MRS 纵向相加
<!-- bilingual-en:start -->
*The Samuelson condition vertically sums individual MRS values for the same unit of a public good*
<!-- bilingual-en:end -->

> [!summary] 有效条件
> 在可微、内点且没有其他未处理扭曲的标准纯公共品模型中，Pareto 有效供给量满足
> $$\sum_i MRS_{G,x}^{i}=MRT_{G,x}.$$
> 若用货币作为计价物，同一条件可写成所有人对下一单位 $G$ 的边际支付意愿之和等于边际成本。

这里 $MRS_{G,x}^{i}=MU_G^i/MU_x^i$，表示个人 $i$ 愿为多一单位 $G$ 放弃多少单位私人物品 $x$；$MRT_{G,x}$ 表示社会多生产一单位 $G$ 必须牺牲多少单位 $x$。两边因此使用同一种计价物。

纵向相加来自公共品的共同消费约束。若社会多提供一单位 $G$，每个人都同时得到这一单位；因而要在**同一数量**上把个人边际估值相加。与此相对，私人物品在给定共同价格下把各人需求数量横向相加，因为一单位只由一人消费。

一个最小数值例：在当前供给量上，额外一单位防洪使两户的边际支付意愿分别为 60 和 40，而边际成本为 80。此时社会边际收益是 $60+40=100$，高于 80，因而还应增加供给。只看任一家的 60 或 40 都会漏掉对另一家的同步收益。

等号必须保留模型边界。在通常的凹性条件下，若零供给是最优角点，则边界处满足 $\sum_i MRS_i\le MRT$；若容量上限 $\bar G$ 是最优角点，则边界处满足 $\sum_i MRS_i\ge MRT$。若还有拥挤、分配效应、扭曲性筹资或多个交互政策，则不能把这个简式当成完整政策解。

> [!question]- 自检
> 为什么不能把两户对防洪的需求数量横向相加，像普通商品那样得到市场需求？
>
> **答案：** 防洪的同一实物单位同时保护两户，不是分别把两单位卖给两户。因此在同一供给量上相加的是边际估值，而不是数量。

## 来源与核验

- [Samuelson (1954), “The Pure Theory of Public Expenditure”](https://doi.org/10.2307/1925895)：原始提出集体消费品的 Pareto 有效条件及个人 MRS 曲线的纵向相加解释。
- [MIT OCW 14.41, Lecture 04 transcript](https://ocw.mit.edu/courses/14-41-public-finance-and-public-policy-fall-2024/1Rr_DjnI_7PLygHHTz1ALsghEdxpM1Fbl_transcript.pdf)：核对私人物品横向加总数量、公共品纵向加总边际支付意愿，以及个人 MRS 之和等于边际成本的区别。
- [MIT OCW 14.41, Lecture 05 transcript](https://ocw.mit.edu/courses/14-41-public-finance-and-public-policy-fall-2024/1pIqH99D-YxwcN7FbFRwD2aO2JWo2NUiI_transcript.pdf)：核对私人物品与公共品的边际条件区别及烟花数值例。
