---
aliases:
  - Romer品种扩张模型是通过研发增加专门耐用品设计的三部门增长模型
  - Romer expanding-variety model
  - Romer 1990 model
student_os: knowledge-atom
atom_id: MACRO-ENDO-012
atom_type: definition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# Romer品种扩张模型是通过研发增加专门耐用品设计的三部门增长模型
<!-- bilingual-en:start -->
*The Romer expanding-variety model is a three-sector growth model in which research creates designs for additional specialized producer durables.*
<!-- bilingual-en:end -->

Romer（1990）的品种扩张模型把技术进步表示为可用机器设计的范围扩大：研发部门发明新设计，中间品部门依设计制造并出租专门耐用品，最终品部门使用它们生产消费品与投资品。新设计扩大生产选择，并通过专利收益激励研发；原模型没有设置新品种淘汰旧品种的机制。
<!-- bilingual-en:start -->
Romer (1990) represents technical progress as an expansion in available machine designs. Research creates designs; intermediate producers manufacture and rent the specialized durables; final producers use them to produce goods for consumption and investment. New designs expand productive opportunities, and patent returns motivate research. The original model does not make new varieties render old varieties obsolete.
<!-- bilingual-en:end -->

本卡采用原文版本：劳动力 $L>0$ 与总 [[人力资本]] $H>0$ 固定；$A>0$ 是可用设计的数量或连续模型中的范围，$x(i)$ 是第 $i$ 种耐用品的实物数量，$K$ 按所耗最终品资源计量。人力资本分为最终品部门的 $H_Y$ 和研发部门的 $H_A$。取 $\alpha,\beta>0$、$\alpha+\beta<1$，并缩写 $\gamma=1-\alpha-\beta$：
<!-- bilingual-en:start -->
This card uses the original specification. Labor $L$ and total [[人力资本|human capital]] $H$ are fixed. $A$ counts available designs, or measures their range in the continuum model; $x(i)$ is the physical quantity of durable $i$; and $K$ is measured in final-output resources. Human capital is allocated between production, $H_Y$, and research, $H_A$. Define $\gamma=1-\alpha-\beta$ with positive $\alpha,\beta$ and $\alpha+\beta<1$.
<!-- bilingual-en:end -->

$$
H_Y+H_A=H,\qquad
Y=H_Y^\alpha L^\beta\int_0^A x(i)^\gamma\,di.
$$

中间品厂每制造一单位耐用品消耗 $\eta>0$ 单位最终品资源，耐用品不折旧。它先取得永久专利，再把机器租给最终品厂。于是物质资本账和最终品用途账是
<!-- bilingual-en:start -->
An intermediate producer uses $\eta>0$ units of final-output resources to create each unit of a durable. Durables do not depreciate. The producer first obtains a perpetual patent and then rents machines to final producers. The capital and final-output resource accounts are therefore as follows.
<!-- bilingual-en:end -->

$$
K=\eta\int_0^A x(i)\,di,\qquad \dot K=Y-C.
$$

研发用人力资本和已有设计知识生产新设计，其中 $\delta>0$ 是研发效率：
<!-- bilingual-en:start -->
Research uses human capital and existing design knowledge to produce further designs, with research-productivity parameter $\delta>0$.
<!-- bilingual-en:end -->

$$\dot A=\delta H_A A.$$

研发的真实代价是研究人力不能同时用于当期最终品生产；支付工资和购买专利是相应的收入与资产交易，不能把专利购买额再作为一项额外最终品消耗重复扣除。$\delta$ 在这里不是 AK 模型的折旧率，$H$ 的部门分配也不等于 [[人力资本线性积累|人力资本存量的内生积累]]。
<!-- bilingual-en:start -->
Research uses human capital that could otherwise produce final output. Wages and patent purchases are the associated income and asset transactions; patent expenditure must not be deducted again as an additional final-good resource use. Here $\delta$ is not the depreciation parameter of an AK model, and allocating fixed $H$ is distinct from [[人力资本线性积累|accumulating the stock of human capital]].
<!-- bilingual-en:end -->

三部门的行为依次由 [[品种扩张的生产效率]]、[[Romer研发知识外溢]]、[[Romer中间品加价]]、[[Romer专利价值]] 和 [[Romer研发人力配置]] 展开。教材中另有“用最终品直接研发、机器用后完全折旧”的版本；其研发效率、资源账和增长率公式需要随整套模型一起更换。
<!-- bilingual-en:start -->
The mechanism is developed through [[品种扩张的生产效率|the productivity of variety]], [[Romer研发知识外溢|research spillovers]], [[Romer中间品加价|intermediate markups]], [[Romer专利价值|patent value]] and [[Romer研发人力配置|research allocation]]. A different textbook version uses final goods directly in research and machines that depreciate fully after use. Its research parameters, resource accounts and growth formula must be taken from that model together.
<!-- bilingual-en:end -->

## 来源与核验

- Romer（1990），[Endogenous Technological Change](https://web.stanford.edu/~klenow/Romer_1990.pdf#page=10)，印刷S79–S85（PDF第10–16页）：三部门、固定 $L,H$、生产式(1′)、资本账(2)、研发式(3)、永久专利和无淘汰设定。$\gamma$ 仅缩写原文的 $1-\alpha-\beta$。
- [Acemoglu，MIT 14.452，2024 Lecture 8](https://economics.mit.edu/sites/default/files/inline-files/Economic%20Growth%20Lecture%208%202024.pdf#page=4)，slides 4、6–8、36–37：核对 lab-equipment 与研究劳动版本的不同投入、折旧及研发方程，未把它们的参数代入本卡。
<!-- bilingual-en:start -->
Romer supplies the original three-sector specification. The MIT lecture is used to identify alternative teaching specifications and preserve the distinction between their equations and those used here.
<!-- bilingual-en:end -->
