---
aliases:
  - 专利中间品厂按恒弹性需求加价出租耐用品并取得经营收益
  - Intermediate-input markups in the Romer model
student_os: knowledge-atom
atom_id: MACRO-ENDO-015
atom_type: proposition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 专利中间品厂按恒弹性需求加价出租耐用品并取得经营收益
<!-- bilingual-en:start -->
*A patented intermediate producer marks up the rental price of its durable under constant-elasticity demand and earns an operating return.*
<!-- bilingual-en:end -->

在 [[Romer品种扩张模型]] 中，中间品厂买下设计后，成为该品种的唯一供应者。最终品厂竞争性地选择机器用量，因此其边际产出决定支付意愿；中间品厂沿这条向下倾斜的需求曲线选择租价与数量。这里是 [[垄断利润最大化]] 的一个具体应用。
<!-- bilingual-en:start -->
In the [[Romer品种扩张模型|Romer expanding-variety model]], buying a design makes the intermediate producer the sole supplier of that variety. Competitive final producers choose machine quantities according to marginal productivity. The intermediate producer selects a rental price and quantity on the resulting downward-sloping demand curve, applying [[垄断利润最大化|monopoly profit maximization]].
<!-- bilingual-en:end -->

令 $\gamma=1-\alpha-\beta\in(0,1)$，并记 $B=H_Y^\alpha L^\beta>0$，其中 $H_Y,L$ 是最终品人力资本与劳动力。由最终品生产函数对该品种用量 $x$ 求导，得到反需求；$p$ 是**租价**：
<!-- bilingual-en:start -->
Let $\gamma=1-\alpha-\beta\in(0,1)$ and $B=H_Y^\alpha L^\beta>0$, where $H_Y,L$ are production human capital and labor. Differentiating final output with respect to the variety's quantity $x$ gives inverse demand. The price $p$ is a rental price.
<!-- bilingual-en:end -->

$$p(x)=\gamma Bx^{\gamma-1}.$$

每单位耐用品耗费 $\eta>0$ 单位最终品资源，不折旧；利率为 $r>0$，因此每期出租一单位的资金机会成本是 $r\eta$。原文还允许耐用品转回通用资本，使厂商可逐时调整数量。设计费已经支付后，经营选择为
<!-- bilingual-en:start -->
Each nondepreciating durable unit costs $\eta>0$ final-output units, so its flow opportunity cost at interest rate $r>0$ is $r\eta$. The original model also allows conversion back to general capital, permitting quantity adjustment at each date. After paying the design fee, the operating choice is:
<!-- bilingual-en:end -->

$$
\max_{x\ge0}\;\pi(x)=\gamma Bx^\gamma-r\eta x,
\qquad \pi'(x)=\gamma^2Bx^{\gamma-1}-r\eta.
$$

当 $x>0$ 时，$\pi''(x)=\gamma^2(\gamma-1)Bx^{\gamma-2}<0$；一阶导数从正无穷下降到 $-r\eta$。所以正的内点候选是唯一全局最大值，解为
<!-- bilingual-en:start -->
For positive $x$, the second derivative is negative, while the first derivative declines from positive infinity to $-r\eta$. The positive interior candidate is therefore the unique global maximum:
<!-- bilingual-en:end -->

$$
x=\left(\frac{\gamma^2 B}{r\eta}\right)^{1/(1-\gamma)},\qquad
p=\frac{r\eta}{\gamma}>r\eta,
$$
$$
\pi=(p-r\eta)x=(1-\gamma)px
=\gamma(1-\gamma)Bx^\gamma.
$$

例如 $\gamma=1/2$、$B=1$、$r=0.05$、$\eta=5$，则 $x=1$、租价 $p=0.5$、单位资金机会成本 $r\eta=0.25$，每期经营收益 $\pi=0.25$。这项收益尚未扣除最初设计投资；它的现值怎样支持设计购买，见 [[Romer专利价值]]。比较加价时要用 $p$ 对 $r\eta$，不能把租价与一次性建造成本 $\eta$ 混比。
<!-- bilingual-en:start -->
With $\gamma=1/2$, $B=1$, $r=0.05$ and $\eta=5$, the solution is $x=1$, rental price $0.5$, unit funding cost $0.25$ and operating return $0.25$. This return has not yet deducted the initial design investment. Its role in financing that purchase is explained by [[Romer专利价值|patent value]]. Compare the rental markup with $r\eta$, not with the one-time construction cost $\eta$.
<!-- bilingual-en:end -->

## 来源与核验

- Romer（1990），[Endogenous Technological Change](https://web.stanford.edu/~klenow/Romer_1990.pdf#page=17)，印刷S86–S87（PDF第17–18页），式(4)–(5)：反需求、租赁资金成本、可转回通用资本的设定、最优租价与经营收益。
- 已独立求导检查全局凹性，并逐项复算数例；$\gamma$ 是原文 $1-\alpha-\beta$ 的缩写，未混用其他品种模型的成本参数。
<!-- bilingual-en:start -->
Equations (4)–(5) supply demand and the operating problem. The derivatives, global maximum and numerical example were checked independently using the original rental-cost specification.
<!-- bilingual-en:end -->
