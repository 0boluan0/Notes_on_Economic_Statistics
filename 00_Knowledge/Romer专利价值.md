---
aliases:
  - Romer专利价值是某一设计带来的可取得经营收益流的贴现现值
  - Patent value in the Romer model
student_os: knowledge-atom
atom_id: MACRO-ENDO-016
atom_type: definition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# Romer专利价值是某一设计带来的可取得经营收益流的贴现现值
<!-- bilingual-en:start -->
*Patent value in the Romer model is the discounted present value of the operating returns that a particular design allows its owner to capture.*
<!-- bilingual-en:end -->

[[Romer品种扩张模型]]中的专利价值 $P_A(t)$，是持有某一设计后可取得的未来中间品经营收益，在时点 $t$ 的现值。下标 $A$ 标记设计资产，**不是全部 $A$ 个专利价值的总和**。未来收益先扣除了制造机器所占资本的资金机会成本，尚未扣除买入该设计的初始支出。
<!-- bilingual-en:start -->
In the [[Romer品种扩张模型|Romer expanding-variety model]], $P_A(t)$ is the time-$t$ present value of future operating returns obtainable from one design. The subscript identifies the design asset; it does not mean the total value of all $A$ patents. Operating returns deduct the funding cost of manufactured machines but not the initial expenditure on the design.
<!-- bilingual-en:end -->

设收益流为 $\pi(s)$、连续时间利率为 $r(u)$。永久专利且没有淘汰时，现值为
<!-- bilingual-en:start -->
Let $\pi(s)$ denote the operating-return flow and $r(u)$ the continuous-time interest rate. With a perpetual patent and no obsolescence, present value is:
<!-- bilingual-en:end -->

$$
P_A(t)=\int_t^\infty
\exp\!\left[-\int_t^s r(u)\,du\right]\pi(s)\,ds.
$$

这是 [[复利与贴现]] 在连续收益流上的应用。对上式求导可得资产回报关系；它同时包括经营收益和专利本身的价格变化：
<!-- bilingual-en:start -->
This applies [[复利与贴现|discounting]] to a continuous flow. Differentiating gives an asset-return relation containing both operating returns and changes in the patent's price:
<!-- bilingual-en:end -->

$$r(t)P_A(t)=\pi(t)+\dot P_A(t).$$

在本模型的平衡增长路径上，$r>0$、$\pi$ 和 $P_A$ 都恒定，因而可简写 $P_A=\pi/r$。若 $\dot P_A\ne0$，就须保留资产回报式中的价格变化项。经济的人均产出可以增长，而单个既有品种的利润和专利价值不增长；品种总数仍在增加。
<!-- bilingual-en:start -->
On this model's balanced path, positive $r$, operating returns and patent value are constant, giving $P_A=\pi/r$. When patent value changes, the asset-return equation must retain that price-change term. The economy can grow per person even while an individual existing variety's returns and patent value remain unchanged, because the number of varieties keeps increasing.
<!-- bilingual-en:end -->

例如 [[Romer中间品加价]] 的教学数例给出每期经营收益 $0.25$；按 $r=0.05$ 贴现的永久流价值为 $5$。若新进入者购买设计付 $5$，则进入时净现值为 $0.25/0.05-5=0$，以后每期依然取得 $0.25$。设计市场的竞争使价格等于可取得收益的现值，**自由进入零净现值与经营收益为正可以同时成立**。
<!-- bilingual-en:start -->
The example in [[Romer中间品加价|intermediate-input markups]] yields operating returns of $0.25$. A perpetual stream discounted at $0.05$ is worth $5$. An entrant paying $5$ for the design has entry net present value $0.25/0.05-5=0$ while still receiving $0.25$ in each subsequent period. Competition for designs equates their price to obtainable returns, so zero entry NPV is compatible with positive operating returns.
<!-- bilingual-en:end -->

这个区别与 [[生产者剩余与利润]] 的成本口径要求一致，但这里必须跨期比较：不能把永久收益流与一次性设计成本当作同一期的两个数字相减，也不能把未被专利权覆盖的 [[Romer研发知识外溢|后续研究收益]] 算进私人专利价格。
<!-- bilingual-en:start -->
This respects the cost-basis distinction in [[生产者剩余与利润|producer surplus versus profit]], but the comparison here is intertemporal. A perpetual flow and a one-time design expenditure cannot be subtracted as if they belonged to the same period. [[Romer研发知识外溢|Benefits to subsequent researchers]] that the owner cannot capture are also absent from private patent value.
<!-- bilingual-en:end -->

## 来源与核验

- Romer（1990），[Endogenous Technological Change](https://web.stanford.edu/~klenow/Romer_1990.pdf#page=18)，印刷S87（PDF第18页），式(6)–(6′)：设计价格、未来经营收益现值及常价下 $\pi=rP_A$；印刷S73、S90–S92（PDF第4、21–23页）：自由进入与平衡增长路径。
- 一般资产回报式由原文现值积分求导；数例独立复算并与经营收益卡使用同一组 $r,\pi$。这里未将常价下的简式推广到专利价格变化的路径。
<!-- bilingual-en:start -->
Equation (6) defines the present value, while (6′) gives the constant-price special case. The general asset-return relation was derived from that integral, and the numerical example was checked on the same cost basis as the markup example.
<!-- bilingual-en:end -->
