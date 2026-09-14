---
aliases:
  - "在单位线性城市、端点定位、均匀消费者、线性运输成本 t>0、全市场覆盖与内点分割下，空间差异使 Hotelling 企业的需求随双方价格连续变化"
  - In a covered unit linear city with endpoint firms, uniformly distributed consumers, linear transport cost t>0, and an interior split, spatial differentiation makes each Hotelling firm's demand vary continuously with both prices
  - Hotelling线性城市内点需求
student_os: knowledge-atom
atom_id: GT-OLI-017
atom_set: oligopoly-competition
atom_type: differentiated-demand-system
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hotelling价格竞争]]"
related:
  - "[[Bertrand基准边界]]"
leads_to:
  - "[[Hotelling反应函数]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在单位线性城市、端点定位、均匀消费者、线性运输成本 t>0、全市场覆盖与内点分割下，空间差异使 Hotelling 企业的需求随双方价格连续变化
<!-- bilingual-en:start -->
*In a covered unit linear city with endpoint firms, uniformly distributed consumers, linear transport cost $t>0$, and an interior split, spatial differentiation makes each Hotelling firm's demand vary continuously with both prices*
<!-- bilingual-en:end -->

> [!summary] 空间差异让略高价格不再丢掉全部市场
> 消费者比较的不只是标价，还包括到企业位置的运输成本。离某家更近的消费者可能在它价格略高时仍购买，因此需求随相对价格连续变化。运输成本参数 $t$ 在这个模型里也是产品差异强度，使略高价格仍可能保留需求，并为正加价创造可能。
>
> <!-- bilingual-en:start -->
> Consumers compare posted price plus travel cost. A consumer sufficiently close to a firm may keep buying even when that firm charges slightly more, so demand changes continuously with relative prices. The transport-cost parameter $t$ also measures differentiation, lets a slightly higher price retain demand, and thereby creates the possibility of a positive markup.
> <!-- bilingual-en:end -->

## 从无差异消费者得到需求

把企业 1 放在 0，企业 2 放在 1；消费者 $x$ 均匀分布在 $[0,1]$。两家位置固定，运输成本为每单位距离 $t>0$，消费者保留价值足够高使市场完全覆盖。消费者 $x$ 购买两家产品的总成本分别为

$$
p_1+tx,
\qquad
p_2+t(1-x).
$$

无差异消费者满足

$$
p_1+t\hat x=p_2+t(1-\hat x),
$$

所以

$$
\hat x=\frac12+\frac{p_2-p_1}{2t}.
$$

在 $0<\hat x<1$ 的内点区域，企业 1 的需求为 $q_1=\hat x$，企业 2 的需求为 $1-\hat x$。

<!-- bilingual-en:start -->
Firm 1 is at 0 and firm 2 at 1, consumers are uniform, and total purchase cost is posted price plus linear travel cost. The indifferent consumer is $\hat x=1/2+(p_2-p_1)/(2t)$. Under full coverage and an interior split, firm 1 serves $\hat x$ and firm 2 serves $1-\hat x$.
<!-- bilingual-en:end -->

## 这张卡只拥有需求系统

本卡的原子结论止于

$$
q_1(p_1,p_2)=\frac12+\frac{p_2-p_1}{2t},
\qquad
q_2(p_1,p_2)=1-q_1(p_1,p_2),
$$

并明确它只适用于 $0<\hat x<1$ 的内点。给定成本后的利润最大化属于 [[Hotelling反应函数]]；联立反应并核验角点后的价格结果属于 [[Hotelling均衡]]。这样“需求怎样分配”“企业怎样最优化”和“双方在哪里同时最优”不会被压成一个节点。

<!-- bilingual-en:start -->
This atom owns only the interior demand system. Profit maximization given costs belongs to [[Hotelling反应函数]], while the jointly optimal price pair and its corner checks belong to [[Hotelling均衡]]. Separating demand, optimization, and equilibrium prevents three different claims from being stored as one node.
<!-- bilingual-en:end -->

## 公式的三道边界

- 若保留价值不够高，部分消费者可能不购买，市场不再完全覆盖。
- 若价格差太大，$\hat x$ 会落到 $[0,1]$ 外，需求进入角点，内点需求公式不能继续外推。
- 若企业可先选择位置、运输成本非线性或消费者分布不均匀，需求和均衡都要重新推导。

因此不能把 $t=0$ 直接代进内点公式并称作一个有效的同质品 Bertrand 极限均衡：当差异消失时，需求分配在价格相等处退化为跳变结构，内点公式的适用区域也随之消失。

<!-- bilingual-en:start -->
The formula requires full coverage, an interior indifferent consumer, fixed endpoint locations, linear transport cost, and a uniform distribution. Setting $t=0$ does not produce a valid homogeneous-Bertrand limit equilibrium by direct substitution; the interior region collapses as demand approaches the discontinuous homogeneous-good allocation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对手 $p_2$ 提高时，无差异消费者 $\hat x$ 向哪边移动，企业 1 的需求怎样变化？
>
> **答案：** $\hat x=1/2+(p_2-p_1)/(2t)$ 随 $p_2$ 上升而向企业 2 的方向移动；企业 1 服务区间 $[0,\hat x]$ 变长，需求增加。利润最优化见 [[Hotelling反应函数]]。

## 来源与核验

- MIT 14.271, [*Static Competition and Models of Differentiation, Part 1*, Hotelling Competition](https://ocw.mit.edu/courses/14-271-industrial-organization-i-fall-2022/mit14_271_f22_lec5slides.pdf)：核对无差异消费者、内点需求及全覆盖/角点边界。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#5. Hotelling 模型|本地课程：Hotelling 模型]]：核对端点、均匀消费者、运输成本、本库记号和内点推导。
- [[01_Math/03_game theory/第1次作业补充.pdf|补充习题 2.5]]：支持课程把差异化价格竞争作为同质品 Bertrand 之后的独立问题；该题使用另一需求函数，不用于支持本卡的线性城市公式。
- [[Hotelling价格竞争]]拥有空间差异化价格竞争的定义；本卡只拥有这组具体假设下的内点需求分配。
