---
aliases:
  - "在端点企业、均匀消费者、线性运输成本 t>0、相同边际成本 c≥0、非负连续定价且消费者保留价值 v≥c+2t 的单位 Hotelling 线性城市中，对称 Nash 均衡价格为 c+t"
  - In a unit Hotelling linear city with endpoint firms, uniform consumers, linear transport cost t>0, common marginal cost c≥0, nonnegative continuous prices, and consumer reservation value v≥c+2t, the symmetric Nash equilibrium price is c+t
student_os: knowledge-atom
atom_id: GT-OLI-014
atom_set: oligopoly-competition
atom_type: equilibrium-result
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hotelling反应函数]]"
  - "[[纳什均衡]]"
related:
  - "[[Bertrand边际成本均衡]]"
  - "[[均衡与效率]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在端点企业、均匀消费者、线性运输成本 t>0、相同边际成本 c≥0、非负连续定价且消费者保留价值 v≥c+2t 的单位 Hotelling 线性城市中，对称 Nash 均衡价格为 c+t
<!-- bilingual-en:start -->
*In a unit Hotelling linear city with endpoint firms, uniform consumers, linear transport cost $t>0$, common marginal cost $c\ge0$, nonnegative continuous prices, and consumer reservation value $v\ge c+2t$, the symmetric Nash equilibrium price is $c+t$*
<!-- bilingual-en:end -->

> [!summary] 联立内点反应只给候选，角点比较才把它确认为均衡
> 两条对称反应函数交于 $c+t$。但 Nash 结论还要求任何企业在对手报 $c+t$ 时，都不能靠抢下全部市场或退出市场获得更高利润；本卡把这两个角点和内点利润一起比较。
>
> <!-- bilingual-en:start -->
> The two symmetric interior best responses intersect at $c+t$. A Nash claim additionally requires checking that neither capturing the whole market nor abandoning it beats the interior price when the rival charges $c+t$.
> <!-- bilingual-en:end -->

## 对称候选

每位消费者购买一单位产品可得共同保留价值 $v$，不购买的效用规范为 0。条件 $v\ge c+2t$ 保证：当对手报 $c+t$ 时，即使本方任意单边提价，离对手最远的消费者也能以不超过 $c+2t$ 的总成本向对手购买。因此下面所有相关偏离仍可把总市场需求规范为 1。

由 [[Hotelling反应函数]]

$$
p_1=\frac{t+c+p_2}{2},
\qquad
p_2=\frac{t+c+p_1}{2},
$$

对称解是

$$
p_1^*=p_2^*=c+t.
$$

此时无差异消费者位于 $1/2$，每家销量为 $1/2$，每家利润为 $t/2$。

<!-- bilingual-en:start -->
Each consumer has reservation value $v$ and an outside option normalized to zero. The condition $v\ge c+2t$ keeps the market covered against every unilateral deviation when the rival charges $c+t$. Solving the two interior best responses gives the symmetric candidate $(c+t,c+t)$. The indifferent consumer is at $1/2$, so each firm sells one half and earns profit $t/2$.
<!-- bilingual-en:end -->

## 全局偏离核验

固定对手价格 $p_j=c+t$。把无差异消费者位置截在 $[0,1]$ 后，本方销量与利润可按自身价格分成三段：

1. **$0\le p_i\le c$：** 本方取得全部市场，但 $\pi_i=p_i-c\le0$。
2. **$c<p_i<c+2t$：** 市场内点分割。令 $z=p_i-c$，则

   $$
   \pi_i=z\left(1-\frac{z}{2t}\right).
   $$

   这是严格凹二次式，在 $z=t$，即 $p_i=c+t$ 时达到全段最大值 $t/2$。
3. **$p_i\ge c+2t$：** 本方失去全部市场，利润为 0。

两个角点区域的最高利润都不超过 0，严格低于内点候选的 $t/2$。因此 $c+t$ 是对 $c+t$ 的全局最佳反应，价格对 $(c+t,c+t)$ 构成 Nash 均衡。这里没有声称所有 Hotelling 变体都共享该价格，也没有额外声称这个均衡在所有扩展模型中唯一。

<!-- bilingual-en:start -->
Against $p_j=c+t$, a price at or below $c$ captures the whole market but earns at most zero. A price between $c$ and $c+2t$ earns $z(1-z/(2t))$ for $z=p_i-c$, which is uniquely maximized at $z=t$ with profit $t/2$. A price at or above $c+2t$ loses the market and earns zero. Thus $c+t$ is the global best response to itself, not merely a local interior candidate.
<!-- bilingual-en:end -->

> [!question]- 自检
> 仅仅把两条内点反应函数联立，为什么还不足以确认 $(c+t,c+t)$ 是 Nash 均衡？
>
> **答案：** 内点一阶条件没有比较“降价后拿下全部市场”和“高价后没有销量”的角点。只有证明这些区域的利润也不超过 $t/2$，才能确认没有全局有利偏离。

## 来源与核验

- MIT 14.271, [*Static Competition and Models of Differentiation, Part 1*, Hotelling Competition](https://ocw.mit.edu/courses/14-271-industrial-organization-i-fall-2022/mit14_271_f22_lec5slides.pdf)：核对对称价格 $p=c+t$、内点需求与正加价。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#5. Hotelling 模型|本地课程：Hotelling 模型]]：核对本库线性城市设定与对称解。
- [[Hotelling反应函数]]拥有内点最优化；本卡独立补上全局角点比较并只拥有对称均衡结论。
