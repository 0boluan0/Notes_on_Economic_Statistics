---
aliases:
  - "在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 的 Cournot 双寡头中，唯一 Nash 均衡使每家产量等于 (a-c)/(3b)"
  - Under P(Q)=max{a-bQ,0}, 0<c<a, b>0, and identical constant marginal cost c, the unique Cournot-duopoly Nash equilibrium gives each firm output (a-c)/(3b)
student_os: knowledge-atom
atom_id: GT-OLI-004
atom_set: oligopoly-competition
atom_type: equilibrium-result
status: source-checked
mastery_state: unassessed
requires:
  - "[[Cournot反应函数]]"
  - "[[纳什均衡]]"
related:
  - "[[均衡与效率]]"
leads_to:
  - "[[Cournot厂商数]]"
  - "[[Stackelberg线性均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 的 Cournot 双寡头中，唯一 Nash 均衡使每家产量等于 (a-c)/(3b)
<!-- bilingual-en:start -->
*Under $P(Q)=\max\{a-bQ,0\}$, $0<c<a$, $b>0$, and identical constant marginal cost $c$, the unique Cournot-duopoly Nash equilibrium gives each firm output $(a-c)/(3b)$*
<!-- bilingual-en:end -->

> [!summary] 均衡是相互最佳反应，不是平分垄断产量
> 两家企业同时选产量时，每家都把对手产量当作既定，再选择自己的利润最大量。只有两条反应函数同时成立，预期才与实际相符。对称来自模型参数对称和交点解，不是先把总量猜出来再机械除以二。
>
> <!-- bilingual-en:start -->
> Each firm optimizes against the other's quantity. Equilibrium requires both reaction functions to hold at once, making conjectures consistent with actual choices. Symmetry follows from the symmetric primitives and the solution; it is not obtained by guessing an industry quantity and dividing it in half.
> <!-- bilingual-en:end -->

## 联立反应函数

沿用 $P(Q)=\max\{a-bQ,0\}$、$a>c>0$、$b>0$ 与相同边际成本 $c$。正边际成本排除了“价格已经为 0 时仍可无成本任意过量生产”的弱最优退化情形。内点均衡满足

$$
q_1=\frac{a-c-bq_2}{2b},
\qquad
q_2=\frac{a-c-bq_1}{2b}.
$$

两式相减得到 $q_1=q_2$，再代回任一式：

$$
q_1^*=q_2^*=\frac{a-c}{3b}.
$$

所以

$$
Q^*=\frac{2(a-c)}{3b},
\qquad
P^*=a-bQ^*=\frac{a+2c}{3},
$$

每家利润为

$$
\pi_i^*=(P^*-c)q_i^*=\frac{(a-c)^2}{9b}.
$$

因为 $a>c>0$，均衡产量为正，且零价格下的过量生产会产生严格成本，故不存在那类退化的额外最佳反应。两条分段线性最佳反应在可行域只有这一交点，因此这是唯一 Nash 均衡。

<!-- bilingual-en:start -->
Solving the two best-response equations yields $q_1^*=q_2^*=(a-c)/(3b)$. Hence $Q^*=2(a-c)/(3b)$, $P^*=(a+2c)/3$, and each firm's profit is $(a-c)^2/(9b)$. Since $a>c>0$, the solution lies on the positive branch and zero-price overproduction is strictly costly. The two truncated linear best responses have no other feasible intersection, so the Nash equilibrium is unique.
<!-- bilingual-en:end -->

## 怎样验证，而不是只记结果

一份完整检查应回到每家企业自己的偏离：给定 $q_2^*$，$q_1^*$ 是否最大化 $\pi_1$；给定 $q_1^*$，$q_2^*$ 是否最大化 $\pi_2$。总利润最大或两家产量相等都不是 Nash 的定义。垄断者面对同一需求会选择总量 $(a-c)/(2b)$；Cournot 总量更高，说明两家独立企业没有把自己增产对另一家利润造成的价格损失全部内部化。

这个福利比较只适用于当前需求、成本和市场覆盖。若要判断效率，应另行明确消费者剩余与可行结果，见 [[均衡与效率]]。

<!-- bilingual-en:start -->
Verification returns to unilateral deviations: each quantity must maximize that firm's own profit holding the rival at its equilibrium choice. Equal quantities or maximum total profit are not the Nash definition. A monopolist facing the same demand would choose total output $(a-c)/(2b)$, below Cournot total output, because independent firms do not internalize the full price effect their extra output imposes on the rival.
<!-- bilingual-en:end -->

> [!question]- 自检
> 有人把垄断总产量 $(a-c)/(2b)$ 平分，得到每家 $(a-c)/(4b)$，并称其为 Cournot 均衡。最短的反驳是什么？
>
> **答案：** 把 $q_2=(a-c)/(4b)$ 代入企业 1 的最佳反应，可得 $BR_1(q_2)=3(a-c)/(8b)$，不等于 $(a-c)/(4b)$；企业 1 有利可图地偏离，所以该组合不是 Nash 均衡。

## 来源与核验

- MIT 14.126, [*Game Theory, Lecture Notes*, Cournot Duopoly](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)：核对相互最佳反应、唯一交点及标准化线性模型的 $q_i^*=(1-c)/3$。
- MIT 14.12, [*Chapter 7: Application—Imperfect Competition*](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/a870a72380a584e8d1ffd2b34fa24c9e_MIT14_12F12_chapter7.pdf)：交叉核对 Cournot 双寡头反应函数、均衡产量、价格与利润。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#3.3. 对称均衡|本地课程：对称 Cournot 均衡]]：核对本库使用的 $a,b,c$ 记号与联立步骤。
