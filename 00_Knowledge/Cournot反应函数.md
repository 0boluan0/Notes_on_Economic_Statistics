---
aliases:
  - "在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 下，Cournot 企业的最佳反应是对手产量的截断递减函数"
  - Under P(Q)=max{a-bQ,0}, 0<c<a, b>0, and identical constant marginal cost c, a Cournot firm's best response is a truncated decreasing function of rival output
student_os: knowledge-atom
atom_id: GT-OLI-003
atom_set: oligopoly-competition
atom_type: best-response-derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[Cournot竞争]]"
  - "[[最佳反应]]"
related:
  - "[[战略替代与互补]]"
leads_to:
  - "[[Cournot均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 下，Cournot 企业的最佳反应是对手产量的截断递减函数
<!-- bilingual-en:start -->
*Under $P(Q)=\max\{a-bQ,0\}$, $0<c<a$, $b>0$, and identical constant marginal cost $c$, a Cournot firm's best response is a truncated decreasing function of rival output*
<!-- bilingual-en:end -->

> [!summary] 先固定对手产量，再优化自己的产量
> 对手多生产一单位，会降低共同市场价格和本方边际收益。在标准线性双寡头中，这把本方内点最优产量降低半单位；若对手产量已经足够大，非负约束把最佳反应截在 0，而不是继续给出负产量。
>
> <!-- bilingual-en:start -->
> Holding rival output fixed, an additional rival unit lowers the common market price and the firm's marginal revenue. In the standard linear duopoly, the interior best response falls by one half unit for each additional rival unit. Once rival output is sufficiently large, nonnegativity truncates the response at zero rather than allowing a negative quantity.
> <!-- bilingual-en:end -->

## 从利润函数得到完整反应函数

设

$$
P(Q)=\max\{a-bQ,0\},\qquad a>c>0,\quad b>0,
$$

两家企业边际成本均为 $c$。给定企业 2 的产量 $q_2$，企业 1 在正价格相关区域的利润为

$$
\pi_1(q_1;q_2)=\bigl[a-b(q_1+q_2)-c\bigr]q_1.
$$

其一阶导数和二阶导数是

$$
\frac{\partial\pi_1}{\partial q_1}=a-c-2bq_1-bq_2,
\qquad
\frac{\partial^2\pi_1}{\partial q_1^2}=-2b<0.
$$

因此内点候选为 $(a-c-bq_2)/(2b)$。加入 $q_1\ge0$ 后，完整最佳反应是

$$
BR_1(q_2)=\max\left\{0,\frac{a-c-bq_2}{2b}\right\},
$$

企业 2 对称。二阶导数为负保证正的内点候选是唯一利润最大点；当候选非正时，边界 $q_1=0$ 最优。

还需排除被截断需求隐藏的零价格区域。若 $q_1+q_2\ge a/b$ 且 $q_1>0$，则 $P=0$，企业 1 的利润为 $-cq_1<0$；改选 $q_1=0$ 至少得到 0。因此零价格区域中的任何正产量都不可能优于停产。这个比较与正价格区的严格凹性合在一起，才证明上式是全局完整最佳反应，而不只是局部候选。

<!-- bilingual-en:start -->
With inverse demand $P(Q)=\max\{a-bQ,0\}$ and constant marginal cost $c$, firm 1's positive-price profit conditional on $q_2$ is $[a-b(q_1+q_2)-c]q_1$. The first-order condition gives the interior candidate $(a-c-bq_2)/(2b)$, and the second derivative is negative. In the zero-price region, any positive output earns $-cq_1<0$ and is dominated by zero output. Combining this global boundary check with $q_1\ge0$ yields $BR_1(q_2)=\max\{0,(a-c-bq_2)/(2b)\}$.
<!-- bilingual-en:end -->

## 斜率表达的是战略替代

在反应函数的正值区间，

$$
\frac{dBR_1}{dq_2}=-\frac12.
$$

这不是说“企业讨厌对手”，而是说对手产量上升后，本方最优产量下降；按 [[战略替代与互补]] 的语言，产量在这个模型中是战略替代。斜率 $-1/2$ 来自特定的线性需求与恒定边际成本，不能推广成所有 Cournot 模型的固定数值。一般需求下应重新最大化 $q_iP(q_i+Q_{-i})-C_i(q_i)$，并检查角点与多重最佳反应。

<!-- bilingual-en:start -->
On the positive branch, $dBR_1/dq_2=-1/2$: quantities are strategic substitutes in this model. The numerical slope comes from linear demand and constant marginal cost, not from the word Cournot itself. Nonlinear demand or costs require a fresh optimization and may introduce corners or multiple best responses.
<!-- bilingual-en:end -->

> [!question]- 自检
> 直接由一阶条件算得 $q_1=-3$，能否把它写成企业 1 的 Cournot 最佳反应？
>
> **答案：** 不能。产量策略集合是非负的；负的内点候选表示利润在可行边界处由 $q_1=0$ 最大化。必须使用带 $\max\{0,\cdot\}$ 的完整反应函数。

## 来源与核验

- MIT 14.126, [*Game Theory, Lecture Notes*, eqs. 2.16–2.18](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)：核对线性 Cournot 利润最大化、非负截断和最佳反应图。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#3.2. 最优反应函数|本地课程：Cournot 最优反应]]：核对 $P=a-bQ$ 下的一阶条件、反应函数与策略替代解释。
- [[01_Math/03_game theory/第2次作业.pdf|作业 2，第 1 题]]：支持由残余需求推导双方 response curve 的课程练习语境。
