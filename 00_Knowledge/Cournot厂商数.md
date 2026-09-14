---
aliases:
  - "在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 的 n≥1 家 Cournot 模型中，厂商数增加会提高唯一 Nash 均衡的总产量并把价格压向边际成本，但每家产量与利润下降"
  - Under P(Q)=max{a-bQ,0}, 0<c<a, b>0, and identical constant marginal cost c with n≥1 Cournot firms, more firms raise total output in the unique Nash equilibrium and push price toward marginal cost while reducing each firm's output and profit
student_os: knowledge-atom
atom_id: GT-OLI-005
atom_set: oligopoly-competition
atom_type: comparative-statics-result
status: source-checked
mastery_state: unassessed
requires:
  - "[[Cournot均衡]]"
related:
  - "[[均衡与效率]]"
  - "[[Bertrand边际成本均衡]]"
part_of:
  - "[[寡头竞争.canvas|寡头竞争]]"
---

# 在截断线性逆需求 P(Q)=max{a-bQ,0}、0<c<a、b>0 与相同恒定边际成本 c 的 n≥1 家 Cournot 模型中，厂商数增加会提高唯一 Nash 均衡的总产量并把价格压向边际成本，但每家产量与利润下降
<!-- bilingual-en:start -->
*Under $P(Q)=\max\{a-bQ,0\}$, $0<c<a$, $b>0$, and identical constant marginal cost $c$ with $n\ge1$ Cournot firms, increasing the number of firms raises total output in the unique Nash equilibrium and pushes price toward marginal cost while reducing each firm's output and profit*
<!-- bilingual-en:end -->

> [!summary] “更接近竞争”是一个带模型条件的极限
> 在相同需求和成本下，新增独立厂商把更多产量带进市场。单个企业的市场份额和价格影响变小，均衡价格与边际成本的差距按 $1/(n+1)$ 缩小。但这不是说现实中只要企业数增加，价格就必然按同一公式下降。
>
> <!-- bilingual-en:start -->
> Holding demand and costs fixed, an additional independent firm brings more output into the market. Each firm's share and price impact shrink, and the equilibrium markup falls at rate $1/(n+1)$. This is a conditional model result, not a universal empirical law that prices follow the same formula whenever firm count rises.
> <!-- bilingual-en:end -->

## $n$ 家企业的唯一均衡

设 $n$ 家企业同时选择产量，逆需求为

$$
P(Q)=\max\{a-bQ,0\},\qquad a>c>0,\quad b>0,
$$

且每家边际成本都为 $c$。令 $Q_{-i}=\sum_{j\ne i}q_j$。企业 $i$ 的正值最佳反应满足

$$
q_i=\frac{a-c-bQ_{-i}}{2b}.
$$

这里不能先把对称当作未经证明的假设。首先，均衡总量不可能满足 $Q\ge a/b$：只要有企业生产正产量，零价格与 $c>0$ 会使它亏损并愿意停产；而全员停产又会让任一家以小正产量获利。因此均衡位于正价格区。

任意活跃企业 $i$ 的一阶条件可写成

$$
q_i+Q=\frac{a-c}{b}.
$$

所以所有活跃企业的产量相同。若恰有 $k<n$ 家活跃，联立后它们各生产 $(a-c)/[b(k+1)]$，总量为 $k(a-c)/[b(k+1)]$。此时任一不活跃企业在 0 处的边际利润是

$$
a-c-bQ=\frac{a-c}{k+1}>0,
$$

因而会进入并生产小正产量，矛盾。故 $n$ 家企业全都活跃且对称，令 $q_i=q$ 与 $Q_{-i}=(n-1)q$ 得

$$
q^*=\frac{a-c}{b(n+1)}.
$$

由此得到

$$
Q^*=\frac{n(a-c)}{b(n+1)},
\qquad
P^*=c+\frac{a-c}{n+1},
\qquad
\pi_i^*=\frac{(a-c)^2}{b(n+1)^2}.
$$

当 $n=1$ 时，公式回到线性垄断产量；当 $n=2$ 时，回到 [[Cournot均衡|双寡头 Cournot 均衡]]。

<!-- bilingual-en:start -->
Any equilibrium must lie in the positive-price region because positive output at zero price loses money while all-zero output admits profitable entry. Every active firm's first-order condition implies $q_i+Q=(a-c)/b$, so all active firms produce equally. If only $k<n$ firms were active, their total output would leave an inactive firm's marginal profit at zero equal to $(a-c)/(k+1)>0$, contradicting inactivity. Hence all $n$ firms are active and the unique equilibrium has $q_i^*=(a-c)/[b(n+1)]$, total output $Q^*=n(a-c)/[b(n+1)]$, price $P^*=c+(a-c)/(n+1)$, and per-firm profit $(a-c)^2/[b(n+1)^2]$.
<!-- bilingual-en:end -->

## 极限告诉我们什么，也没有告诉我们什么

$$
\lim_{n\to\infty}Q^*=\frac{a-c}{b},
\qquad
\lim_{n\to\infty}P^*=c,
\qquad
\lim_{n\to\infty}\pi_i^*=0.
$$

这说明在固定线性需求、相同恒定边际成本、无进入成本且企业独立选择产量的模型里，Cournot 结果趋近完全竞争基准。它没有处理固定成本导致的自然集中、成本异质性、产品差异、容量约束、合谋或进入后需求变化。厂商数量只是模型中的一个参数；市场结构变化若同时改变其他原始条件，就要重新求解。

<!-- bilingual-en:start -->
As $n\to\infty$, total output approaches $(a-c)/b$, price approaches $c$, and each firm's profit approaches zero. The limit holds with fixed linear demand, identical constant marginal cost, no entry cost, and independent quantity choice. Cost heterogeneity, differentiation, capacity, collusion, fixed costs, or demand changes require a new model.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“$n$ 越大，价格越接近边际成本”不能直接证明拆分任何一家现实企业都会提高福利？
>
> **答案：** 公式把需求、成本技术和产品同质性固定，只改变独立决策者数量。现实拆分可能改变固定成本、规模经济、产品质量、创新或合谋能力；这些变化没有被该比较静态包含。

## 来源与核验

- MIT 14.12, [*Chapter 7: Application—Imperfect Competition*, Cournot with $n$ firms](https://ocw.mit.edu/courses/14-12-economic-applications-of-game-theory-fall-2012/a870a72380a584e8d1ffd2b34fa24c9e_MIT14_12F12_chapter7.pdf)：核对 $n$ 企业对称产量、总量、价格、利润及竞争极限。
- MIT 14.01SC, [*Exam 2 Review*, Lecture 17 summary, p. 5](https://ocw.mit.edu/courses/14-01sc-principles-of-microeconomics-fall-2011/66a11c228027ef6cd3dbc53a30e64921_MIT14_01SCF11_rev_midtm02.pdf)：交叉核对厂商数趋于无穷时 Cournot 价格趋向边际成本。
- [[01_Math/03_game theory/04_案例（囚徒困境与纳什均衡）#3.4. 拓展到 n 家寡头|本地课程：$n$ 家 Cournot]]与[[01_Math/03_game theory/第1次作业补充.pdf|补充习题 2.2]]：支持本地课程的一般 $n$ 企业设问与记号。
