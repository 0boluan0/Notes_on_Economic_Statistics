---
aliases:
  - POT尾概率由进入阈值的概率乘以条件超额的生存概率得到
student_os: knowledge-atom
atom_id: RM-EVT-021
atom_type: method
status: source-checked
requires:
  - "[[超阈值法]]"
  - "[[广义Pareto分布]]"
leads_to:
  - "[[POT分位数]]"
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
---

# POT尾概率由进入阈值的概率乘以条件超额的生存概率得到
<!-- bilingual-en:start -->
*A POT tail probability multiplies the threshold-exceedance probability by conditional excess survival*
<!-- bilingual-en:end -->

[[超阈值法]] 拟合的是“已经超过 $u$ 之后，还会超出多少”。要问整个损失分布中 $L>x$ 的概率，必须再乘上进入这段尾部的概率 $p_u=P(L>u)>0$；只用 GPD 生存函数会漏掉这个比例。
<!-- bilingual-en:start -->
[[超阈值法|Peaks over threshold]] models how far loss exceeds $u$, conditional on already crossing it. To obtain $P(L>x)$ in the full loss distribution, multiply by the probability $p_u=P(L>u)>0$ of entering that tail. GPD survival alone omits this factor.
<!-- bilingual-en:end -->

设 $F_u(y)=P(L-u\le y\mid L>u)$。当 $x\ge u$ 时，事件 $\{L>x\}$ 包含在 $\{L>u\}$ 内，所以条件概率恒等式给出
<!-- bilingual-en:start -->
Let $F_u(y)=P(L-u\le y\mid L>u)$. For $x\ge u$, the event $\{L>x\}$ is contained in $\{L>u\}$. Conditional probability therefore gives:
<!-- bilingual-en:end -->

$$P(L>x)=p_u\,[1-F_u(x-u)].$$

若在选定阈值以上采用 [[广义Pareto分布]] 近似，并以 $n$ 个同口径观测中的 $N_u$ 个超阈值观测估计 $p_u$，要求 $0<N_u\le n$，令 $\widehat p_u=N_u/n$，则
<!-- bilingual-en:start -->
Approximate the excess distribution above the chosen threshold by a [[广义Pareto分布|GPD]]. If $N_u$ of $n$ observations under one loss convention exceed $u$, require $0<N_u\le n$, use $\widehat p_u=N_u/n$ and obtain:
<!-- bilingual-en:end -->

$$
\widehat P(L>x)=
\begin{cases}
\widehat p_u\left(1+\widehat\xi\dfrac{x-u}{\widehat\beta}\right)^{-1/\widehat\xi},&\widehat\xi\ne0,\\[4pt]
\widehat p_u\exp\!\left(-\dfrac{x-u}{\widehat\beta}\right),&\widehat\xi=0.
\end{cases}
$$

要求 $\widehat\beta>0$、$x\ge u$；幂式在 $1+\widehat\xi(x-u)/\widehat\beta>0$ 的支撑内部使用。若 $\widehat\xi<0$，模型损失上端点是 $u-\widehat\beta/\widehat\xi$，达到或超过端点的严格超越概率为零，不应把负底数代入幂式。阈值以下的分布没有被这套尾部公式估计。
<!-- bilingual-en:start -->
Require $\widehat\beta>0$ and $x\ge u$, with the power expression evaluated inside its support, where $1+\widehat\xi(x-u)/\widehat\beta>0$. For $\widehat\xi<0$, the fitted loss endpoint is $u-\widehat\beta/\widehat\xi$; strict exceedance probability is zero at or above it. Do not raise a negative base to a fractional power. This tail formula does not estimate the distribution below $u$.
<!-- bilingual-en:end -->

## 一个比例不能漏
<!-- bilingual-en:start -->
*The missing-factor check*
<!-- bilingual-en:end -->

假设 1000 个日损失中有 50 个超过 100 万元，并给定拟合参数 $\widehat\xi=0.2,\widehat\beta=10$ 万元。损失超过 150 万元，需要先进入 5% 的尾部，再在其中超出至少 50 万元：
<!-- bilingual-en:start -->
Suppose 50 of 1,000 daily losses exceed CNY 1 million, with fitted parameters $\widehat\xi=0.2$ and $\widehat\beta=10$ in units of CNY 10,000. Crossing 150 in those units first requires entering the 5% tail, then exceeding the threshold by another 50:
<!-- bilingual-en:end -->

$$
\widehat P(L>150\mid L>100)=\left(1+0.2\frac{50}{10}\right)^{-5}=\frac1{32},
\qquad
\widehat P(L>150)=0.05\times\frac1{32}=0.0015625=0.15625\%.
$$

条件概率为 3.125%，整个分布里的概率为 0.15625%，二者回答的不是同一个问题。这是给定模型的计算，不是用三个参数就证明未来频率已经校准。日损失、月损失或标准化残差也不能在同一个 $N_u/n$ 中混用。
<!-- bilingual-en:start -->
The conditional probability is 3.125%, but the full-distribution probability is 0.15625%. They answer different questions. This is a calculation under the fitted model, not proof of calibrated future frequencies. Daily losses, monthly losses and standardized residuals cannot be mixed in the same $N_u/n$.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，*Extreme Value Theory*，PDF 第 20、27、29 页](https://www.columbia.edu/~mh2078/QRM/EVT_MasterSlides.pdf#page=27)：条件超额、尾概率乘法与经验超阈值比例；已重开并目视第 27 页公式。端点处理按 GPD 支撑核对。
- [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论|第 13 章课程记录]]：提供 $n=1000,N_u=50,u=100,\xi=0.2,\beta=10$ 的题设；本页另取 $x=150$，独立复算条件与无条件概率，未声称重新拟合过原始数据。
<!-- bilingual-en:start -->
- Haugh, PDF pp. 20, 27 and 29, supports conditional excesses, probability multiplication and the empirical exceedance proportion; p. 27 was reopened and visually checked. Endpoint handling was checked against GPD support.
- The [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论|Chapter 13 record]] supplies the exercise parameters. This page independently evaluates $x=150$ and distinguishes the two probabilities; it does not claim a refit of the original observations.
<!-- bilingual-en:end -->
