---
aliases:
  - BSM看涨Delta与风险中性到期实值概率来自不同的期望权重
  - Delta and the probability of finishing in the money
student_os: knowledge-atom
atom_id: FI-MKT-007
atom_type: distinction
status: source-checked
requires:
  - "[[Delta]]"
  - "[[BSM期权定价]]"
related:
  - "[[风险模拟P-Q分工]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# BSM看涨Delta与风险中性到期实值概率来自不同的期望权重
<!-- bilingual-en:start -->
*BSM call Delta and the risk-neutral probability of finishing in the money use different expectation weights*
<!-- bilingual-en:end -->

[[Delta]] 回答“现货小变动使期权价值改变多少”；到期实值概率回答“到期时 $S_T>K$ 的可能性多大”。在 [[BSM期权定价|BSM 欧式看涨模型]] 中，它们分别是

$$\Delta_C=e^{-q\tau}N(d_1),\qquad Q(S_T>K\mid\mathcal F_t)=N(d_2),\qquad d_1=d_2+\sigma\sqrt\tau.$$

<!-- bilingual-en:start -->
[[Delta|Delta]] measures local price sensitivity; the probability of finishing in the money measures the event $S_T>K$. In the [[BSM期权定价|BSM European call model]], these are $e^{-q\tau}N(d_1)$ and $N(d_2)$ respectively.
<!-- bilingual-en:end -->

从风险中性价格 $C=e^{-r\tau}E^Q[(S_T-K)^+]$ 求现货导数时，还要乘 $\partial S_T/\partial S=S_T/S$。所以 Delta 含有**股价权重**，不只是对行权事件的指示函数求平均；权重造成了 $d_1$ 与 $d_2$ 的差异。
<!-- bilingual-en:start -->
Differentiating the risk-neutral call value with respect to spot introduces $\partial S_T/\partial S=S_T/S$. Delta therefore includes a stock-price weight, rather than just averaging an exercise-event indicator. This explains the difference between $d_1$ and $d_2$.
<!-- bilingual-en:end -->

例如 $S=K=100,r=0.05,q=0,\sigma=0.20,\tau=1$ 时，Delta 约为 $N(0.35)=0.636831$，风险中性实值概率约为 $N(0.15)=0.559618$。两者都不能直接当成真实世界的上涨或盈利概率：真实事件频率还需要 $P$，而交易盈利又取决于买入成本、融资和路径。
<!-- bilingual-en:start -->
For $S=K=100,r=0.05,q=0,\sigma=0.20,\tau=1$, Delta is about 0.636831 and the risk-neutral ITM probability about 0.559618. Neither is directly the real-world probability of a rise or trading profit: actual frequencies use $P$, and profit also depends on cost, financing and path.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Kohn–Allen, [*Derivative Securities*, §4](https://math.nyu.edu/~kohn/derivative.securities/2007/section4.pdf), pp. 6–9：风险中性分布与截断支付期望；[§5](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf), pp. 1–3、5：看涨 Delta。
- 概率与导数数值按同一 BSM 参数分别复算；不采用“Delta 就是实值概率”的无条件说法。
<!-- bilingual-en:start -->
*Sources and verification*

- Kohn–Allen, [*Derivative Securities*, Section 4](https://math.nyu.edu/~kohn/derivative.securities/2007/section4.pdf), pp. 6–9, supports the risk-neutral distribution and truncated-payoff expectations. [Section 5](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf), pp. 1–3 and 5, supports call Delta.
- The probability and derivative were calculated separately using the same BSM parameters. Delta is not unconditionally identified with the probability of finishing in the money.
<!-- bilingual-en:end -->
