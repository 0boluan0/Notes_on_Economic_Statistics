---
aliases:
  - "Theta是其他输入固定时价值对日历时间的一阶偏导数"
student_os: knowledge-atom
atom_id: FI-GREEK-005
atom_type: definition
status: source-checked
requires:
  - "[[Greeks]]"
related:
  - "[[Greek报价换算]]"
  - "[[Greeks损益归因]]"
leads_to:
  - "[[Gamma与Theta的对冲损益]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Theta是其他输入固定时价值对日历时间的一阶偏导数
<!-- bilingual-en:start -->
*Theta is the first partial derivative of value with respect to calendar time while the other inputs are held fixed*
<!-- bilingual-en:end -->

Theta 是在合约到期日 $T$ 固定、其他估值输入保持不变时，价值 $V$ 随日历时间 $t$ 推进的局部变化率。若改用剩余期限 $\tau=T-t$ 表达同一价值函数，时间推进意味着 $\tau$ 减少，因此
<!-- bilingual-en:start -->
Theta is the local rate at which value $V$ changes as calendar time $t$ advances, with the contractual expiry date $T$ and other valuation inputs fixed. Rewriting the same value function in terms of remaining maturity $\tau=T-t$ reverses the direction of time:
<!-- bilingual-en:end -->

$$\Theta=\frac{\partial V}{\partial t}=-\frac{\partial V}{\partial\tau}.$$

若 $t$ 用年计，Theta 的单位是价值／年。一天的局部价格变化用 $\Theta\,\delta t$ 估计，其中 $\delta t$ 必须按模型的日计数约定换成年；若明确采用每年 $365$ 个日历日，则每日尺度为 $\Theta/365$。不能把每年导数直接乘“1 天”，也不能把某个系统的 $252$ 或 $365$ 当作通用约定。
<!-- bilingual-en:start -->
When $t$ is measured in years, theta has units of value per year. Estimate one day's local price change as $\Theta\,\delta t$, converting the day into years under the model's day-count convention. An explicitly chosen 365-calendar-day year gives a daily scale of $\Theta/365$. An annual derivative cannot be multiplied directly by “one day,” and neither 252 nor 365 is a universal system convention.
<!-- bilingual-en:end -->

例如数学示例 $V(\tau)=10\sqrt{\tau}$ 在 $\tau=1$ 时有 $V_\tau=5$、Theta 为 $-5$。按 $365$ 日口径，一天的一阶变化约为 $-5/365=-0.01370$。若把对剩余期限的正导数 $5$ 叫作同一 Theta，就会把时间方向反过来。
<!-- bilingual-en:start -->
For the mathematical example $V(\tau)=10\sqrt{\tau}$ at $\tau=1$, the remaining-maturity derivative is $5$ while theta is $-5$. Under a 365-day convention, the first-order daily change is approximately $-5/365=-0.01370$. Calling the positive remaining-maturity derivative the same theta would reverse the time direction.
<!-- bilingual-en:end -->

“时间损耗”是常用名称，不是 Theta 必须为负的定义。普通期权的符号仍取决于具体合约和参数；例如 NYU 的欧式看跌公式允许正或负 Theta。Theta 也不等于真实一天的总损益：市场输入变化、期间现金流与融资等必须按 [[Greeks损益归因|损益归因口径]] 一起处理。
<!-- bilingual-en:start -->
“Time decay” is a common name, not a requirement that theta be negative. Even ordinary-option signs depend on the contract and parameters; the NYU European-put formula permits either sign. Theta is not total realized daily P&L: market-input changes, interim cash flows, and financing must be treated under the chosen [[Greeks损益归因|P&L attribution convention]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Courant，Kohn 与 Allen，Section 5，第 5、8–9 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开第 4–9 页、目视第 5、8 页；核对对剩余期限减少求导的符号、看跌 Theta 的双向符号和第 9 页的每日换算说明。
- [OIC，Theta](https://www.optionseducation.org/advancedconcepts/theta)：已重开全文，核对其他输入固定的日尺度及不同模型没有统一时间衰减约定；本卡采用带符号导数，不把正的“损耗金额”与之混用。
<!-- bilingual-en:start -->
- [NYU Courant, Kohn and Allen, Section 5, pp. 5 and 8–9](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), reopened with pp. 5 and 8 visually checked, supports differentiation with respect to decreasing maturity, either sign for put theta, and daily scaling.
- [OIC, Theta](https://www.optionseducation.org/advancedconcepts/theta), reopened in full, supports daily reporting with other inputs fixed and the lack of an industry-wide decay convention. This card consistently uses a signed derivative rather than an unsigned decay amount.
<!-- bilingual-en:end -->
