---
aliases:
  - DHS私人信息模型是高估自身信号精度使价格过度响应私人信息的定价模型
  - DHS private-information model
student_os: knowledge-atom
atom_id: FI-BF-044
atom_type: definition
status: source-checked
requires:
  - "[[过度自信]]"
leads_to:
  - "[[DHS信心反馈]]"
part_of:
  - "[[行为金融与套利限制.canvas]]"
---

# DHS私人信息模型是高估自身信号精度使价格过度响应私人信息的定价模型
<!-- bilingual-en:start -->
*The DHS private-information model links excessive perceived signal precision to price overreaction*
<!-- bilingual-en:end -->

Daniel–Hirshleifer–Subrahmanyam 的固定信心模型把[[过度自信]]具体化为：知情投资者低估自己私人信号的误差方差，却没有同样高估公共信号的精度。私人消息因而在其估值中权重过大，后续公共信息逐渐纠偏。
<!-- bilingual-en:start -->
In the fixed-confidence DHS model, informed investors understate private-signal noise without similarly overstating public-signal precision. Private information receives excessive valuation weight; later public information gradually corrects it.
<!-- bilingual-en:end -->

用一个自拟正态信号算例看清“过重”：先验基本价值 $\theta\sim N(0,1)$，信号 $s=\theta+\varepsilon$，真实噪声 $\varepsilon\sim N(0,1)$ 且与 $\theta$ 独立。看到 $s=2$，正确后验均值是 $2/(1+1)=1$；若主观模型仍取独立正态噪声，却误设为 $N(0,1/4)$，自己的估值就成了 $2/(1+1/4)=1.6$。差异来自估计精度，不是消息本身更好。
<!-- bilingual-en:start -->
Original normal-signal illustration: $\theta\sim N(0,1)$ and independent $\varepsilon\sim N(0,1)$ give posterior mean 1 when $s=\theta+\varepsilon=2$. A subjective model with independent $N(0,1/4)$ noise raises the assessed mean to 1.6. Precision, not better news, creates the difference.
<!-- bilingual-en:end -->

这个后验算例只展示个人估值；把它变成价格还需要模型的市场设定。进一步让信心随结果变化，才进入[[DHS信心反馈]]，不能从一个固定的过度自信参数直接跳到所有动量结论。
<!-- bilingual-en:start -->
The illustration is an individual valuation, not a standalone market-equilibrium proof. Outcome-dependent confidence is the additional mechanism in [[DHS信心反馈|DHS confidence feedback]].
<!-- bilingual-en:end -->

## 来源与核验

- [Daniel, Hirshleifer & Subrahmanyam (1998), *Investor Psychology and Security Market Under- and Overreactions*，§II](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00077)：核对私人信号精度偏差及固定信心模型。数值为本卡自拟的正态后验演算，不是论文实证估计。

<!-- bilingual-en:start -->
Section II supports the fixed-confidence mechanism; the numbers are an independently computed posterior illustration.
<!-- bilingual-en:end -->
