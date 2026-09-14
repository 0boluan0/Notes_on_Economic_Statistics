---
aliases:
  - "HHI指数是界定市场中所有经营者市场份额的平方和"
  - "Herfindahl–Hirschman index"
  - "赫芬达尔—赫希曼指数"
student_os: knowledge-atom
atom_id: FI-FUND-053
atom_type: definition
status: source-checked
requires:
  - "[[市场集中度]]"
part_of:
  - "[[宏观、行业与公司基本面分析.canvas]]"
---

# HHI指数是界定市场中所有经营者市场份额的平方和
<!-- bilingual-en:start -->
*The HHI is the sum of all operators' squared market shares in a defined market.*
<!-- bilingual-en:end -->

若 $N$ 家经营者份额 $s_i\geq0$、总和为 1，HHI 的比例口径为 $H=\sum_i s_i^2$。若改用百分数 $100s_i$ 计算，则指标是 $10{,}000H$；两种口径必须标明，不能把数值直接混用。
<!-- bilingual-en:start -->
For $N$ operators with nonnegative shares summing to one, the fractional-share HHI is $H=\sum_i s_i^2$. Squaring percentage shares $100s_i$ instead gives $10{,}000H$. State the convention rather than mixing the two scales.
<!-- bilingual-en:end -->

固定 $N\geq1$ 时，$1/N\leq H\leq1$：均等份额达到下界，一家占全部市场达到上界。例如，四家份额 30%、30%、20%、20%，则
<!-- bilingual-en:start -->
For fixed $N\geq1$, $1/N\leq H\leq1$: equal shares attain the lower bound and one operator holding the whole market attains the upper bound. For shares of 30%, 30%, 20%, and 20%,
<!-- bilingual-en:end -->

$$H=0.3^2+0.3^2+0.2^2+0.2^2=0.26,\qquad HHI_{\%}=2600.$$

下界由 $(\sum_i s_i)^2\leq N\sum_i s_i^2$ 得到。指标加重了较大份额的影响，但不直接测量加价率；任何法律审查阈值都依赖制度和时期，不能写进 HHI 的普遍定义。
<!-- bilingual-en:start -->
The lower bound follows from $(\sum_i s_i)^2\leq N\sum_i s_i^2$. Squaring places more weight on larger shares, but does not directly measure markups. Legal screening thresholds depend on jurisdiction and time and are not part of the universal definition.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [U.S. Department of Justice, Herfindahl-Hirschman Index](https://www.justice.gov/atr/herfindahl-hirschman-index)：核对平方和、百分数口径及 2600 算例；不沿用当期执法阈值作一般定理。
- [OpenStax, Principles of Microeconomics 3e, §11.1, The Herfindahl-Hirschman Index](https://openstax.org/books/principles-microeconomics-3e/pages/11-1-corporate-mergers)：核对全体市场份额进入计算。归一化与上下界独立推导核算。
<!-- bilingual-en:start -->
The DOJ page supports the squared-share formula, percentage convention, and 2600 example; its enforcement thresholds are not generalized. OpenStax supports using all shares. Normalization and bounds were independently derived and checked.
<!-- bilingual-en:end -->
