---
aliases:
  - PEG是市盈率除以按百分数值表示的预期每股收益增长率
  - Price-earnings-to-growth ratio
  - P/E-to-growth ratio
student_os: knowledge-atom
atom_id: FI-VAL-024
atom_type: definition
status: source-checked
requires:
  - "[[市盈率]]"
related:
  - "[[PEG比较边界]]"
part_of:
  - "[[股票与企业价值评估.canvas]]"
---

# PEG是市盈率除以按百分数值表示的预期每股收益增长率
<!-- bilingual-en:start -->
PEG divides P/E by expected EPS growth expressed as a percentage number.
<!-- bilingual-en:end -->

PEG把一个明确盈利口径的[[市盈率]]，与同一基期出发、指定未来年数的预期EPS年复合增长率并列成一个比率。若增长率用小数 $g$ 表示，则百分数值 $G=100g$，常用定义为：
<!-- bilingual-en:start -->
PEG combines a specified [[市盈率|P/E]] with expected annual compound EPS growth over a stated horizon from a matched earnings base. With decimal growth $g$, the percentage number is $G=100g$:
<!-- bilingual-en:end -->

$$
PEG=\frac{PE}{G}=\frac{PE}{100g}.
$$

例如价格40元、基期EPS为2元，PE为20；若预计三年后EPS为2.662元，则 $g=(2.662/2)^{1/3}-1=10\%$，$G=10$，PEG为 $20/10=2$。不能把10%按0.10直接放进这一定义的分母而得到200，也不能把 $100g$ 写成 $100\%\times g$。
<!-- bilingual-en:start -->
Price 40 and base EPS 2 give P/E of 20. Forecast EPS of 2.662 three years later implies 10% compound growth, so $G=10$ and PEG is 2. Dividing by decimal 0.10 would instead give 200; multiplying $g$ by 100% does not perform the required unit conversion.
<!-- bilingual-en:end -->

记录PEG时要同时写出PE采用的盈利年、预测增长的起止年和预测来源。若改用下一年EPS计算forward PE，增长基期也要相应核对，不能不说明便把同一段增长重复计入。通常比较使用正盈利和正增长；零增长使此式无定义，负值不适用普通的低PEG排序。其比较含义另见[[PEG比较边界]]。
<!-- bilingual-en:start -->
Record the earnings year, forecast start and end years, and forecast source. Changing to forward EPS also requires checking the growth base rather than silently counting the same growth twice. Ordinary comparisons use positive earnings and growth; zero growth is undefined and negative values do not support ordinary low-PEG ranking. See [[PEG比较边界|the comparison boundary]].
<!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/06_证券投资学/证券投资学.pdf#page=644|证券投资学，PDF644]]：原公式为PE除以复合增长率乘100，支持百分数值口径；正文三年增长例独立计算。
- [Damodaran，PEG Ratios，PDF6、8](https://people.stern.nyu.edu/adamodar/pdfiles/peg.pdf#page=6)：支持盈利基期、预测期限、来源一致性；表内PE20.65与增长19.50%对应PEG约1.06，验证百分数值单位。
<!-- bilingual-en:start -->
The course formula and Damodaran's numerical table confirm the percentage-number convention; his definition checks support matched bases and forecast horizons. The three-year example is independently calculated.
<!-- bilingual-en:end -->
