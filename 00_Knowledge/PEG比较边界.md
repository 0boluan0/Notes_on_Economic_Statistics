---
aliases:
  - 较低PEG不保证低估因为风险、支付率和增长路径仍影响这个比率
  - Limits of PEG comparisons
  - PEG does not neutralize growth
student_os: knowledge-atom
atom_id: FI-VAL-025
atom_type: boundary
status: source-checked
requires:
  - "[[PEG]]"
  - "[[市盈率基本面约束]]"
related:
  - "[[股利支付率]]"
  - "[[相对估值]]"
part_of:
  - "[[股票与企业价值评估.canvas]]"
---

# 较低PEG不保证低估因为风险、支付率和增长路径仍影响这个比率
<!-- bilingual-en:start -->
A lower PEG does not guarantee undervaluation because risk, payout and the growth path still affect the ratio.
<!-- bilingual-en:end -->

把PE除以预期增长率，并没有把公司调整成“只剩价格差异”的可比对象。[[PEG]]仍受到股权成本、可分配现金以及增长持续时间的影响；因此“PEG小于1”不是独立的价值判决。
<!-- bilingual-en:start -->
Dividing P/E by expected growth does not make firms comparable except for price. [[PEG|PEG]] still depends on the cost of equity, distributable cash and growth duration; a value below one is not a standalone valuation verdict.
<!-- bilingual-en:end -->

用[[市盈率基本面约束|稳定DDM的基期PE]]做一个可检验的反例。若支付率 $d>0$、小数增长率 $0<g<k_e$，且该模型的其他稳定条件成立，则：
<!-- bilingual-en:start -->
Use the [[市盈率基本面约束|base-period P/E from stable DDM]] for a testable counterexample. With positive payout, $0<g<k_e$ and the model's other stable assumptions:
<!-- bilingual-en:end -->

$$
PEG=\frac{d(1+g)}{100g(k_e-g)}.
$$

这一步只是把基期PE再除以 $100g$。风险对应的 $k_e$ 和支付率 $d$ 仍在式中，$g$ 还出现在 $1+g$、$g$ 及 $k_e-g$ 三处；没有发生“除掉增长”的线性消去。模型内固定 $d,k_e$ 时，$g$ 趋近0或趋近 $k_e$ 都可使该比率变大。
<!-- bilingual-en:start -->
This is base P/E divided by $100g$. Both $k_e$ and payout remain, while growth appears in three factors, so it has not cancelled linearly. Within this model family, holding payout and the discount rate fixed makes the ratio rise as growth approaches zero or the discount rate.
<!-- bilingual-en:end -->

例如两家公司都按模型公允定价，$d=20\%$、$g=4\%$。股权成本12%的公司有PE=2.6、PEG=0.65；股权成本8%的公司有PE=5.2、PEG=1.30。前者PEG不到1，仍没有任何低估：它只是要求更高回报。固定股权成本和增长率而改变支付率，也会改变模型PEG。
<!-- bilingual-en:start -->
Consider two firms priced exactly at model value, both with 20% payout and 4% growth. At a 12% cost of equity, P/E is 2.6 and PEG 0.65; at 8%, they are 5.2 and 1.30. The sub-one PEG is not undervaluation: it reflects a higher required return. Changing payout while keeping growth and the discount rate fixed also changes model PEG.
<!-- bilingual-en:end -->

实际比较还要统一增长基期和预测年数：三年高增长、随后放缓，与长期保持相同增长，不是同一路径。上式只用于证明一个边界，不能用它把短期高增长率强塞进永续模型；增长所需再投资也要与支付率一致。
<!-- bilingual-en:start -->
Match growth bases and forecast horizons: three years of rapid growth followed by a slowdown is not perpetual growth at that rate. The formula establishes a boundary, not permission to put a short-run growth forecast into a perpetuity. Reinvestment must also be consistent with payout.
<!-- bilingual-en:end -->

## 来源与核验

- [Damodaran，PEG Ratios，PDF3、10–11、16](https://people.stern.nyu.edu/adamodar/pdfiles/peg.pdf#page=3)：支持低PEG可能反映风险、支付/再投资效率和非线性增长关系，而不是低估保证。这里只引用这些结构结论，不把旧市场样本的系数当当前事实。
- 本卡从稳定DDM独立推导含 $100g$ 的式子，并复算两个条件价格；非线性及极限说明均限定在明示模型内。
<!-- bilingual-en:start -->
Damodaran supports the remaining risk, payout and nonlinear-growth effects. The decimal-to-percentage formula, two model-valued examples and limiting behaviour are independently checked under the stated assumptions; historical market regressions are not used as current facts.
<!-- bilingual-en:end -->
