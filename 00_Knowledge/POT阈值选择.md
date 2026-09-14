---
aliases:
  - "POT阈值选择要在尾部近似偏差与超额样本不足之间权衡并检查相邻阈值的稳健性"
  - POT threshold selection
student_os: knowledge-atom
atom_id: RM-EVT-014
atom_type: decision-rule
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[超阈值极限定理]]"
  - "[[GPD阈值稳定性]]"
related:
  - "[[均值超额图]]"
  - "[[GPD拟合诊断]]"
  - "[[尾部外推不确定性]]"
---

# POT阈值选择要在尾部近似偏差与超额样本不足之间权衡并检查相邻阈值的稳健性
<!-- bilingual-en:start -->
*POT threshold selection balances tail-approximation bias against scarce excess data and checks robustness across nearby thresholds.*
<!-- bilingual-en:end -->

阈值太低，中心分布可能仍显著偏离 GPD；阈值太高，超额变少，尾参数和远端分位的估计更不稳定。[[超阈值极限定理]]没有指定一条适用于所有分布、样本量和风险目标的“95% 阈值”规则。
<!-- bilingual-en:start -->
A low threshold can retain substantial departure from a GPD tail. A very high threshold leaves fewer excesses and less stable tail-parameter and distant-quantile estimates. The [[超阈值极限定理|threshold-excess limit theorem]] does not prescribe a universal 95th-percentile threshold for every distribution, sample size, and risk target.
<!-- bilingual-en:end -->

可执行的选择路径是：先固定损失单位、期限、是否过滤以及目标风险水平，再确定候选阈值；对每个候选保留超额数 $N_u$，检查[[均值超额图]]、形状与修正尺度稳定性，以及[[GPD拟合诊断]]。在近似可接受、数据仍足够的候选区间中选择，并报告相邻阈值下目标风险值如何变化。风险值大小本身不是挑选阈值的拟合标准。
<!-- bilingual-en:start -->
Fix loss units, horizon, filtering choice, and the risk target before specifying candidate thresholds. Retain the excess count $N_u$ for each candidate and inspect the [[均值超额图|mean excess plot]], shape and modified-scale stability, and [[GPD拟合诊断|GPD fit diagnostics]]. Select within a range where approximation is credible and enough data remain, then report how the risk estimate changes at nearby thresholds. The size of the resulting risk number is not itself a goodness-of-fit criterion.
<!-- bilingual-en:end -->

跨阈值的样本相互嵌套，所以参数估计、图上误差带和检验结果相关。单点置信区间不是整条稳定曲线的同时置信带；逐个检验直到找到一个“不拒绝”的阈值，也需要考虑选择过程。一个宽误差带下看似平坦的区间，可能只表示数据太少，不能识别变化。
<!-- bilingual-en:start -->
Samples are nested across thresholds, making estimates, uncertainty bands, and test results dependent. Pointwise intervals are not simultaneous bands for an entire stability curve. Testing until one threshold is not rejected also introduces selection. A visually flat region with very wide intervals may merely indicate that the data cannot identify variation.
<!-- bilingual-en:end -->

例如 1000 个连续观测按经验分位设置候选阈值时，90%、95%、99% 附近通常只保留约 100、50、10 个超额。10 个超额不能因“更极端”就自动胜出；100 个也不能因“更多”就自动证明 GPD 近似可用。固定数值阈值时 $N_u$ 随样本变化；固定超额个数时阈值是次序统计量，二者的随机性口径应写清楚。
<!-- bilingual-en:start -->
For 1,000 continuous observations, empirical 90th-, 95th-, and 99th-percentile thresholds typically leave about 100, 50, and 10 excesses. Ten observations do not automatically win by being more extreme, nor do 100 establish a valid GPD approximation by being more numerous. With a fixed numerical threshold, $N_u$ varies across samples; with a fixed excess count, the threshold is an order statistic. State which construction is used.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [McNeil–Frey，作者版 PDF 第 7–8、10–11 页](https://statmath.wu.ac.at/~frey/publications/evt-garch.pdf#page=7)：支持阈值的偏差—方差权衡、固定超额个数的随机阈值及有限样本阈值敏感性。第 10–11 页的具体分布模拟不被推广为统一最优比例；本卡数量例只说明样本规模。
  <!-- bilingual-en:start -->
  These pages support the bias–variance trade-off, random thresholds from fixed excess counts, and finite-sample sensitivity. Their distribution-specific simulation does not supply a universal optimal fraction. The count example here illustrates sample size only.
  <!-- bilingual-en:end -->
- [Belzile，UNIL 2025，“Common challenges”“Caveats of graphical diagnostics”](https://lbelzile.github.io/UNIL-2025-choosing-threshold/UNIL-choosing_threshold.html#common-challenges)：支持嵌套数据、序贯选择、多重检验与单点误差带边界；已重开相应正文。
  <!-- bilingual-en:start -->
  The reopened sections support the limitations from nested data, sequential selection, multiple testing, and pointwise intervals.
  <!-- bilingual-en:end -->
