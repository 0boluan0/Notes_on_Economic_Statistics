---
aliases:
  - BSV信念模型是投资者在错误的均值回归与趋势延续模型内更新盈利预期的定价模型
  - BSV model of investor sentiment
student_os: knowledge-atom
atom_id: FI-BF-043
atom_type: definition
status: source-checked
requires:
  - "[[代表性启发式]]"
related:
  - "[[DHS私人信息模型]]"
  - "[[HS信息扩散模型]]"
  - "[[套利限制]]"
part_of:
  - "[[行为金融与套利限制.canvas]]"
---

# BSV信念模型是投资者在错误的均值回归与趋势延续模型内更新盈利预期的定价模型
<!-- bilingual-en:start -->
*The BSV model prices an asset through earnings forecasts updated within a misspecified reversion–trend model*
<!-- bilingual-en:end -->

Barberis–Shleifer–Vishny 模型中，真实盈利服从随机游走；代表性投资者却相信盈利变化在“容易反向”和“容易同向延续”两个状态间切换。他观察新盈利后，用贝叶斯规则调整当前处于哪种状态的概率，但不放弃这套错误模型。
<!-- bilingual-en:start -->
True earnings follow a random walk. The representative investor instead believes changes switch between reversal-prone and continuation-prone regimes, updating regime probabilities by Bayes' rule without abandoning the misspecified model.
<!-- bilingual-en:end -->

因此，错误不只是“不会更新概率”。如果他主要相信变化会反转，一次盈利好消息便被当作暂时成分，价格调整不足；连续同向消息又可能让他过分相信趋势将持续，推高长期预测。模型在一定参数范围内同时产生反应不足与过度反应，心理动机包括保守更新与[[代表性启发式]]。
<!-- bilingual-en:start -->
The error is not simply failure to update. Reversal beliefs discount an isolated earnings surprise; repeated same-sign news can produce excessive extrapolation. Under suitable parameters, both underreaction and overreaction arise, motivated by conservatism and [[代表性启发式|representativeness]].
<!-- bilingual-en:end -->

最小识别例：同样是连续三次盈利上升，正确的随机游走基准不因此认定下一次增量更可能为正；对 BSV 投资者，同号消息则相对于本期更新前的预测先验增加“趋势状态”的权重。这不声称跨期后验必定逐次上升，状态转移还会改变下一期先验。仅凭三次上涨，不能确认现实数据真的服从 BSV；还要检验盈利过程与预期变化。原模型也没有内生推导专业资金为何无法纠偏，那个环节另见[[套利限制]]。
<!-- bilingual-en:start -->
Three increases do not make the next increment more likely positive under the specified random-walk benchmark. Same-sign news raises the trend weight relative to that period's predictive prior; regime transitions can change the next prior, so posterior weights need not rise monotonically across dates. The sequence alone cannot identify BSV. The model leaves corrective-trading limits to [[套利限制|separate work]].
<!-- bilingual-en:end -->

## 来源与核验

- [Barberis, Shleifer & Vishny (1998), *A Model of Investor Sentiment*，pp. 309–310、318–323](https://shleifer.scholars.harvard.edu/sites/g/files/omnuum10626/files/shleifer/files/model_invest_sent.pdf)：核对真实随机游走、错误的双状态模型、模型内贝叶斯更新与参数边界；p. 309 明确把套利失败留给其他研究。

<!-- bilingual-en:start -->
The original specification distinguishes a wrong earnings model from Bayesian updating within it, and explicitly treats limits to arbitrage outside this model.
<!-- bilingual-en:end -->
