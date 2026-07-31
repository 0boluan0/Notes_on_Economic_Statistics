**SmartEcon – “巴塞尔资本知识点一张网”**
<!-- bilingual-en:start -->
**SmartEcon — “The Basel Capital Framework on One Page”**
<!-- bilingual-en:end -->

## **⬇️ 第一层：监管演进时间线（**
<!-- bilingual-en:start -->
*⬇️ Layer 1: Timeline of Regulatory Evolution*
<!-- bilingual-en:end -->

## **知道“谁管什么”**
<!-- bilingual-en:start -->
*Know which framework governs which risk*
<!-- bilingual-en:end -->

## **）**

```
Basel I(1988)        只看信用风险 + 8 % 规则
   └─1996 修正案     加入交易簿市场风险（VaR × mc 或标准法）
Basel II(2004)       三支柱：最低资本 + 监管审查 + 市场披露
   ├─信用风险   SA / IRB
   ├─市场风险   承接 1996
   └─操作风险   BIA / SA / AMA
Basel III            引入杠杆率、资本缓冲、LCR/NSFR
```

---

## **⬇️ 第二层：两本账的“分水岭”**
<!-- bilingual-en:start -->
*⬇️ Layer 2: The Boundary Between the Two Books*
<!-- bilingual-en:end -->

|**维度**|**银行簿 (Banking Book)**|**交易簿 (Trading Book)**|
|---|---|---|
|定位|长期持有赚息差|短期买卖赚价差/流动性|
|会计|摊余成本 or FVOCI|Fair Value through P&L|
|**风险类别**|_信用风险为主_|_市场风险为主_ (+ [[Specific Variance|Specific]])|
|资本公式|权重 × 余额 × 8 %|VaR×mc + SRC _or_ 标准法表格|
|常见业务|贷款、HTM债|债券做市、利率互换、期权|
<!-- bilingual-en:start -->
| **Dimension** | **Banking book** | **Trading book** |
| --- | --- | --- |
| Purpose | Hold positions over longer horizons and earn interest margins | Trade, make markets, hedge, or provide liquidity over shorter horizons |
| Accounting | Often amortised cost or FVOCI, depending on classification | Generally fair value through profit or loss |
| **Dominant risk** | *Credit risk* | *Market risk*, plus instrument-specific risk |
| Historical capital approach | Exposure × risk weight × 8% | VaR × multiplier plus a specific-risk charge, or the standardised tables |
| Typical business | Loans and held-to-maturity debt | Bond market-making, interest-rate swaps, and options |
<!-- bilingual-en:end -->

> **口诀：**“长留进银行簿，短炒进交易簿；信用吃权重，市场看 VaR。”
> <!-- bilingual-en:start -->
> **Mnemonic:** “Long-term holdings go to the banking book; positions held for trading go to the trading book. Credit capital follows risk weights, while the historical market-risk framework used VaR.”
> <!-- bilingual-en:end -->

---

## **⬇️ 第三层：三大风险的** 
<!-- bilingual-en:start -->
*⬇️ Layer 3: The Three Main Risk Classes*
<!-- bilingual-en:end -->

## **核心公式 & 查表口径**
<!-- bilingual-en:start -->
*Core formulas and where their parameters come from*
<!-- bilingual-en:end -->

|**风险块**|**你要准备的输入**|**一行公式**|**表格/参数从哪查**|
|---|---|---|---|
|**[[Credit Risk|信用风险]]**|余额、评级 or [[PD|PD]]/[[LGD|LGD]]|[[Risk-Weighted Assets|RWA]] = 余额 × 权重 (或 12.5 K [[EAD|EAD]])|[[Basel Accords|Basel]] SA 权重表 or IRB ρ-[[PD|PD]] 函数|
|**[[Market Risk|市场风险]]**|交易簿 P&L 历史|**VaR10d,99% × mc** (+SRC)|回溯测试 mc 表；SRC 权数表|
|**[[Operational Risk|操作风险]]**|毛收入 or 损失数据库|BIA: -|15 %×Gross|
<!-- bilingual-en:start -->
| **Risk block** | **Inputs** | **Condensed historical formula** | **Where to obtain parameters** |
| --- | --- | --- | --- |
| **[[Credit Risk|Credit risk]]** | Exposure, rating, or [[PD|PD]]/[[LGD|LGD]] | [[Risk-Weighted Assets|RWA]] = exposure × risk weight, or $12.5K\times$ [[EAD|EAD]] under IRB | [[Basel Accords|Basel]] standardised risk-weight tables or the IRB correlation and [[PD|PD]] functions |
| **[[Market Risk|Market risk]]** | Trading-book P&L history | Historical IMA: ten-day 99% VaR × multiplier, plus SRC | Backtesting multiplier and specific-risk tables |
| **[[Operational Risk|Operational risk]]** | Gross income or internal loss data | Historical BIA: 15% × gross income | The applicable Basel operational-risk approach |
<!-- bilingual-en:end -->

---

## **⬇️ 第四层：常考“流程图”**
<!-- bilingual-en:start -->
*⬇️ Layer 4: Commonly Examined Procedures*
<!-- bilingual-en:end -->

### **1. 衍生品（无净额）——CEM**
<!-- bilingual-en:start -->
*1. Derivatives without a netting agreement — CEM*
<!-- bilingual-en:end -->

```
现期敞口  = max(V,0)
Add-on    = Notional × α(按类别+期限查表)
EAD       = 现期敞口 + Add-on
RWA       = EAD × 对手权重
资本金    = RWA × 8 %
```

> α 速记：利率 0 / 0.5 / 1.5；商品 10 / 12 / 15（≤1y / 1-5y / >5y）
> <!-- bilingual-en:start -->
> Add-on mnemonic under the historical table: interest rates 0 / 0.5 / 1.5; general commodities 10 / 12 / 15 for maturities ≤1 year / 1–5 years / >5 years.
> <!-- bilingual-en:end -->

### **2. 有净额——再加两步**
<!-- bilingual-en:start -->
*2. With recognised netting — add two further steps*
<!-- bilingual-en:end -->

```
NRR = Σ正值 / Σ绝对值
θ   = 0.4 + 0.6×NRR
Add-on 改为 θ×ΣAdd-on
```

### **3. 交易簿 VaR 路**
<!-- bilingual-en:start -->
*3. The Historical Trading-Book VaR Route*
<!-- bilingual-en:end -->

```
1-day VaR(99%)
↓ ×√10
10-day VaR
↓ 回溯测试 → mc
VaR × mc
↓ + SRC
市场风险资本
```

---

## **🔑 记忆小抄（只背这 6 条）**
<!-- bilingual-en:start -->
*🔑 Six-point memory sheet*
<!-- bilingual-en:end -->

1. **总资本 = 8 % × [[Risk-Weighted Assets|RWA]]**（核心）
2. **银行簿 [[Risk-Weighted Assets|RWA]] = 余额 × 权重**
3. **交易簿 VaR 用 10 d × mc**
4. **mc 表：红叉 0-4→3.0，5-9→3.4，10-14→3.5，15+→4.0**
5. **NRR 折扣：θ = 0.4 + 0.6 η**
6. **利率 Add-on 0 | 0.5 | 1.5 %**
<!-- bilingual-en:start -->
1. **Historical Basel I/II minimum total capital = 8% × [[Risk-Weighted Assets|RWA]].**
2. **Banking-book [[Risk-Weighted Assets|RWA]] = exposure × risk weight.**
3. **The 1996 internal-model approach used ten-day VaR and a regulatory multiplier.**
4. **Backtesting multiplier:** 0–4 exceptions → 3.00; 5 → 3.40; 6 → 3.50; 7 → 3.65; 8 → 3.75; 9 → 3.85; 10 or more → 4.00.
5. **Recognised netting reduced the add-on through $0.4+0.6\times\text{NGR}$.**
6. **Historical interest-rate add-ons:** 0 / 0.5 / 1.5%.
<!-- bilingual-en:end -->



# 1. 对银行资本进行监管的原因
<!-- bilingual-en:start -->
*1. Why Bank Capital Is Regulated*
<!-- bilingual-en:end -->

降低单体与连锁破产风险。~~防破产,防止一连串破产~~
<!-- bilingual-en:start -->
The purpose is to reduce the probability that one institution fails and that its failure propagates through the system. ~~Capital cannot make failure impossible, but it can make both individual and cascading failures less likely.~~
<!-- bilingual-en:end -->

1. **降低破产概率，稳住信心** —— 完全杜绝银行破产并不现实，但足额资本能把破产概率压到极低，从而维持公众与企业对金融体系的信任 。
<!-- bilingual-en:start -->
1. **Reduce the probability of failure and preserve confidence.** Eliminating bank failure altogether is unrealistic, but sufficient loss-absorbing capital can make it much less likely and support public and corporate confidence in the financial system.
<!-- bilingual-en:end -->
    
2. **抑制存款保险诱发的道德风险** —— 有了存款保险，银行容易“拿别人的钱去冒更大的险”；强制资本缓冲把“肆意加杠杆”变得代价高昂 。
<!-- bilingual-en:start -->
2. **Limit moral hazard created by deposit insurance.** When deposits are insured, banks may be tempted to take greater risks with protected funding. Mandatory capital makes aggressive leverage more costly to bank owners.
<!-- bilingual-en:end -->
    
3. **防范系统性风险** —— 一家巨型银行倒下可能连锁拖垮同业；监管部门关注整个体系的“火灾隔断”效果，而资本要求正是防火墙 。
<!-- bilingual-en:start -->
3. **Limit systemic risk.** The failure of a large, interconnected bank can transmit losses and liquidity stress to other institutions. Capital requirements provide one layer of protection against that contagion.
<!-- bilingual-en:end -->

# 2. basel I

> 目标：学会 **① 资本分层（Tier 1 / Tier 2）**、**② [[Risk-Weighted Assets|风险加权资产]] [[Risk-Weighted Assets|RWA]] 的四档权重**、**③ 计算 Cooke Ratio 并判断合规性**。
> <!-- bilingual-en:start -->
> Objective: understand **(1) the historical Tier 1/Tier 2 capital structure**, **(2) the four broad [[Risk-Weighted Assets|risk-weight]] categories used in Basel I**, and **(3) how to calculate the Cooke capital ratio and assess compliance**.
> <!-- bilingual-en:end -->

| **模块**           | **核心内容**                                      | **监管意图**        |
| ---------------- | --------------------------------------------- | --------------- |
| **资本定义**         | Tier 1（核心资本：股本 + 留存收益）；Tier 2（附属资本：次级债、一般准备等） | 确保最能吸收损失的是真金白银  |
| **[[Risk-Weighted Assets|风险加权资产]] ([[Risk-Weighted Assets|RWA]])** | 把资产按 0 % / 20 % / 50 % / 100 % 四档系数加权         | 让“多赚多压本”，降低监管套利 |
| **Cooke Ratio**  | [[Basel Capital Adequacy Ratio|资本充足率]] = 总资本 ÷ [[Risk-Weighted Assets|RWA]] ≥ **8 %**，且 Tier 1 ≥ 4 %    | 全球统一尺子，限制过度杠杆   |
<!-- bilingual-en:start -->
| **Module** | **Core content** | **Regulatory purpose** |
| --- | --- | --- |
| **Capital definition** | Tier 1: equity and disclosed reserves; Tier 2: eligible supplementary items such as subordinated debt and general provisions | Ensure that recognised capital can absorb losses |
| **[[Risk-Weighted Assets|Risk-weighted assets]] ([[Risk-Weighted Assets|RWA]])** | Apply historical weights of 0%, 20%, 50%, or 100% to broad asset classes | Relate capital to measured credit exposure, albeit coarsely |
| **Cooke ratio** | [[Basel Capital Adequacy Ratio|Total capital ratio]] = total eligible capital ÷ [[Risk-Weighted Assets|RWA]] ≥ **8%**, with Tier 1 historically at least 4% | Establish a common international minimum and constrain leverage |
<!-- bilingual-en:end -->

|**资产类别**|**示例**|**权重 (wᵢ)**|
|---|---|---|
|**0 %**|OECD 主权债、本国政府债|0.00|
|**20 %**|OECD 银行同业存放、政府机构债|0.20|
|**50 %**|住宅抵押贷款|0.50|
|**100 %**|企业贷款、股权、非常驻主权债|1.00|
<!-- bilingual-en:start -->
| **Historical weight** | **Illustrative Basel I category** | **Coefficient** |
| --- | --- | ---: |
| **0%** | Certain OECD and domestic sovereign claims | 0.00 |
| **20%** | Certain OECD bank and public-sector claims | 0.20 |
| **50%** | Qualifying residential mortgages | 0.50 |
| **100%** | Corporate lending, equity, and other claims | 1.00 |
<!-- bilingual-en:end -->

_表外项目_ 先乘 **[[Credit Conversion Factor|信用转换系数（CCF）]]** 再乘风险权重，例如：
<!-- bilingual-en:start -->
For an *off-balance-sheet item*, first apply the **[[Credit Conversion Factor|credit conversion factor (CCF)]]** and then the counterparty risk weight. The note gives these examples:
<!-- bilingual-en:end -->

- 授信承诺 ≤ 1 年：CCF = 20 %
- OTC 利率互换：CCF = 0.0 %（早期免资本）
<!-- bilingual-en:start -->
- Credit commitment with maturity of no more than one year: CCF = 20% under the convention used in this note.
- Interest-rate swap with maturity of no more than one year: the historical potential-future-exposure add-on is 0.0%, although positive current exposure is still counted.
<!-- bilingual-en:end -->

[[Cooke Ratio|库克比率]]
$$\text{[[Cooke Ratio|库克比率]]}=\frac{\text{资本（一级资本+二级资本）}}{\text{[[Risk-Weighted Assets|风险加权资产]]（[[Risk-Weighted Assets|RWA]]）}}$$
<!-- bilingual-en:start -->
[[Cooke Ratio|Cooke ratio]]
<!-- bilingual-en:end -->
# 3.G30

|**年份**|**事件**|**缺口暴露**|
|---|---|---|
|**1987**|黑色星期一，股指期货与现金市场价差失控|传统头寸限额无法涵盖衍生品联动风险|
|**1991**|Metallgesellschaft 远期滚动套保巨亏|银行没有日盯市 → 损失被拖延发现|
|**1992**|Procter & Gamble 利率互换“爆仓”|VaR/压力测试缺位，董事会不懂产品|
<!-- bilingual-en:start -->
| **Year shown in the note** | **Event** | **Risk-management gap exposed** |
| --- | --- | --- |
| **1987** | Black Monday and severe dislocation between index futures and cash markets | Traditional position limits did not capture linked derivative and cash-market risk |
| **1993** | Metallgesellschaft's large losses from rolling oil hedges | Liquidity, basis, and funding risk could overwhelm a hedge even when its long-run economics appeared defensible |
| **1994** | Procter & Gamble's leveraged interest-rate derivatives loss | Boards and users could misunderstand nonlinear products when valuation, limits, and stress testing were weak |
<!-- bilingual-en:end -->

面对这些教训，G30（汇聚交易商、财务主管、律师及学者）用一份 40 页报告列出 **20 项操作守则 + 4 条监管呼吁**，成为业界第一份“衍生品风险管理 ISO 标准” 。
<!-- bilingual-en:start -->
In response to such failures, the Group of Thirty brought together dealers, treasurers, lawyers, and academics and issued *Derivatives: Practices and Principles*. Its **20 recommendations for dealers and end-users plus four recommendations for legislators, regulators, and supervisors** became an early industry blueprint for derivative-risk governance.
<!-- bilingual-en:end -->

| **主题**         | **关键守则（选摘 & 编号）**                                                                            | **要解决的痛点**      | **PPT 章节** |
| -------------- | -------------------------------------------------------------------------------------------- | --------------- | ---------- |
| **A. 治理与文化**   | (1)董事会批准风险政策；(2)独立风险部门；(3)制定清晰授权矩阵                                                           | 风险“谁拍板、谁监督”不清   |            |
| **B. 计量与监控**   | (4)每日盯市 _(mark-to-market)_；(5)统一 **VaR** 口径；(6)设置头寸限额 _(limits)_；(7)[[Stress Testing|压力测试]]                     | 账面价格滞后 & 模型口径各异 |            |
| **C. 信用风险管理**  | (8)净额结算应计入敞口；(9)设交易对手限额独立于前台；(10)审慎使用抵押品 / 保证金；(11)关注潜在未来敞口 _(PFE)_                          | 衍生品的双向信用风险被低估   |            |
| **D. 人才与系统**   | (12)保证交易、风险、后台人员资质；(13)IT 系统需捕获完整交易数据；(14)及时生成对账与管理报告                                        | “人/机”双短板导致操作风险  |            |
| **E. 财务报告与用途** | (15)衍生品收益来源要与被对冲项目配对披露；(16)禁止“纯投机”掩饰为对冲；(17)按公允价值列报表外项目；(18)披露模型假设；(19)与审计充分沟通；(20)持续评估政策有效性 | 提高透明度，抑制掩饰性风险   |            |
<!-- bilingual-en:start -->
| **Theme** | **Selected practices** | **Problem addressed** | **Slides** |
| --- | --- | --- | --- |
| **A. Governance and culture** | Board-approved risk policy; an independent risk function; clear delegated authorities | Unclear ownership and oversight of risk | |
| **B. Measurement and monitoring** | Daily mark-to-market; consistent **VaR** conventions; position limits; [[Stress Testing|stress testing]] | Stale valuations and inconsistent model conventions | |
| **C. Credit-risk management** | Recognise netting; set independent counterparty limits; use collateral and margin prudently; monitor potential future exposure (PFE) | Undermeasurement of two-way derivative credit exposure | |
| **D. People and systems** | Qualified front-, middle-, and back-office staff; complete transaction capture; timely reconciliation and management reports | Operational risk created by weak people or systems | |
| **E. Reporting and purpose** | Link hedge results to hedged items; distinguish hedging from speculation; fair-value reporting; disclose model assumptions; engage auditors; review policy effectiveness | Poor transparency and concealed risk | |
<!-- bilingual-en:end -->

> **口诀**：**“管人、管模、管限额；看市、看信用、看系统。”**
> <!-- bilingual-en:start -->
> **Mnemonic:** **“Govern people, models, and limits; monitor markets, credit, and systems.”**
> <!-- bilingual-en:end -->


1. **立法支持净额结算的法律效力**——让 ISDA Master Agreement 真正落地；否则银行无权在破产时抵销正负敞口。
<!-- bilingual-en:start -->
1. **Give netting agreements legal enforceability.** An ISDA Master Agreement cannot reduce exposure in insolvency unless the applicable legal regime permits close-out and offset of positive and negative replacement values.
<!-- bilingual-en:end -->
    
2. **推动公开市场公平透明**——提升衍生品交易信息披露，防范信息不对称。
<!-- bilingual-en:start -->
2. **Promote fair and transparent markets.** Improve disclosure about derivative activity and reduce information asymmetry.
<!-- bilingual-en:end -->
    
3. **监管机构应评价银行 VaR 与压力测试质量**，并将其纳入资本要求。
<!-- bilingual-en:start -->
3. **Require supervisors to assess the quality of banks' VaR and stress testing** and reflect deficiencies in supervisory action or capital.
<!-- bilingual-en:end -->
    
4. **跨国监管合作**——避免监管套利，提高对复杂跨境衍生品的监督力度。
<!-- bilingual-en:start -->
4. **Coordinate supervision across borders.** This limits regulatory arbitrage and improves oversight of complex cross-border derivative activity.
<!-- bilingual-en:end -->


|**G30 守则 →**|**1996 市场风险修正案**|**[[Basel Accords|Basel]] II / Pillar-3**|
|---|---|---|
|日盯市、VaR(99%,10d)|被写进 **内部模型法** 资本公式 Max(VaRt-1, mc × VaRavg)|银行须披露 VaR、压力测试及限额执行情况，供市场监督|
|回溯测试 & 惩罚系数 mc|例外次数 5-9 对应 mc 3.4-3.85 表格直接源自 G30 思想|—|
|净额结算计量|引入 **NRR**（净替换比率）公式显著降低 [[Risk-Weighted Assets|RWA]]|—|
|独立风险部 / 人才资质|被 [[Basel Accords|Basel]] II 第 2 支柱“监管审查过程”吸收，视作合规条件|—|
<!-- bilingual-en:start -->
| **G30 practice** | **1996 Market Risk Amendment** | **[[Basel Accords|Basel]] II and Pillar 3** |
| --- | --- | --- |
| Daily marking to market and ten-day 99% VaR | Incorporated into the historical internal-models capital approach | Banks disclose market-risk measures and risk-management practices |
| Backtesting and a multiplier add-on | Five to nine exceptions raised the multiplier from 3.40 to 3.85; ten or more raised it to 4.00 | — |
| Recognition of close-out netting | The net-to-gross ratio reduced the potential-future-exposure add-on | — |
| Independent risk control and qualified staff | Reflected in supervisory review and qualitative model-approval standards | — |
<!-- bilingual-en:end -->

**结论**：G30 = [[Basel Accords|Basel]] I 的“流程补丁”，对之后所有巴塞尔迭代具有“原型设计”意义。
<!-- bilingual-en:start -->
**Conclusion:** The G30 recommendations were not a formal amendment to [[Basel Accords|Basel]] I, but they supplied governance and measurement practices that influenced later derivatives supervision and Basel implementation.
<!-- bilingual-en:end -->

# 4.[[Netting|净额结算]]
<!-- bilingual-en:start -->
*4. [[Netting|Netting]]*
<!-- bilingual-en:end -->

## 4.1 什么叫净额结算？为什么能降信用风险
<!-- bilingual-en:start -->
*4.1 What Is Netting, and Why Does It Reduce Credit Risk?*
<!-- bilingual-en:end -->

|**名称**|**定义**|**风险削减机理**|
|---|---|---|
|**支付净额 (payment [[Netting|netting]])**|到期日把正负现金流先抵销，只结算“净额”|发生违约前就已减少待收/待付金额|
|**清算/关闭净额 (close-out [[Netting|netting]])**|若对手违约，双方所有合约同时终止，按**净赔偿额**结算|把“挑好合约赖账、挑坏合约履约”的 **选摘权**（cherry-picking）拔掉，确保你的正价值和负价值“同生共死”|
<!-- bilingual-en:start -->
| **Type** | **Definition** | **How it reduces risk** |
| --- | --- | --- |
| **Payment [[Netting|netting]]** | Offset cash flows due on the same settlement date and transfer only the net amount | Reduces the amount awaiting payment before a default occurs |
| **Close-out [[Netting|netting]]** | Upon default, terminate covered contracts and combine their replacement values into one net claim | Prevents selective performance, or “cherry-picking,” and makes positive and negative contract values part of one close-out set |
<!-- bilingual-en:end -->

在 OTC 衍生品里，**ISDA Master Agreement + CSA** 赋予 close-out [[Netting|netting]] 的法律效力。若获监管认可，银行可在资本计算里使用“净敞口”而不是“逐笔敞口”，信用 [[Risk-Weighted Assets|RWA]] 立刻打折。
<!-- bilingual-en:start -->
For OTC derivatives, an enforceable **ISDA Master Agreement plus a CSA** can support close-out [[Netting|netting]]. When the legal and regulatory conditions are met, exposure is calculated for the netting set rather than by simply adding every positive trade exposure, reducing credit [[Risk-Weighted Assets|RWA]].
<!-- bilingual-en:end -->

## 4.2 **CEM → NRR → [[EAD|EAD]] → [[Risk-Weighted Assets|RWA]]**

1. **现期敞口法 (Current Exposure Method, CEM)**
<!-- bilingual-en:start -->
1. **Current Exposure Method (CEM)**
<!-- bilingual-en:end -->
    
    - 每笔合约敞口 =  max(V,0)  (正价值) +  α × L  (未来潜在敞口 _add-on_)
<!-- bilingual-en:start -->
- Exposure for a trade equals current exposure $\max(V,0)$ plus a potential-future-exposure add-on $\alpha\times L$, where $L$ is notional amount under the notation used here.
<!-- bilingual-en:end -->
        
2. **净替换比率 NRR**
<!-- bilingual-en:start -->
2. **Net-to-gross ratio (NGR), called NRR in parts of the source note**
<!-- bilingual-en:end -->
    
    $\text{NRR}=\frac{\sum_{i=1}^{N}\max(V_i,0)}{\sum_{i=1}^{N}|V_i|}$
    
    ——用**净额正敞口** ÷ **绝对敞口** 量化净额效率（范围 0–1）。
<!-- bilingual-en:start -->
It measures the effectiveness of netting as **net current replacement cost divided by gross positive replacement cost**, and lies between 0 and 1.
<!-- bilingual-en:end -->
    
3. **等价信用量 ([[EAD|EAD]])** — 有净额时
<!-- bilingual-en:start -->
3. **Exposure at default ([[EAD|EAD]]) with recognised netting**
<!-- bilingual-en:end -->
    
    $\text{[[EAD|EAD]]}= \underbrace{\sum_{i}\max(V_i,0)}{\text{现期}} \;+\;\bigl(0.4 + 0.6 \times \text{NRR}\bigr)\,\times \sum{i}L_i$
    
    没有净额时就是$\sum \max(V_i,0) + \sum \alpha_i L_i$。
<!-- bilingual-en:start -->
A clean form of the historical CEM calculation is $\text{EAD}=\sum_i\max(V_i,0)+(0.4+0.6\,\text{NGR})\sum_i\alpha_iL_i$, subject to the precise legal-netting and regulatory rules.
<!-- bilingual-en:end -->
    
4. **[[Risk-Weighted Assets|风险加权资产]] ([[Risk-Weighted Assets|RWA]])**
<!-- bilingual-en:start -->
Without recognised netting, the corresponding gross calculation is $\sum \max(V_i,0) + \sum \alpha_i L_i$.
<!-- bilingual-en:end -->
    
    $\text{[[Risk-Weighted Assets|RWA]]} = \text{[[EAD|EAD]]}\, \times \text{对手权重}$

其中,add-on
<!-- bilingual-en:start -->
4. **[[Risk-Weighted Assets|Risk-weighted assets]] ([[Risk-Weighted Assets|RWA]])**
<!-- bilingual-en:end -->

| **合约类型**  | **≤ 1 年** | **1 – 5 年** | **> 5 年** | **来源** |
| --------- | --------- | ----------- | --------- | ------ |
| **利率衍生品** | 0 %       | **0.5 %**   | **1.5 %** |        |
| **商品†**   | **10 %**  | 12 %        | 15 %      |        |
<!-- bilingual-en:start -->
$\text{RWA} = \text{EAD}\times\text{counterparty risk weight}$.
<!-- bilingual-en:end -->

† 金价或外汇衍生品用另一列；题目里写的是“一般商品”，因此用 Commodity 列。
<!-- bilingual-en:start -->
The add-on coefficients used in the note are:
<!-- bilingual-en:end -->

NRR-[[Theta|theta]]-[[EAD|EAD]]-RAW-节省百分比
<!-- bilingual-en:start -->
| **Contract type** | **≤1 year** | **1–5 years** | **>5 years** | **Historical table** |
| --- | ---: | ---: | ---: | --- |
| **Interest-rate derivatives** | 0% | **0.5%** | **1.5%** | CEM add-on |
| **General commodities** | **10%** | 12% | 15% | CEM add-on |
<!-- bilingual-en:end -->

# 5.1996修正案
<!-- bilingual-en:start -->
*Gold and foreign-exchange derivatives use different columns in the historical table. The question discussed here specifies a general commodity, so it uses the commodity column.*
<!-- bilingual-en:end -->

> **目标**：把“VaR × 惩罚系数 mc”资本公式拆开讲，直到你能自己算出一家银行的市场风险资本。
> <!-- bilingual-en:start -->
> NGR → netting multiplier → [[EAD|EAD]] → [[Risk-Weighted Assets|RWA]] → percentage capital saving
> <!-- bilingual-en:end -->

 **1️⃣ 为什么 Basle I 之后还需要“1996 修正案”？**
<!-- bilingual-en:start -->
5. The 1996 Market Risk Amendment
<!-- bilingual-en:end -->

| **隐患**          | **[[Basel Accords|Basel]] I 没覆盖**     | **危险在哪**         |
| --------------- | ------------------- | ---------------- |
| **交易簿波动**       | [[Basel Accords|Basel]] I 只针对贷款等“银行簿” | 做市业务持仓大、价格日内跳动剧烈 |
| **债券利差 & 股票价格** | 权重 100 % 太粗糙        | 一条利率曲线扭一扭，债券亏几千万 |
| **衍生品敏感度**      | 仅算信用 Add-on         | 市场价格波动对期权、期货影响巨大 |
<!-- bilingual-en:start -->
**Objective:** unpack the historical “VaR × regulatory multiplier” formula until you can calculate a bank's market-risk capital charge.
<!-- bilingual-en:end -->

**结论**：需要一套**量化市场风险**且能接入日常风控的资本工具——于是《1996 市场风险修正案》诞生。
<!-- bilingual-en:start -->
**1️⃣ Why was the 1996 amendment needed after Basel I?**
<!-- bilingual-en:end -->

 **2️⃣ 两条路线：**
<!-- bilingual-en:start -->
| **Gap** | **What [[Basel Accords|Basel]] I did not adequately cover** | **Why it mattered** |
| --- | --- | --- |
| **Trading-book volatility** | Basel I was centred on banking-book credit risk | Large market-making inventories could change value sharply within a day |
| **Interest rates, credit spreads, and equity prices** | A broad 100% credit weight was too coarse | Yield-curve and spread movements could generate large trading losses |
| **Derivative sensitivities** | Credit add-ons did not measure market-price sensitivity | Options and futures could react nonlinearly to market moves |
<!-- bilingual-en:end -->

**标准法** **vs** **内部模型法 (IMA)**
<!-- bilingual-en:start -->
**Conclusion:** A capital framework was needed that quantified market risk and connected with daily risk management. This led to the 1996 Market Risk Amendment.
<!-- bilingual-en:end -->

| **路线**    | **监管给的公式**                      | **银行要做什么**                    |
| --------- | ------------------------------- | ----------------------------- |
| **标准法**   | 为各资产类别设“[[duration|久期]]-区段”或“贝塔权数”，逐块加总      | 会计科目+表格填报，简单但资本通常较高           |
| **内部模型法** | **VaR10d, 99 % × mc** + SRC特定风险 | 自己搭 VaR 引擎，满足 **[[Model Validation|模型验证]] & 回溯测试** |
<!-- bilingual-en:start -->
**2️⃣ Two routes:**
<!-- bilingual-en:end -->

> 后来 [[Basel Accords|Basel]] 2.5 / FRTB 继续演化 IMA，但 1996 版是第一代。
> <!-- bilingual-en:start -->
> **Standardised approach versus the internal models approach (IMA)**
> <!-- bilingual-en:end -->

 **3️⃣ 公式拆解（IMA 路线）**
<!-- bilingual-en:start -->
| **Route** | **Regulatory method** | **What the bank must do** |
| --- | --- | --- |
| **Standardised approach** | Apply prescribed duration bands, risk weights, and aggregation rules | Map accounting positions into regulatory tables; simple, but often conservative |
| **Internal models approach** | Historical formula based on ten-day 99% VaR and a multiplier, plus specific-risk capital | Build an approved VaR engine and satisfy qualitative standards, [[Model Validation|model validation]], and backtesting |
<!-- bilingual-en:end -->

1. **每日计算 1-day VaR(99 %)**——取过去 ≥ 1 年历史数据或蒙特卡洛。
2. **换算 10 天**：
    $VaR_{10d}=VaR_{1d}\times\sqrt{10}$
3. **惩罚系数 mc（模型校准）**
    $\text{Capital}{MR}= \max\bigl(VaR{t-1},\; mc\times \overline{VaR}_{60}\bigr)$
    - $VaR_{t-1}：昨日最新 10d VaR$
    - $\overline{VaR}_{60}：过去 60 天 10d VaR 均值$
    - **mc** 取值 3.0–4.0，由 **250 天回溯测试**“超越次数”决定（见下表）
<!-- bilingual-en:start -->
[[Basel Accords|Basel]] 2.5 and the Fundamental Review of the Trading Book later changed the market-risk framework substantially; the 1996 amendment was the first Basel internal-models regime.
<!-- bilingual-en:end -->

| **例外次数 (12 月)** | **mc** |
| --------------- | ------ |
| 0–4             | 3.0    |
| 5–9             | 3.4    |
| 10–14           | 3.5    |
| 15+             | 4.0    |
<!-- bilingual-en:start -->
**3️⃣ Decomposing the Historical IMA Formula**
<!-- bilingual-en:end -->

4. **SRC（[[Specific Variance|Specific]] Risk Capital）**——弥补 VaR 对债券违约/股票单一持仓过度分散带来的盲点（可用标准法或经批准的内部模型）。
<!-- bilingual-en:start -->
1. **Calculate one-day 99% VaR each day** from at least one year of history or an approved simulation model.
2. **Convert to a ten-day horizon** under the historical square-root rule: $VaR_{10d}=VaR_{1d}\times\sqrt{10}$.
3. **Apply the regulatory multiplier $m_c$:** $\text{Capital}_{MR}=\max\!\left(VaR_{t-1},\,m_c\overline{VaR}_{60}\right)$ under the simplified historical presentation here.
- $VaR_{t-1}$ is the latest ten-day VaR.
- $\overline{VaR}_{60}$ is the mean ten-day VaR over the preceding 60 business days.
- $m_c$ ranges from 3.0 to 4.0 and depends on the number of exceptions in a 250-day backtest.
<!-- bilingual-en:end -->

> **最终市场风险资本** = VaR × mc **+ SRC**。
> <!-- bilingual-en:start -->
> | **Exceptions in 250 trading days** | **Multiplier $m_c$** |
> | ---: | ---: |
> | 0–4 | 3.00 |
> | 5 | 3.40 |
> | 6 | 3.50 |
> | 7 | 3.65 |
> | 8 | 3.75 |
> | 9 | 3.85 |
> | 10 or more | 4.00 |
> <!-- bilingual-en:end -->

| **数据**                | **数值**                                            | **步骤**              |
| --------------------- | ------------------------------------------------- | ------------------- |
| 1-day VaR(99 %)       | $2 m                                              | 蒙特卡洛 or 历史          |
| **① 10-day VaR**      | 2 ×√10 ≈ \$6.32 m                                 | 扩⻓[[Holding Period|持有期]]               |
| 回溯测试 250 天 **例外 6 次** | mc = 3.4                                          | 查表                  |
| **② Capital core**    | $\max(6.32,\; 3.4×6.00)=\max(6.32,20.4)=\$20.4 m$ | 假设 60-日均 VaR = $6 m |
| **③ SRC**             | $4 m                                              | 用债券久期表法算            |
| **总市场风险资本**           | **$24.4 m**                                       | ② + ③               |
<!-- bilingual-en:start -->
4. **Specific risk capital charge (SRC).** This covered issuer-specific default and price risk not adequately captured by a diversified general-market VaR model, using either prescribed charges or an approved specific-risk model.
<!-- bilingual-en:end -->

**对比**：如果走标准法，很多中小行常被算出 >$30 m；因此只要模型合格，IMA 省资本显著。
<!-- bilingual-en:start -->
**Historical total market-risk charge** = VaR-based charge **+ SRC**.
<!-- bilingual-en:end -->

# 6.[[Basel Accords|Basel]] II

## 6.1 内容
<!-- bilingual-en:start -->
*| **Input or result** | **Value** | **Step** | | --- | ---: | --- | | One-day 99% VaR | USD 2 million | Monte Carlo or historical estimate | | **① Ten-day VaR** | $2\sqrt{10}\approx\$6.32$ million | Extend the [[Holding Period|holding period]] | | Six exceptions in 250 days | $m_c=3.50$ under the original schedule | Read the regulatory table | | **② Core VaR charge** | $\max(6.32,3.50\times6.00)=\$21.0$ million | Assume mean ten-day VaR is USD 6 million | | **③ SRC** | USD 4 million | Apply the specific-risk method | | **Total historical market-risk capital** | **USD 25.0 million** | ② + ③ |*
<!-- bilingual-en:end -->

| **支柱**                    | **问题**              | **监管答案（[[Basel Accords|Basel]] II 做了什么？）**                             |
| ------------------------- | ------------------- | ---------------------------------------------------- |
| **Pillar 1 最低资本金要求**      | _“到底要放多少钱当安全垫？”_    | 依照 **[[Credit Risk|信用风险]] + [[Market Risk|市场风险]] + [[Operational Risk|操作风险]]** 三块算 **[[Risk-Weighted Assets|RWA]]**，再 × 8 % 得所需资本。 |
| **Pillar 2 监管审查过程 (SRP)** | _“公式没覆盖的风险怎么办？”_    | 银行自己做 **ICAAP**（内部资本评估），监督机关可要求加码资本或限额。              |
| **Pillar 3 市场纪律**         | _“光靠官方盯不够，得让市场也盯。”_ | 公开披露 VaR、[[Risk-Weighted Assets|RWA]]、[[PD|PD]] / [[LGD|LGD]] 等关键信息，投资者会“用钱投票”。              |
<!-- bilingual-en:start -->
**Comparison:** A standardised calculation could produce a charge above USD 30 million in this illustration. An approved internal model could therefore reduce measured capital, which is precisely why model approval, backtesting, and supervisory floors matter.
<!-- bilingual-en:end -->

> **一句话记忆**：**“算本钱、查漏网、晒太阳。”**
> <!-- bilingual-en:start -->
> 6.1 Core Content
> <!-- bilingual-en:end -->

## 6.2 最低资本金要求
<!-- bilingual-en:start -->
*| **Pillar** | **Question** | **What [[Basel Accords|Basel]] II introduced** | | --- | --- | --- | | **Pillar 1: minimum capital requirements** | *How much capital must be held against measured risk?* | Calculate [[Risk-Weighted Assets|RWA]] for **[[Credit Risk|credit risk]], [[Market Risk|market risk]], and [[Operational Risk|operational risk]]**, then apply the applicable minimum capital ratios | | **Pillar 2: supervisory review process** | *What about material risks not captured by Pillar 1?* | Banks conduct ICAAP; supervisors assess it and can require extra capital, stronger controls, or limits | | **Pillar 3: market discipline** | *How can investors and counterparties also monitor the bank?* | Disclose material information about capital, [[Risk-Weighted Assets|RWA]], [[PD|PD]], [[LGD|LGD]], market risk, and risk management |*
<!-- bilingual-en:end -->

抵押品风险:
<!-- bilingual-en:start -->
**One-sentence mnemonic:** **“Calculate the minimum, review what the formula misses, and disclose the result to daylight.”**
<!-- bilingual-en:end -->

| **路线**                 | **关键词**         | **资本敏感度** | **你要做什么**                 |
| ---------------------- | --------------- | --------- | ------------------------- |
| **标准法 (SA)**           | 外部评级、风险权重表      | ★☆☆       | 直接查表：AAA 20 %、BB 150 %… ） |
| **内部评级法 – 基础 (F-IRB)** | 自估 [[PD|PD]]，监管给 [[LGD|LGD]]   | ★★☆       | 建 [[PD|PD]] 模型，评级系统须通过监督检查。      |
| **内部评级法 – 高级 (A-IRB)** | 自估 [[PD|PD]]+[[LGD|LGD]]+[[EAD|EAD]]+M | ★★★       | 全套自估，最省资本，但门槛高。           |
<!-- bilingual-en:start -->
6.2 Minimum Capital Requirements
<!-- bilingual-en:end -->

> **公式（简化）**：
> $K = [[LGD|LGD]] \Bigl[\,N\!\bigl(\tfrac{1}{\sqrt{1-R}}\,G([[PD|PD]]) + \sqrt{\tfrac{R}{1-R}}\;G(0.999)\bigr) - [[PD|PD]] \Bigr]$$\quad\Longrightarrow\quad [[Risk-Weighted Assets|RWA]] = 12.5 \times K \times [[EAD|EAD]]$
> <!-- bilingual-en:start -->
> Collateral and credit-risk treatment:
> <!-- bilingual-en:end -->


[[Market Risk|市场风险]]
<!-- bilingual-en:start -->
| **Route** | **Key inputs** | **Risk sensitivity** | **What the bank does** |
| --- | --- | --- | --- |
| **Standardised approach (SA)** | External ratings and prescribed risk weights | ★☆☆ | Apply the relevant table, such as the historical 20% or 150% corporate weights under specified conditions |
| **Foundation IRB (F-IRB)** | Bank estimates [[PD|PD]]; regulation supplies other parameters such as [[LGD|LGD]] | ★★☆ | Build and validate a [[PD|PD]] rating system subject to supervisory approval |
| **Advanced IRB (A-IRB)** | Bank estimates [[PD|PD]], [[LGD|LGD]], [[EAD|EAD]], and maturity inputs | ★★★ | Maintain a fully validated internal system; potentially more risk-sensitive, but subject to much higher approval and governance standards |
<!-- bilingual-en:end -->

- **1996 修正案的 VaR × mc** 直接并入 [[Basel Accords|Basel]] II。
- 如果你已通过 **内部模型法 (IMA)** 回溯测试，就把那套数字塞进 Pillar 1。
<!-- bilingual-en:start -->
- The 1996 amendment's historical VaR-and-multiplier approach was incorporated into the market-risk component of [[Basel Accords|Basel]] II.
- A bank using the **internal models approach (IMA)** had to obtain approval and pass ongoing backtesting before its model output could enter Pillar 1.
<!-- bilingual-en:end -->

[[Operational Risk|操作风险]]（[[Basel Accords|Basel]] I 完全没管到的“新成员”）
<!-- bilingual-en:start -->
[[Operational Risk|Operational risk]], a major category that [[Basel Accords|Basel]] I did not explicitly capitalise.
<!-- bilingual-en:end -->

|**路线**|**口径**|**计算法**|**资本量级**|
|---|---|---|---|
|**BIA** 基础指标法|“毛利 × 15 %”|过去三年平均营业净收入 × 15 %|最高|
|**TSA/SA** 标准法|分业务线 × 系数 (12–18 %)|零售、批发、交易等各自套系数再加和|中等|
|**AMA** 高级计量法|自建模型，VaR(99.9 %, 1y)|需历史损失数据库、[[Scenario Analysis|情景分析]]|最低，但监管门槛高|
<!-- bilingual-en:start -->
| **Historical route** | **Basis** | **Calculation** | **General capital effect** |
| --- | --- | --- | --- |
| **BIA: Basic Indicator Approach** | A fixed percentage of gross income | Average positive annual gross income over three years × 15% | Coarse and often conservative |
| **TSA/SA: Standardised Approach** | Business-line income × prescribed beta factors | Apply 12–18% factors by business line and aggregate | More differentiated |
| **AMA: Advanced Measurement Approaches** | Approved internal operational-risk model | Internal and external loss data, [[Scenario Analysis|scenario analysis]], and control factors | Potentially more risk-sensitive, but with demanding approval requirements |
<!-- bilingual-en:end -->

##  6.3监管审查过程
<!-- bilingual-en:start -->
*6.3 Supervisory Review Process*
<!-- bilingual-en:end -->

1. **你有没有把所有重大风险都写进 ICAAP？**
2. **资本缓冲够不够 cover 压力情景？**
3. **治理、内部审计、数据质量合格吗？**
<!-- bilingual-en:start -->
1. **Does ICAAP cover every material risk?**
2. **Is the capital buffer sufficient under stress?**
3. **Are governance, internal audit, and data quality adequate?**
<!-- bilingual-en:end -->

> 若答得敷衍，监管可以“Pillar 2 加层压舱石”——让你比公式多放资本或直接砍仓。
> <!-- bilingual-en:start -->
> If the answers are weak, supervisors can use Pillar 2 to require capital above the formulaic minimum, strengthen controls, impose limits, or reduce positions.
> <!-- bilingual-en:end -->

## 6.4披露清单
<!-- bilingual-en:start -->
*6.4 Disclosure Checklist*
<!-- bilingual-en:end -->

- **量化表**：[[Risk-Weighted Assets|RWA]] 分布、平均 [[PD|PD]]、违约资产回收率。
- **质化文字**：风险管理组织架构、VaR 模型假设、应急流动性计划。
- **更新频率**：至少半年一次，大行通常随年报附 _Pillar 3 Report_。
<!-- bilingual-en:start -->
- **Quantitative disclosures:** the composition of [[Risk-Weighted Assets|RWA]], average [[PD|PD]], recovery or loss rates, and capital ratios.
- **Qualitative disclosures:** governance, risk-management organisation, VaR model assumptions, and contingency funding plans.
- **Frequency:** the source note uses at least semiannual disclosure as a memory rule; large banks commonly publish a dedicated *Pillar 3 Report* alongside regular reporting.
<!-- bilingual-en:end -->

**好处**：投资者、评级机构用脚投票 → 让银行“怕丢脸、怕股价跌”。
<!-- bilingual-en:start -->
**Benefit:** investors, creditors, and rating agencies can react to disclosed risk, creating market discipline alongside formal supervision.
<!-- bilingual-en:end -->

# 7. 偿付能力法案 II
<!-- bilingual-en:start -->
*7. Solvency II*
<!-- bilingual-en:end -->

## **0. 为什么银行“巴塞尔”、保险“偿二代”要分家？**
<!-- bilingual-en:start -->
*0. Why Do Banks Use Basel While Insurers Use Solvency II?*
<!-- bilingual-en:end -->

|**行业**|**典型风险**|**资产负债特征**|**监管思路**|
|---|---|---|---|
|**银行**|信贷违约、市场波动、操作|负债随时可被提走（存款）|**流动性 + 短期价格冲击**|
|**保险**|赔付不确定、市场波动|负债久、资产久（保单 10–20 年）|**承保波动 + 市场波动**|
<!-- bilingual-en:start -->
| **Industry** | **Typical risks** | **Balance-sheet structure** | **Regulatory emphasis** |
| --- | --- | --- | --- |
| **Banks** | Credit default, market moves, liquidity, and operations | Deposits and wholesale funding may run or mature quickly | **Liquidity resilience, loss absorption, and shorter-horizon market shocks** |
| **Insurers** | Uncertain claims, underwriting, market, and counterparty risk | Long-duration assets and policy liabilities, often extending 10–20 years | **Underwriting uncertainty and one-year changes in own funds** |
<!-- bilingual-en:end -->

🔑 **关键词对照**
<!-- bilingual-en:start -->
🔑 **Key comparison**
<!-- bilingual-en:end -->

- **[[Basel Accords|Basel]] 的 VaR 99 % ×10d** → 捕捉“10 天里面的最坏亏损”
<!-- bilingual-en:start -->
- The **ten-day 99% VaR** convention belongs to the historical [[Basel Accords|Basel]] market-risk internal-models framework; current FRTB market-risk rules use expected shortfall and liquidity horizons instead.
<!-- bilingual-en:end -->
    
- **Solvency II 的 VaR 99.5 % ×1 y** → 问“一年之内活不过去的概率≤0.5 %”
<!-- bilingual-en:start -->
- **Solvency II calibrates the SCR to a one-year 99.5% VaR standard for basic own funds**, corresponding to a 0.5% probability of a larger one-year loss under the model.
<!-- bilingual-en:end -->

## **1. Solvency II 三层结构（和 [[Basel Accords|Basel]] 三支柱对着背）**
<!-- bilingual-en:start -->
*1. Solvency II's Three Pillars, Compared with the [[Basel Accords|Basel]] Structure*
<!-- bilingual-en:end -->

|**Solvency II**|**[[Basel Accords|Basel]] “对应物”**|**口语速记**|
|---|---|---|
|**1. SCR / MCR 最低资本**|Pillar 1 最低资本|保险版“8 %”——但用 VaR_{99.5\%,1y}|
|**2. 监管审查（ORSA）**|Pillar 2 ICAAP|保险自己写“活下来”剧本，监管可加资本|
|**3. 公开披露（SFCR）**|Pillar 3 披露|年度“体检报告”给市场看|
<!-- bilingual-en:start -->
| **Solvency II** | **Closest [[Basel Accords|Basel]] analogue** | **Plain-language memory cue** |
| --- | --- | --- |
| **1. SCR and MCR quantitative requirements** | Pillar 1 minimum requirements | Capital thresholds based on insurer risks, not a flat “insurance 8%” |
| **2. Supervisory review and ORSA** | Pillar 2 and ICAAP | The insurer assesses how it remains solvent under its own risk profile; supervisors can intervene |
| **3. Public reporting and SFCR** | Pillar 3 disclosure | A regular solvency and financial-condition report for the public |
<!-- bilingual-en:end -->

> **公式感**
> <!-- bilingual-en:start -->
> **Capital intuition**
> <!-- bilingual-en:end -->

- > **MCR**（Minimum Capital Requirement）≈ “警戒线”；低于就立刻干预
- > **SCR**（Solvency Capital Requirement）≈ “安全垫”；要保持在上方
<!-- bilingual-en:start -->
- > **MCR**, the Minimum Capital Requirement, is the lower intervention threshold; breach can trigger severe supervisory action.
- > **SCR**, the Solvency Capital Requirement, is the risk-based solvency buffer that eligible own funds are expected to cover.
<!-- bilingual-en:end -->


## **2. 怎么算** 
<!-- bilingual-en:start -->
*2. How Is the SCR Calculated?*
<!-- bilingual-en:end -->

## **SCR**

## **？──两条路线**
<!-- bilingual-en:start -->
*Two Routes*
<!-- bilingual-en:end -->

|**路线**|**用什么表**|**适合谁**|**思路一句话**|
|---|---|---|---|
|**标准公式**|EIOPA 给的模块权数|中小保险|像拼乐高：市场 + 承保 + 信用 + 操作|
|**内部模型**|监管批准后自算|大型跨国险企|类似银行 VaR 模型，能省资本|
<!-- bilingual-en:start -->
| **Route** | **Inputs** | **Typical user** | **Core idea** |
| --- | --- | --- | --- |
| **Standard formula** | EIOPA-prescribed shocks, factors, and correlations | Many small and medium insurers | Aggregate market, underwriting, counterparty, and operational risks using prescribed modules |
| **Internal model** | A full or partial model approved by the supervisor | Larger or more complex insurers | Model the insurer's own risk profile subject to validation, governance, and calibration standards |
<!-- bilingual-en:end -->

### **标准公式 6 块乐高**
<!-- bilingual-en:start -->
*Six Simplified Building Blocks in the Standard Formula*
<!-- bilingual-en:end -->

1. **[[Market Risk|市场风险]]**（利率、股指、汇率、房地产）
2. **寿险承保风险**（死亡、长寿、费用、Lapse）
3. **非寿险承保风险**（赔付波动、灾变）
4. **健康险风险**
5. **信用/对手风险**
6. **[[Operational Risk|操作风险]]**
<!-- bilingual-en:start -->
1. **[[Market Risk|Market risk]]**, including interest rates, equities, foreign exchange, property, spreads, and concentration
2. **Life underwriting risk**, including mortality, longevity, expenses, and lapse
3. **Non-life underwriting risk**, including premium, reserve, and catastrophe risk
4. **Health underwriting risk**
5. **Counterparty default risk**
6. **[[Operational Risk|Operational risk]]**, calculated outside the basic-solvency-capital aggregation in the standard formula
<!-- bilingual-en:end -->
    

  

> **组合公式**：
> <!-- bilingual-en:start -->
> **Simplified aggregation formula:**
> <!-- bilingual-en:end -->

> $SCR = \sqrt{\sum_i \sum_j Corr_{ij} \, SCR_i \, SCR_j}$

> （监管提供相关矩阵 Corr_{ij}）
> <!-- bilingual-en:start -->
> The supervisor provides the correlation matrix $Corr_{ij}$.
> <!-- bilingual-en:end -->

---

## **3. 1 分钟例子（用标准公式里的“单模块”口感）**
<!-- bilingual-en:start -->
*3. One-Minute Two-Module Example*
<!-- bilingual-en:end -->

  

> **简化假设**：一家寿险公司
> <!-- bilingual-en:start -->
> **Simplifying assumption:** consider a life insurer with only two modules.
> <!-- bilingual-en:end -->

- > 市场风险模块 SCR_M = 300 m
- > 寿险承保风险 SCR_S = 250 m
- > 两模块相关系数 Corr = 25 %
<!-- bilingual-en:start -->
- > Market-risk module $SCR_M=300$ million
- > Life-underwriting module $SCR_S=250$ million
- > Correlation between the modules $Corr=25\%$
<!-- bilingual-en:end -->

计算：
<!-- bilingual-en:start -->
Calculation:
<!-- bilingual-en:end -->

$SCR = \sqrt{300^2 + 250^2 + 2×0.25×300×250} \approx \sqrt{90{,}000 + 62{,}500 + 37{,}500} = \sqrt{190{,}000} ≈ 436 m$

- **最低资本** = 436 m
- **Tier 1 要占 ≥ 50 %** → ≥218 m
- **Tier 2/3** 可以补剩下的一部分（监管上限 50 %）
<!-- bilingual-en:start -->
- The aggregated **SCR** is approximately 436 million.
- Under the simplified eligibility rule used in this note, Tier 1 own funds must cover at least 50%, or at least 218 million.
- Eligible Tier 2 and, subject to stricter limits, Tier 3 own funds may cover part of the balance. This is an own-funds eligibility rule, not a calculation of the MCR.
<!-- bilingual-en:end -->
    
> **对比**：[[Basel Accords|Basel]] 里我们先算 [[Risk-Weighted Assets|RWA]]，再 ×8 %；Solvency II 直接给出“需要多少自己资本额”。
> <!-- bilingual-en:start -->
> **Comparison:** The historical Basel presentation first calculated [[Risk-Weighted Assets|RWA]] and then applied capital ratios. Solvency II instead aggregates risk-module capital charges to obtain a euro amount of required insurer own funds.
> <!-- bilingual-en:end -->

---

## **4. 和 [[Basel Accords|Basel]] 数字对着看，更易记！**
<!-- bilingual-en:start -->
*4. Side-by-Side Memory Table for Historical Basel and Solvency II*
<!-- bilingual-en:end -->

|**项目**|**[[Basel Accords|Basel]] 8 %**|**Solvency II**|
|---|---|---|
|信用 / 市场 VaR|99 % ×10d|99.5 % ×1y|
|[[Operational Risk|操作风险]]|BIA/SA/AMA|专门模块|
|模型惩罚系数|mc (回溯)|“Model change buffer” + 监管校准|
|核心资本比例|Tier 1 ≥50 % 总资本|同（Tier 1 ≥50 % SCR）|
<!-- bilingual-en:start -->
| **Item** | **Historical [[Basel Accords|Basel]] convention in these notes** | **Solvency II** |
| --- | --- | --- |
| Market-risk horizon and tail measure | Ten-day 99% VaR under the old IMA | One-year 99.5% VaR calibration for the SCR |
| [[Operational Risk|Operational risk]] | Historical BIA, TSA, or AMA | Separate standard-formula charge or internal-model component |
| Model supervision | Backtesting multiplier and qualitative standards | Internal-model approval, validation, calibration, and model-change policy |
| Highest-quality capital | Historical Tier 1 minimum equal to half the 8% total minimum | Tier 1 must cover at least 50% of the SCR under the simplified eligibility comparison |
<!-- bilingual-en:end -->

如需更细分主题，可按需展开下一节（逐条推导与表内口径说明）。~~告诉我你想 dive 的主题，我再按“超慢速”模式开下一节！~~
<!-- bilingual-en:start -->
More detailed topics can be expanded section by section, including derivations and the precise scope of each regulatory table. ~~The final sentence in the source is an AI-generated invitation to request another topic.~~
<!-- bilingual-en:end -->

# 作业
<!-- bilingual-en:start -->
*Homework*
<!-- bilingual-en:end -->

## 15.1
>[!question] 
>“当一家钢铁企业破产时，其他在同一行业的企业可能会受益，因为这时竞争对手少了一个。但是当一家银行破产时，其他银行并不一定受益。”  请解释这一观点。  
><!-- bilingual-en:start -->
>“When a steel producer fails, other firms in the industry may benefit because one competitor has disappeared. When a bank fails, however, other banks do not necessarily benefit.” Explain this claim.
><!-- bilingual-en:end -->

**答题序号：15.1  分值：___**
<!-- bilingual-en:start -->
**Question 15.1  Marks: ___**
<!-- bilingual-en:end -->

1. **比较两类企业的竞争结构**
<!-- bilingual-en:start -->
1. **Compare the competitive structures of the two industries**
<!-- bilingual-en:end -->
    
    - _制造业（钢铁）_：产品同质、供给弹性有限。单一家厂商退出 → **行业供给减少** → 价格及剩余企业市场份额↑ → 直接收益。
<!-- bilingual-en:start -->
- *Steel manufacturing:* products are relatively homogeneous and short-run supply may be inelastic. One producer's exit reduces industry supply, potentially raising prices and the surviving firms' market shares.
<!-- bilingual-en:end -->
        
    - _银行业_：核心服务是**金融中介与支付清算**，高度互联，客户对“安全/信任”敏感；一家银行退出 ≠ 简单减少供给。
<!-- bilingual-en:start -->
- *Banking:* the core activities are financial intermediation, payments, and settlement. Banks are highly interconnected, and customers care intensely about safety and trust. One bank's exit is therefore not merely a reduction in supply.
<!-- bilingual-en:end -->
        
2. **外部性差异**
<!-- bilingual-en:start -->
2. **The externalities differ**
<!-- bilingual-en:end -->
    
    - _制造业_：破产主要是**私人成本**，负面外部性小。
<!-- bilingual-en:start -->
- *Manufacturing:* much of the cost of failure is borne privately by owners, creditors, workers, and local stakeholders, with less immediate financial-system contagion.
<!-- bilingual-en:end -->
        
    - _银行业_：破产带来 **系统性外部性**：
<!-- bilingual-en:start -->
- *Banking:* failure can create **systemic externalities:**
<!-- bilingual-en:end -->
        
        1. **信息传染**：投资者/存款人恐慌，质疑其他银行资产质量 → 同业融资收缩；
<!-- bilingual-en:start -->
1. **Information contagion:** investors and depositors may question the asset quality of other banks, causing withdrawals and a contraction in interbank funding.
<!-- bilingual-en:end -->
            
        2. **直接连锁**：同业拆借、衍生品敞口、支付系统结算链条 → 违约敞口扩散；
<!-- bilingual-en:start -->
2. **Direct balance-sheet contagion:** interbank loans, derivative exposures, and payment-system obligations can transmit default losses to counterparties.
<!-- bilingual-en:end -->
            
        3. **流动性挤兑**：存款转移至钱荒或货币基金，**同业负债成本↑**。
<!-- bilingual-en:start -->
3. **Liquidity runs:** deposits and wholesale funding may flee to cash, money-market funds, or perceived safe banks, raising surviving banks' funding costs.
<!-- bilingual-en:end -->
            
3. **监管与品牌效应**
<!-- bilingual-en:start -->
3. **Regulatory and reputational effects**
<!-- bilingual-en:end -->
    
    - 银行业受**严格监管**；破产触发监管接管、FDIC 赔付、处置成本由行业分摊（如存款保险费率上升），**使幸存银行承担额外负担**。
<!-- bilingual-en:start -->
- Banking is heavily regulated. Failure can trigger resolution, insured-deposit payouts, and industry-funded costs such as higher deposit-insurance assessments, burdening surviving banks.
<!-- bilingual-en:end -->
        
    - 信誉损失：整体行业信任度下降 → 需求外逃至国债、货币市场基金或大型国际银行，**幸存银行难以获利**。
<!-- bilingual-en:start -->
- Trust can fall across the sector, shifting demand toward government securities, money-market funds, or large perceived-safe institutions and weakening the profitability of other banks.
<!-- bilingual-en:end -->
        
4. **结论**
<!-- bilingual-en:start -->
4. **Conclusion**
<!-- bilingual-en:end -->
    
    - 钢铁企业破产 → 供给侧退出，**价格效应与市场份额效应 > 外部性成本**，同行常常净受益。
<!-- bilingual-en:start -->
- A steel producer's failure removes supply, so the price and market-share gains to competitors may exceed the spillover costs.
<!-- bilingual-en:end -->
        
    - 银行破产 → **负外部性（系统性风险）大于竞争缓解带来的益处**，其他银行可能因恐慌、连锁违约、监管成本增加而受损，甚至导致行业整体不稳定。故“其他银行并不一定受益”。
<!-- bilingual-en:start -->
- A bank failure can impose contagion, liquidity, and resolution costs larger than any benefit from reduced competition. Other banks may therefore lose rather than gain, and the sector as a whole may become less stable.
<!-- bilingual-en:end -->
 
## 15.6 
 
>[!question] 
>采用《[[Basel Accords|巴塞尔协议]] I》计算下列与某企业进行的交易（无净额结算协议）所需的资本金：  
 (a) 9 年期利率互换，名义本金 2.5 亿美元，当前市价 –200 万美元；  
 (b) 4 年期利率互换，名义本金 1 亿美元，当前市价 350 万美元；  
 (c) 6 个月期商品衍生品交易，名义本金 5 000 万美元，当前市价 100 万美元。  
<!-- bilingual-en:start -->
Under the historical [[Basel Accords|Basel]] I current-exposure method and with no netting agreement, calculate the capital required for these transactions with a corporate counterparty:
(a) a nine-year interest-rate swap with notional USD 250 million and current market value negative USD 2 million;
(b) a four-year interest-rate swap with notional USD 100 million and current market value USD 3.5 million;
(c) a six-month general-commodity derivative with notional USD 50 million and current market value USD 1 million.
<!-- bilingual-en:end -->

解题思路
<!-- bilingual-en:start -->
Approach
<!-- bilingual-en:end -->

《[[Basel Accords|巴塞尔协议]] I》（1995 年 OTC 衍生品修订案）对**场外衍生品信用风险资本**的计算步骤：
<!-- bilingual-en:start -->
The historical OTC-derivatives amendment to [[Basel Accords|Basel]] I calculates **counterparty credit-risk capital** as follows:
<!-- bilingual-en:end -->

1. **算出当前曝险额 (Current Exposure, CE)**
<!-- bilingual-en:start -->
1. **Calculate current exposure (CE)**
<!-- bilingual-en:end -->
    
    - 仅计入 _正_ 的市价；负值视为 0。
<!-- bilingual-en:start -->
- Include only *positive* current market values; replace a negative value with zero.
<!-- bilingual-en:end -->
        
2. **算出潜在未来曝险额 (Potential Future Exposure, PFE)**
<!-- bilingual-en:start -->
2. **Calculate potential future exposure (PFE)**
<!-- bilingual-en:end -->
    
    $PFE=名义本金×附加系数(add-on)\text{PFE}= \text{名义本金}\times \text{附加系数(add-on)}$
    - 利率合约 add-on：0.0 % / 0.5 % / **1.5 %**（分别对应 ≤1 年、1–5 年、>5 年）。
<!-- bilingual-en:start -->
- Historical interest-rate add-ons are 0.0%, 0.5%, and **1.5%** for maturities ≤1 year, 1–5 years, and >5 years, respectively.
<!-- bilingual-en:end -->
        
    - 其他商品合约 add-on：**10 %**（所有期限统一）。
<!-- bilingual-en:start -->
- Under the general-commodity column used by this question, the ≤1-year add-on is **10%**.
<!-- bilingual-en:end -->
        
3. **信用当量 ([[Credit Risk|Credit]] Equivalent Amount, CEA)**
<!-- bilingual-en:start -->
3. **Credit Equivalent Amount ([[Credit Risk|Credit Equivalent Amount]], CEA)**
<!-- bilingual-en:end -->
    
    $CEA=CE+PFE\text{CEA}= \text{CE}+ \text{PFE}$
4. **[[Risk-Weighted Assets|风险加权资产]] ([[Risk-Weighted Assets|RWA]])**
<!-- bilingual-en:start -->
4. **[[Risk-Weighted Assets|Risk-weighted assets]] ([[Risk-Weighted Assets|RWA]])**
<!-- bilingual-en:end -->
    
    - 假设交易对手为一般企业 ⇒ **风险权重 100 %**。
<!-- bilingual-en:start -->
- Assume an ordinary corporate counterparty and the historical **100% risk weight**.
<!-- bilingual-en:end -->
        
    
    $[[Risk-Weighted Assets|RWA]]=CEA×100\%\text{[[Risk-Weighted Assets|RWA]]}= \text{CEA}\times100\%$
5. **资本要求**（巴塞尔 I 统一乘 8 %）
<!-- bilingual-en:start -->
5. **Capital requirement:** apply the historical 8% total-capital minimum under Basel I.
<!-- bilingual-en:end -->
    
    $K=0.08×RWAK = 0.08 \times \text{[[Risk-Weighted Assets|RWA]]}$

 逐项计算
<!-- bilingual-en:start -->
Calculate each transaction separately:
<!-- bilingual-en:end -->

| 项目        | 名义本金 (美元) | 期限          | 市价 CE (美元)  | add-on    | PFE (美元)                 | CEA (美元)                  | [[Risk-Weighted Assets|RWA]] (美元) | 资本金 K = 8 %×[[Risk-Weighted Assets|RWA]] |
| --------- | --------- | ----------- | ----------- | --------- | ------------------------ | ------------------------- | -------- | --------------- |
| (a) 利率互换  | 2.5 亿     | 9 年 (>5 年)  | 0（–200 万取0） | **1.5 %** | 2.5e8×1.5 %= **3.75 百万** | **3.75 百万**               | 3.75 百万  | **0.30 百万**     |
| (b) 利率互换  | 1 亿       | 4 年 (1–5 年) | **3.50 百万** | **0.5 %** | 1e8×0.5 %= **0.50 百万**   | 3.50 + 0.50 = **4.00 百万** | 4.00 百万  | **0.32 百万**     |
| (c) 商品衍生品 | 5 千万      | 0.5 年       | **1.00 百万** | **10 %**  | 5e7×10 %= **5.00 百万**    | 1.00 + 5.00 = **6.00 百万** | 6.00 百万  | **0.48 百万**     |
<!-- bilingual-en:start -->
| Item | Notional (USD) | Maturity | Positive CE | Add-on | PFE | CEA | [[Risk-Weighted Assets|RWA]] | Capital at 8% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) Interest-rate swap | 250 million | 9 years (>5 years) | 0, because -2 million is floored at zero | **1.5%** | **3.75 million** | **3.75 million** | 3.75 million | **0.30 million** |
| (b) Interest-rate swap | 100 million | 4 years (1–5 years) | **3.50 million** | **0.5%** | **0.50 million** | **4.00 million** | 4.00 million | **0.32 million** |
| (c) General-commodity derivative | 50 million | 0.5 years | **1.00 million** | **10%** | **5.00 million** | **6.00 million** | 6.00 million | **0.48 million** |
<!-- bilingual-en:end -->

结论
<!-- bilingual-en:start -->
Conclusion
<!-- bilingual-en:end -->

- 各合约所需资本：
    - (a) **0.30 百万美元**
    - (b) **0.32 百万美元**
    - (c) **0.48 百万美元**
- **合计资本要求 ≈ 1.10 百万美元**
- 
关键步骤：确认正值市价 → 选取正确 add-on 系数 → 计算 PFE → 得到 CEA → 乘 100 % 风险权重 → 乘 8 % 得资本金。
<!-- bilingual-en:start -->
- Required capital by contract:
- (a) **USD 0.30 million**
- (b) **USD 0.32 million**
- (c) **USD 0.48 million**
- **Total capital requirement ≈ USD 1.10 million**

Key steps: floor negative market values at zero → select the product and maturity add-on → calculate PFE → add CE to obtain CEA → apply the 100% counterparty weight → apply the 8% capital ratio.
<!-- bilingual-en:end -->

## 15.10 

>[!question] 
>银行的**交易账户**与**银行账户**有何区别？  
  某银行目前持有一笔面值 1 000 万美元的贷款；在贷款到期时，客户希望改为把自己的债券出售给银行。  
  这种变化将怎样影响该行的监管资本金的数量？  
<!-- bilingual-en:start -->
How do the **trading book** and **banking book** differ? A bank currently holds a USD 10 million loan. At maturity, the customer proposes instead to sell the bank its own bond. How would the change affect the bank's regulatory capital?
<!-- bilingual-en:end -->

一、交易账户 (Trading Book) 与 银行账户 (Banking/Bank Book) 的区别
<!-- bilingual-en:start -->
I. Trading Book versus Banking Book
<!-- bilingual-en:end -->

|维度|交易账户|银行账户|
|---|---|---|
|**目的**|为短期买卖、做市或对冲而持有|为持有至到期、放贷、资产负债管理|
|**会计处理**|按公允价值 **每日** 重新估值 (mark-to-market)|多数资产按摊余成本或减值后成本|
|**主要风险类别**|**[[Market Risk|市场风险]]** 为主（利率、汇率、信用利差波动）|**[[Credit Risk|信用风险]]** 为主（违约 & 迁移）+ 利率重定价风险|
|**资本计量**|巴塞尔市场风险框架：10 天、99% VaR × 乘数＋特定风险 (或后续 sVaR/IRC 等)|巴塞尔信用风险框架：• 标准法：[[Risk-Weighted Assets|RWA]]=100%×面值RWA = 100\%\times\text{面值}• IRB：一年、99.9% VaR（[[PD|PD]]/[[LGD|LGD]]/[[EAD|EAD]] 模型）|
|**典型资本水平**|早期 (<[[Basel Accords|Basel]] 2.5) 往往 **低于** 银行账户；[[Basel Accords|Basel]] 2.5 起因 IRC 等显著提高|通常高于或与交易账户相当|
<!-- bilingual-en:start -->
| Dimension | Trading book | Banking book |
| --- | --- | --- |
| **Purpose** | Positions held for trading, market-making, or trading-related hedging | Lending, longer-term holdings, and asset–liability management |
| **Valuation and accounting** | Generally fair-valued and marked frequently | Accounting depends on classification; many loans are measured at amortised cost subject to impairment |
| **Dominant regulatory risk** | **[[Market Risk|Market risk]]**, including rates, FX, equities, spreads, and options | **[[Credit Risk|Credit risk]]**, plus banking-book interest-rate and other risks |
| **Historical capital method used in the exercise** | Ten-day 99% VaR × multiplier plus specific risk, before later sVaR, IRC, and FRTB reforms | Standardised credit [[Risk-Weighted Assets|RWA]] or IRB using [[PD|PD]], [[LGD|LGD]], [[EAD|EAD]], and maturity |
| **Capital comparison** | Under early rules, some positions attracted less capital when classified in the trading book; later reforms reduced this arbitrage | Often more capital under the simplified example, but the answer depends on borrower, instrument, collateral, and applicable framework |
<!-- bilingual-en:end -->

 二、题设情形与资本影响
<!-- bilingual-en:start -->
II. Applying the Assumptions in the Question
<!-- bilingual-en:end -->

> **现状**：银行账簿持有一笔 **贷款**，面值 \$10 000 000。  
> **变更**：贷款到期，客户改为将 **自家债券** 出售给银行。
> <!-- bilingual-en:start -->
> **Current position:** a **loan** with face value USD 10 million is held in the banking book.
> **Proposed change:** when the loan matures, the customer sells the bank its own **bond**.
> <!-- bilingual-en:end -->

 1. 贷款（银行账户）下的资本
<!-- bilingual-en:start -->
1. Illustrative capital for the banking-book loan
<!-- bilingual-en:end -->

- 假设客户为普通企业，标准法风险权重 **100 %**  
- [[Risk-Weighted Assets|风险加权资产]]  
  $$\text{[[Risk-Weighted Assets|RWA]]}_{\text{loan}} = 10\,000\,000 \times 100\% = 10\,000\,000$$
- 资本要求  
  $$K_{\text{loan}} = 8\% \times \text{[[Risk-Weighted Assets|RWA]]} = 0.08 \times 10\,000\,000 = \textbf{\$800\,000}$$
<!-- bilingual-en:start -->
- Assume an ordinary corporate borrower and a 100% standardised risk weight.
- [[Risk-Weighted Assets|RWA]] = USD 10 million.
- At the historical 8% total-capital ratio, required capital is USD 800,000.
<!-- bilingual-en:end -->

 2. 债券（若列入交易账户）下的资本
<!-- bilingual-en:start -->
2. Illustrative capital if the bond qualifies for the trading book
<!-- bilingual-en:end -->

- 采用市场风险 VaR 法（[[Basel Accords|Basel]] II，未计入 [[Basel Accords|Basel]] 2.5 的 sVaR/IRC）。  
- 假设该债券 10 天、99 % VaR ≈ **1 %** 名义本金  
  $$\text{VaR} = 10\,000\,000 \times 1\% = 100\,000$$
- 监管乘数（最低）  
  $$m = 3$$
- 资本要求  
  $$K_{\text{trading}} = m \times \text{VaR} = 3 \times 100\,000 = \textbf{\$300\,000}$$
<!-- bilingual-en:start -->
- Use the historical market-risk VaR method under [[Basel Accords|Basel]] II, before the [[Basel Accords|Basel]] 2.5 sVaR and IRC additions.
- Assume ten-day 99% VaR equals **1%** of notional, or USD 100,000.
- Use the minimum multiplier of 3.
- The illustrative market-risk charge is therefore USD 300,000.
<!-- bilingual-en:end -->

 3. 资本变化
<!-- bilingual-en:start -->
3. Change in capital
<!-- bilingual-en:end -->

$$\[[Delta|Delta]] K = K_{\text{trading}} - K_{\text{loan}} = 300\,000 - 800\,000 = -\$500\,000$$

> **结果**：若贷款到期改为持有债券并列入交易账户，监管资本 **减少约 \$500 000**（基于 VaR = 1 % 的假设）。  
> 若采用 [[Basel Accords|Basel]] 2.5（需计提 sVaR/IRC），交易账户资本可能回升甚至超过 \$800 000，监管套利空间被显著压缩。
> <!-- bilingual-en:start -->
> **Result under the exercise's assumptions:** capital falls by about **USD 500,000** if the loan is replaced by a bond that genuinely qualifies for the trading book and has VaR equal to 1% of notional. This is not an automatic consequence of issuing a bond: trading-book classification depends on trading intent, governance, and regulatory eligibility. Under [[Basel Accords|Basel]] 2.5, sVaR and IRC could raise the trading-book charge to or above USD 800,000, narrowing the arbitrage.
> <!-- bilingual-en:end -->

 三、关键步骤回顾
<!-- bilingual-en:start -->
III. Key Steps
<!-- bilingual-en:end -->

1. 明确两账簿定义与适用风险框架；  
2. 贷款按信用风险标准法计算 $$\text{[[Risk-Weighted Assets|RWA]]}$$，资本 = 8 %；  
3. 债券列入交易账簿，用 $$K = m \times \text{VaR}$$ 计市场风险资本；  
4. 比较两者，得出资本增减方向与原因。
<!-- bilingual-en:start -->
1. Define the two books and identify the applicable risk framework.
2. Calculate the loan under the standardised credit-risk approach: capital = 8% of [[Risk-Weighted Assets|RWA]].
3. Under the exercise's historical trading-book assumption, calculate the bond charge as $K=m\times VaR$.
4. Compare the two results and explain both the direction and the assumptions behind the change.
<!-- bilingual-en:end -->

## 15.12 
 
>[!question] 
>银行有时会利用“监管套利”来降低资本要求。请说明监管套利的含义。  
><!-- bilingual-en:start -->
>Banks sometimes use “regulatory arbitrage” to reduce capital requirements. Explain what regulatory arbitrage means.
><!-- bilingual-en:end -->

 答题结构
<!-- bilingual-en:start -->
Answer structure
<!-- bilingual-en:end -->

1. **定义**
<!-- bilingual-en:start -->
1. **Definition**
<!-- bilingual-en:end -->
    
    - **监管套利**：银行通过**法律允许**但**不改变真实经济风险**的手段，利用监管规则的缺陷或差异，使名义上的风险加权资产 ([[Risk-Weighted Assets|RWA]]) 或表内资本需求降低，从而减少需持有的监管资本。
<!-- bilingual-en:start -->
- **Regulatory arbitrage** occurs when a bank uses legally available differences, gaps, or measurement choices in regulation to reduce reported [[Risk-Weighted Assets|RWA]] or required capital without a commensurate reduction in the underlying economic risk.
<!-- bilingual-en:end -->
        
    - 本质：**经济资本 ≠ 监管资本** 之间的缺口被人为放大。
<!-- bilingual-en:start -->
- In essence, it deliberately widens the gap between **economic risk and regulatory capital**.
<!-- bilingual-en:end -->
        
2. **常见手段**
<!-- bilingual-en:start -->
2. **Common techniques**
<!-- bilingual-en:end -->
    
    1. **资产重分类**
<!-- bilingual-en:start -->
1. **Asset reclassification**
<!-- bilingual-en:end -->
        
        - 将高风险贷款**转入交易账户**，按市场风险 VaR 计提资本（早期 VaR 资本往往低于信用风险资本）。
<!-- bilingual-en:start -->
- Reclassify positions into a regulatory book or category with a lower measured charge, such as using an early trading-book VaR charge where it is lower than banking-book credit capital.
<!-- bilingual-en:end -->
            
    2. **证券化与出售**
<!-- bilingual-en:start -->
2. **Securitisation and sale**
<!-- bilingual-en:end -->
        
        - 打包次级贷款发行 MBS/CDO，将风险转移至投资者，仅保留表外信用支持，[[Risk-Weighted Assets|RWA]] 显著下降。
<!-- bilingual-en:start -->
- Package loans into MBS or CDO structures, transfer them to investors, and retain only selected off-balance-sheet support, thereby reducing reported [[Risk-Weighted Assets|RWA]] even when residual risk remains.
<!-- bilingual-en:end -->
            
    3. **利用风险权重差异**
<!-- bilingual-en:start -->
3. **Exploit differences in risk weights**
<!-- bilingual-en:end -->
        
        - 购买低权重（如政府/AAA 证券）却带有市场利差风险的资产；或者通过 **信用转移 (CDS)** 将对手方换成低权重机构。
<!-- bilingual-en:start -->
- Hold low-risk-weight assets that still carry material spread or concentration risk, or use a credit-risk transfer such as a CDS to substitute a lower-weighted protection provider.
<!-- bilingual-en:end -->
            
    4. **表外安排**
<!-- bilingual-en:start -->
4. **Off-balance-sheet structures**
<!-- bilingual-en:end -->
        
        - 设立 SPV、ABCPC、SIV 等，将资产移出资产负债表，仅承担流动性担保或回购承诺。
<!-- bilingual-en:start -->
- Move assets into SPVs, ABCP conduits, or SIVs while retaining liquidity facilities, guarantees, or repurchase commitments whose economic risk is not fully reflected in the capital charge.
<!-- bilingual-en:end -->
            
    5. **模型假设优化**（IRB/AMA）
<!-- bilingual-en:start -->
5. **Optimise internal-model assumptions**
<!-- bilingual-en:end -->
        
        - 调整 [[PD|PD]]、[[LGD|LGD]]、资产相关系数估计，使内部模型输出的资本需求低于实际风险。
<!-- bilingual-en:start -->
- Choose aggressive estimates of [[PD|PD]], [[LGD|LGD]], correlations, or other parameters so that an IRB or AMA model produces less capital than the true risk would justify.
<!-- bilingual-en:end -->
            
3. **影响与风险**
<!-- bilingual-en:start -->
3. **Effects and risks**
<!-- bilingual-en:end -->
    
    - **降低监管资本缓冲** → 在压力情景下更易资本不足；
<!-- bilingual-en:start -->
- **A smaller regulatory buffer** makes capital shortfalls more likely under stress.
<!-- bilingual-en:end -->
        
    - **信息不透明** → 市场难以评估真实敞口，易引发系统性风险；
<!-- bilingual-en:start -->
- **Opacity** makes it harder for markets and supervisors to assess the bank's true exposure and can amplify systemic risk.
<!-- bilingual-en:end -->
        
    - **增加监管复杂度** → 促使监管框架不断升级（[[Basel Accords|Basel]] II→2.5→III）。
<!-- bilingual-en:start -->
- **Greater regulatory complexity** encourages successive reforms, including [[Basel Accords|Basel]] II, Basel 2.5, and Basel III.
<!-- bilingual-en:end -->
        
4. **监管回应**
<!-- bilingual-en:start -->
4. **Regulatory responses**
<!-- bilingual-en:end -->
    
    - 引入 **[[Stress Testing|压力测试]]、sVaR、IRC、CRM、杠杆率与流动性指标**；
<!-- bilingual-en:start -->
- Add **[[Stress Testing|stress testing]], stressed VaR, IRC, comprehensive risk measures, leverage ratios, and liquidity requirements** where relevant.
<!-- bilingual-en:end -->
        
    - 提高对结构化产品与表外项目的资本要求，减少模型自由度。
 结论
<!-- bilingual-en:start -->
- Increase capital for structured and off-balance-sheet exposures and constrain unsupported model discretion.

**Conclusion**
<!-- bilingual-en:end -->

监管套利是银行利用监管规则缺口主动“搬动资产”或“重塑风险表象”以最小化资本要求的行为。虽然短期提高资本效率，但会积累隐性风险并削弱金融体系稳健性，因此成为巴塞尔协议迭代升级的主要驱动力之一。
<!-- bilingual-en:start -->
Regulatory arbitrage minimises measured capital by moving assets or reshaping their regulatory appearance without proportionately reducing economic risk. It may improve short-run measured capital efficiency, but it can accumulate hidden exposures and weaken financial resilience, making it an important driver of successive Basel reforms.
<!-- bilingual-en:end -->

## 15.17 

>[!question] 
某银行资产包含 2 亿美元的零售贷款（非住房抵押贷款），[[PD|PD]] = 1%，[[LGD|LGD]] = 70%。  
  根据《[[Basel Accords|巴塞尔协议]] II》IRB 法：  
  ① 计算风险加权资产；  
  ② 给出所需的第一类资本（Tier 1）和第二类资本（Tier 2）。  
<!-- bilingual-en:start -->
A bank has USD 200 million of non-mortgage retail loans with [[PD|PD]] = 1% and [[LGD|LGD]] = 70%. Under the [[Basel Accords|Basel]] II IRB approach: (1) calculate [[Risk-Weighted Assets|risk-weighted assets]]; and (2) state the historical minimum Tier 1 and total-capital amounts, together with the amount that Tier 2 could supply if Tier 1 is held at its minimum.
<!-- bilingual-en:end -->

 已知数据  
- **[[EAD|EAD]]（敞口余额）** = \$200 000 000  
- **[[PD|PD]]** = 1% = 0.01  
- **[[LGD|LGD]]** = 70% = 0.70  
- 资产类别：**“其他零售”**（非住房抵押贷款）  
- 使用《巴塞尔Ⅱ》IRB 基础法（零售资产无到期调整）  
<!-- bilingual-en:start -->
Given:
- **[[EAD|EAD]]**, or exposure at default = USD 200 million
- **[[PD|PD]]** = 1% = 0.01
- **[[LGD|LGD]]** = 70% = 0.70
- Asset class: **other retail**, not residential mortgages
- Use the Basel II retail IRB formula, which has no corporate maturity adjustment
<!-- bilingual-en:end -->

 ① 计算风险加权资产 ([[Risk-Weighted Assets|RWA]])
<!-- bilingual-en:start -->
① Calculate [[Risk-Weighted Assets|risk-weighted assets]] ([[Risk-Weighted Assets|RWA]])
<!-- bilingual-en:end -->

1. **[[Correlation Coefficient|相关系数]] \(R\)**  
  $$
   R = 0.03\!\left(\frac{1-e^{-35\text{[[PD|PD]]}}}{1-e^{-35}}\right)
     + 0.16\!\left[1-\frac{1-e^{-35\text{[[PD|PD]]}}}{1-e^{-35}}\right]
     \approx 0.1216
  $$
2. **资本系数 \(K\)**  
  $$
   K = \text{[[LGD|LGD]]}\Bigl[
       N\!\Bigl(\frac{N^{-1}(\text{[[PD|PD]]})+\sqrt{R}\,N^{-1}(0.999)}
       {\sqrt{1-R}}\Bigr) - \text{[[PD|PD]]}
       \Bigr]
     \approx 0.0573
  $$
3. **[[Risk-Weighted Assets|风险加权资产]]**  
$$
   \text{[[Risk-Weighted Assets|RWA]]} = 12.5 \times K \times \text{[[EAD|EAD]]}
              = 12.5 \times 0.0573 \times \$200\text{m}
              \approx \$143.2\text{m}
  $$
<!-- bilingual-en:start -->
1. **[[Correlation Coefficient|Asset correlation]] $R$**
$R = 0.03\!\left(\frac{1-e^{-35PD}}{1-e^{-35}}\right)+0.16\!\left[1-\frac{1-e^{-35PD}}{1-e^{-35}}\right]\approx0.1216$.
2. **Capital coefficient $K$**
$K=LGD\left[N\!\left(\frac{N^{-1}(PD)+\sqrt{R}N^{-1}(0.999)}{\sqrt{1-R}}\right)-PD\right]\approx0.0573$.
3. **[[Risk-Weighted Assets|Risk-weighted assets]]**
$RWA=12.5\times K\times EAD=12.5\times0.0573\times\$200\text{m}\approx\$143.2\text{m}$.
<!-- bilingual-en:end -->

 ② 计算所需资本
<!-- bilingual-en:start -->
② Calculate the historical capital minima
<!-- bilingual-en:end -->

| 项目              | 计算公式                       | 金额          |
| --------------- | -------------------------- | ----------- |
| **Tier 1 最低资本** | $(0.04 \times \text{[[Risk-Weighted Assets|RWA]]})$ | ≈ \$5.73 m  |
| **Tier 2（上限）**  | 使总资本达到 8%                  | ≈ \$5.73 m  |
| **合计总资本**       | $(0.08 \times \text{[[Risk-Weighted Assets|RWA]]})$ | ≈ \$11.45 m |
<!-- bilingual-en:start -->
| Item | Calculation | Amount |
| --- | --- | ---: |
| **Minimum Tier 1 capital** | $0.04\times[[Risk-Weighted Assets|RWA]]$ | Approximately USD 5.73 million |
| **Possible Tier 2 contribution if Tier 1 is held at 4%** | Amount needed to reach the 8% total minimum, subject to eligibility limits | Approximately USD 5.73 million |
| **Minimum total capital** | $0.08\times[[Risk-Weighted Assets|RWA]]$ | Approximately USD 11.45 million |
<!-- bilingual-en:end -->

**结论**  
- **[[Risk-Weighted Assets|RWA]] ≈ \$143 million**  
- **Tier 1 资本 ≈ \$5.73 million**  
- **Tier 2 资本 ≈ \$5.73 million**（使总资本达到 8%，即 \$11.45 million）
<!-- bilingual-en:start -->
**Conclusion**
- **[[Risk-Weighted Assets|RWA]] ≈ USD 143 million**
- **Minimum Tier 1 capital ≈ USD 5.73 million**
- **Minimum total capital ≈ USD 11.45 million**; Tier 2 could provide approximately USD 5.73 million if Tier 1 is held exactly at the historical 4% minimum and all eligibility conditions are met.
<!-- bilingual-en:end -->

> 关键步骤：计算 \(R\) → 计算 \(K\) → 折算 [[Risk-Weighted Assets|RWA]] → 按 4% / 8% 提取 Tier 1 与 Tier 2。
> <!-- bilingual-en:start -->
> Key steps: calculate $R$ → calculate $K$ → convert to [[Risk-Weighted Assets|RWA]] → apply the historical 4% Tier 1 and 8% total-capital minima.
> <!-- bilingual-en:end -->

## 15.21 
 
>[!question] 
某银行与一家 **AA 级** 企业存在以下交易：  
  (a) 两年期利率互换，名义本金 1 亿美元，当前价值 300 万美元；  
  (b) 9 个月期外汇远期合约，名义本金 1.5 亿美元，当前价值 –500 万美元；  
  (c) 6 个月期黄金期权（多头），名义本金 5 000 万美元，当前价值 700 万美元。  
  在 **无净额** 情况下，按《[[Basel Accords|巴塞尔协议]] I》计算资本金要求；若适用 1995 年净额修正案，资本金要求有何变化？  
  在《[[Basel Accords|巴塞尔协议]] II》**标准法**下的资本金数量又是多少？  
<!-- bilingual-en:start -->
A bank has the following transactions with one **AA-rated corporate counterparty**: (a) a two-year interest-rate swap with notional USD 100 million and current value USD 3 million; (b) a nine-month FX forward with notional USD 150 million and current value negative USD 5 million; and (c) a six-month long gold option with notional USD 50 million and current value USD 0.7 million. Calculate capital under historical [[Basel Accords|Basel]] I without netting, then with the 1995 netting amendment, and under the [[Basel Accords|Basel]] II standardised counterparty risk weight. Gold belongs with foreign exchange and gold in the historical add-on table, not with “other commodities.”
<!-- bilingual-en:end -->

 已知信息  
- 对手方：**AA 级企业**  
- 风险权重：[[Basel Accords|Basel]] I → 100%；[[Basel Accords|Basel]] II-SA → 20%（AAA/AA 档）  
- 1995 净额修正案公式  
  $$
  \text{Add-on}_{\text{net}} = 0.4\sum A_i + 0.6\,\text{NGR}\,\sum A_i,
  \qquad
  \text{NGR}= \frac{\text{CE}_{\text{net}}}{\sum \text{正市价}}
  $$
<!-- bilingual-en:start -->
Given:
- Counterparty: **AA-rated corporate**
- Historical risk weight: [[Basel Accords|Basel]] I → 100%; [[Basel Accords|Basel]] II standardised approach → 20% under the assumption used here
- Netting amendment: $\text{Add-on}_{net}=0.4\sum A_i+0.6\,NGR\sum A_i$, where $NGR=CE_{net}/\sum\text{positive market values}$
<!-- bilingual-en:end -->

| 合约 | 期限 | 名义本金 | 市价 CE (正值) | [[Basel Accords|Basel]] I add-on 系数 | PFE \(A\) |
|------|------|-----------|---------------|--------------------|-----------|
| (a) 利率互换 | 2 年 | \$100 m | **\$3 m** | 0.5 % | **\$0.5 m** |
| (b) 外汇远期 | 9 月 | \$150 m | 0 (–\$5 m) | 1 %  | **\$1.5 m** |
| (c) 黄金期权 | 6 月 | \$50 m | **\$0.7 m** | 10 % | **\$5 m** |
<!-- bilingual-en:start -->
| Contract | Maturity | Notional | Positive CE | Historical add-on | PFE $A$ |
| --- | --- | ---: | ---: | ---: | ---: |
| (a) Interest-rate swap | 2 years | USD 100 million | **USD 3 million** | 0.5% | **USD 0.5 million** |
| (b) FX forward | 9 months | USD 150 million | 0, because -USD 5 million is floored at zero | 1% | **USD 1.5 million** |
| (c) Gold option | 6 months | USD 50 million | **USD 0.7 million** | **1% for FX and gold**, correcting the source's use of the 10% other-commodity factor | **USD 0.5 million** |
<!-- bilingual-en:end -->

 ① [[Basel Accords|Basel]] I（无净额）
<!-- bilingual-en:start -->
① Historical [[Basel Accords|Basel]] I without netting
<!-- bilingual-en:end -->

$$
\text{CEA}_i = \text{CE}_i + A_i
$$

| 合约 | CEA (USD m) | [[Risk-Weighted Assets|RWA]] = 100 % × CEA | 资本 = 8 % × [[Risk-Weighted Assets|RWA]] |
|------|-------------|--------------------|-----------------|
| (a) | 3 + 0.5 = **3.5** | 3.5 | **0.28** |
| (b) | 0 + 1.5 = **1.5** | 1.5 | **0.12** |
| (c) | 0.7 + 5 = **5.7** | 5.7 | **0.456** |
<!-- bilingual-en:start -->
| Contract | CEA (USD m) | [[Risk-Weighted Assets|RWA]] at 100% | Capital at 8% |
| --- | ---: | ---: | ---: |
| (a) | $3+0.5=\mathbf{3.5}$ | 3.5 | **0.280** |
| (b) | $0+1.5=\mathbf{1.5}$ | 1.5 | **0.120** |
| (c) | $0.7+0.5=\mathbf{1.2}$ | 1.2 | **0.096** |
<!-- bilingual-en:end -->

> **[[Basel Accords|Basel]] I 总资本 = 0.28 + 0.12 + 0.456 ≈ \$0.86 m**
> <!-- bilingual-en:start -->
> **Corrected historical [[Basel Accords|Basel]] I total capital = USD 0.496 million.**
> <!-- bilingual-en:end -->

 ② [[Basel Accords|Basel]] I（1995 净额修正案）
<!-- bilingual-en:start -->
② Historical [[Basel Accords|Basel]] I with the 1995 netting amendment
<!-- bilingual-en:end -->

- 正市价总额 = 3 m + 0.7 m = **3.7 m**  
- **净市价 CE\_net = 3.7 m – 5 m = –1.3 m ⇒ 0**  
- **NGR = 0 / 3.7 m = 0**
<!-- bilingual-en:start -->
- Gross positive market value = USD 3.0 million + USD 0.7 million = **USD 3.7 million**.
- **Net current exposure = max(3.7 − 5.0, 0) = 0.**
- **NGR = 0 / 3.7 = 0.**
- Gross add-on = 0.5 + 1.5 + 0.5 = USD 2.5 million, so net add-on = $0.4\times2.5=\mathbf{1.0}$ million.
- Net CEA = USD 1.0 million and Basel I capital = **USD 0.080 million**.
<!-- bilingual-en:end -->

$$
\text{Add-on}_{\text{net}} = 0.4 \times 7\,\text{m} = 2.8\,\text{m}
$$
$$
\text{CEA}_{\text{net}} = 0 + 2.8 = 2.8\,\text{m}
$$
$$
\text{资本} = 8\% \times 2.8 = \text{\$0.224 m}
$$

> **净额后资本从 \$0.86 m 降至 \$0.22 m**
> <!-- bilingual-en:start -->
> **With the gold add-on corrected, recognised netting reduces capital from USD 0.496 million to USD 0.080 million.**
> <!-- bilingual-en:end -->

 ③ [[Basel Accords|Basel]] II – 标准法 (SA)
<!-- bilingual-en:start -->
③ [[Basel Accords|Basel]] II standardised approach under the assumed 20% counterparty weight
<!-- bilingual-en:end -->

 (a) 无净额  
$$
\text{[[Risk-Weighted Assets|RWA]]} = 20\% \times 10.7 = 2.14\,\text{m}, 
\qquad
\text{资本} = 8\% \times 2.14 = \$0.171\,\text{m}
$$
<!-- bilingual-en:start -->
(a) Without netting: CEA is USD 6.2 million, so [[Risk-Weighted Assets|RWA]] are $20\%\times6.2=1.24$ million and capital is **USD 0.0992 million**.
<!-- bilingual-en:end -->

 (b) 承认净额  
$$
\text{[[Risk-Weighted Assets|RWA]]} = 20\% \times 2.8 = 0.56\,\text{m}, 
\qquad
\text{资本} = 8\% \times 0.56 = \$0.045\,\text{m}
$$
<!-- bilingual-en:start -->
(b) With recognised netting: CEA is USD 1.0 million, so [[Risk-Weighted Assets|RWA]] are $20\%\times1.0=0.20$ million and capital is **USD 0.016 million**.
<!-- bilingual-en:end -->

> **[[Basel Accords|Basel]] II-SA 资本要求**  
> - **无净额：≈ \$0.17 m**  
> - **净额：≈ \$0.05 m**
> <!-- bilingual-en:start -->
> **Corrected [[Basel Accords|Basel]] II standardised capital requirement**
> - **Without netting: approximately USD 0.10 million**
> - **With recognised netting: approximately USD 0.016 million**
> <!-- bilingual-en:end -->

 关键步骤  
1. 识别正市价 → 负值计 0。  
2. 取对应产品/期限 add-on 系数算 PFE。  
3. [[Basel Accords|Basel]] I：CEA = CE + PFE → [[Risk-Weighted Assets|RWA]] (100 %) → 8 % 资本。  
4. 1995 净额：公式折减 PFE，CE\_net 为负取 0。  
5. [[Basel Accords|Basel]] II-SA：CEA 同 [[Basel Accords|Basel]] I，但用对手方 20 % 风险权重。
<!-- bilingual-en:start -->
Key steps:
1. Keep positive market values and floor negatives at zero.
2. Apply the correct product and maturity add-on; gold uses the FX-and-gold row.
3. Under Basel I, add CE and PFE, apply the 100% risk weight, and then apply 8% capital.
4. Under recognised netting, reduce PFE through the formula and floor net CE at zero.
5. Under the Basel II standardised assumption, use the same corrected CEA with the 20% counterparty risk weight.
<!-- bilingual-en:end -->

## 15.22 
 
>[!question] 
某银行持有 5 亿美元 **BBB 级企业贷款**；[[PD|PD]] = 0.3%，平均期限 3 年，[[LGD|LGD]] = 60%。  
  (1) 计算《[[Basel Accords|巴塞尔协议]] II》IRB 法下的信用风险 **[[Risk-Weighted Assets|风险加权资产]]**；  
  (2) 进一步计算所需 Tier 1 与 Tier 2 资本；  
  (3) 将结果与《[[Basel Accords|巴塞尔协议]] II》**标准法**及《[[Basel Accords|巴塞尔协议]] I》的资本金进行比较。  
<!-- bilingual-en:start -->
A bank holds USD 500 million of **BBB-rated corporate loans**, with [[PD|PD]] = 0.3%, average maturity three years, and [[LGD|LGD]] = 60%. (1) Calculate credit-risk [[Risk-Weighted Assets|risk-weighted assets]] under the [[Basel Accords|Basel]] II corporate IRB formula; (2) calculate the historical Tier 1 and total-capital minima; and (3) compare the result with the Basel II standardised approach and Basel I.
<!-- bilingual-en:end -->

 已知数据  
- **[[EAD|EAD]]** = \$500 000 000  
- **[[PD|PD]]** = 0.3 % = 0.003  
- **[[LGD|LGD]]** = 60 % = 0.60  
- **M** = 3 年（平均期限）  
- 资产类别：企业贷款（BBB 级，采用《巴塞尔Ⅱ》**IRB 基础法**公式）  
<!-- bilingual-en:start -->
Given:
- **[[EAD|EAD]]** = USD 500 million
- **[[PD|PD]]** = 0.3% = 0.003
- **[[LGD|LGD]]** = 60% = 0.60
- **M** = 3 years
- Asset class: BBB-rated corporate loan under the Basel II foundation-IRB corporate formula
<!-- bilingual-en:end -->

 ①  IRB 法下风险加权资产 ([[Risk-Weighted Assets|RWA]])
<!-- bilingual-en:start -->
① [[Risk-Weighted Assets|Risk-weighted assets]] under IRB
<!-- bilingual-en:end -->

1. **[[Correlation Coefficient|相关系数]] \(R\)**  
   $$
   R = 0.12\!\Bigl(\tfrac{1-e^{-50PD}}{1-e^{-50}}\Bigr)
       + 0.24\!\Bigl[1-\tfrac{1-e^{-50PD}}{1-e^{-50}}\Bigr]
     \approx 0.2233
   $$
<!-- bilingual-en:start -->
1. **[[Correlation Coefficient|Asset correlation]] $R$**
$R=0.12\left(\frac{1-e^{-50PD}}{1-e^{-50}}\right)+0.24\left[1-\frac{1-e^{-50PD}}{1-e^{-50}}\right]\approx0.2233$.
<!-- bilingual-en:end -->

2. **无到期资本系数 \(K_0\)**  
   $$
   z = \frac{N^{-1}([[PD|PD]])+\sqrt{R}\,N^{-1}(0.999)}{\sqrt{1-R}}
      \;\;(\text{取 }N^{-1}(0.999)=3.0902) 
   $$
   $$
   N(z) \approx 0.0725 \quad\Longrightarrow\quad
   K_0 = [[LGD|LGD]]\,[N(z)-[[PD|PD]]] \approx 0.60\,(0.0725-0.003)=0.0417
   $$
<!-- bilingual-en:start -->
2. **Capital coefficient before the maturity adjustment, $K_0$**
$z=\frac{N^{-1}(PD)+\sqrt{R}N^{-1}(0.999)}{\sqrt{1-R}}$, using $N^{-1}(0.999)=3.0902$.
$N(z)\approx0.0725$, so $K_0=LGD[N(z)-PD]\approx0.60(0.0725-0.003)=0.0417$.
<!-- bilingual-en:end -->

3. **到期调整 (M = 3)**  
   $$
   b = \bigl(0.11852-0.05478\ln [[PD|PD]]\bigr)^2 \approx 0.1907
   $$
   $$
   \text{MA} = \frac{1+(M-2.5)\,b}{1-1.5\,b}
             = \frac{1+0.5b}{1-1.5b}\approx1.534
   $$
   $$
   K = K_0 \times \text{MA} \approx 0.0417 \times 1.534 = 0.0640
   $$
<!-- bilingual-en:start -->
3. **Maturity adjustment for $M=3$**
$b=(0.11852-0.05478\ln PD)^2\approx0.1907$.
$MA=\frac{1+(M-2.5)b}{1-1.5b}=\frac{1+0.5b}{1-1.5b}\approx1.534$.
Therefore $K=K_0\times MA\approx0.0417\times1.534=0.0640$.
<!-- bilingual-en:end -->

4. **[[Risk-Weighted Assets|风险加权资产]]**  
   $$
   \text{[[Risk-Weighted Assets|RWA]]}_{\text{IRB}}
     = 12.5 \times K \times [[EAD|EAD]]
     = 12.5 \times 0.0640 \times \$500\text{m}
     \approx \$400\text{ m}
   $$
<!-- bilingual-en:start -->
4. **[[Risk-Weighted Assets|Risk-weighted assets]]**
$RWA_{IRB}=12.5\times K\times EAD=12.5\times0.0640\times\$500\text{m}\approx\$400\text{m}$.
<!-- bilingual-en:end -->

 ②  所需资本  
<!-- bilingual-en:start -->
② Historical capital minima
<!-- bilingual-en:end -->

| 资本项目 | 系数 | IRB [[Risk-Weighted Assets|RWA]] | 资本金额 |
|----------|------|---------|-----------|
| **Tier 1** | 4 % | \$400 m | **\$16 m** |
| **Tier 2** | (补足至 8 %) | 同左 | **\$16 m** |
| **总资本** | 8 % | \$400 m | **\$32 m** |
<!-- bilingual-en:start -->
| Capital item | Ratio | IRB [[Risk-Weighted Assets|RWA]] | Amount |
| --- | ---: | ---: | ---: |
| **Minimum Tier 1** | 4% | USD 400 million | **USD 16 million** |
| **Possible Tier 2 contribution if Tier 1 is held at 4%** | Amount needed to reach 8%, subject to eligibility | Same RWA | **USD 16 million** |
| **Minimum total capital** | 8% | USD 400 million | **USD 32 million** |
<!-- bilingual-en:end -->

 ③  与其他方法比较
<!-- bilingual-en:start -->
③ Comparison with other approaches
<!-- bilingual-en:end -->

| 方法 | 风险权重 | [[Risk-Weighted Assets|RWA]] (USD m) | 总资本 = 8 %×[[Risk-Weighted Assets|RWA]] |
|------|----------|-------------|------------------|
| **[[Basel Accords|Basel]] I** | 100 % | 500 | \$40 m |
| **[[Basel Accords|Basel]] II – 标准法** (BBB) | 100 % | 500 | \$40 m |
| **[[Basel Accords|Basel]] II – IRB 法** |  ≈ 80 % | **400** | **\$32 m** |
<!-- bilingual-en:start -->
| Approach | Risk weight or equivalent | [[Risk-Weighted Assets|RWA]] (USD m) | Total capital at 8% |
| --- | ---: | ---: | ---: |
| **[[Basel Accords|Basel]] I** | 100% | 500 | USD 40 million |
| **[[Basel Accords|Basel]] II standardised approach**, BBB | 100% under the exercise's table | 500 | USD 40 million |
| **[[Basel Accords|Basel]] II IRB** | Approximately 80% equivalent | **400** | **USD 32 million** |
<!-- bilingual-en:end -->

> **结果**：IRB 法 [[Risk-Weighted Assets|RWA]] 较标准法/巴塞尔 I 下降 \$100 m（–20 %），资本需求降低 \$8 m；Tier 1、Tier 2 各减少 \$4 m。  
> <!-- bilingual-en:start -->
> **Result:** IRB produces USD 400 million of [[Risk-Weighted Assets|RWA]], USD 100 million or 20% below the two simplified standardised comparisons. The total-capital minimum falls by USD 8 million; if Tier 1 is held at the historical minimum, Tier 1 and the Tier 2 contribution each fall by USD 4 million.
> <!-- bilingual-en:end -->

**关键步骤**：计算 \(R\) → 求 \(K_0\) → 到期调整 → 折算 [[Risk-Weighted Assets|RWA]] → 按 4 % / 8 % 计算 Tier 1 与总资本，再与其他方法对比。
<!-- bilingual-en:start -->
**Key steps:** calculate $R$ → obtain $K_0$ → apply the maturity adjustment → convert to [[Risk-Weighted Assets|RWA]] → apply the historical 4% Tier 1 and 8% total-capital minima → compare methods.
<!-- bilingual-en:end -->
