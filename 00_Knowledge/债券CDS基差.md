---
aliases:
  - "债券–CDS 基差必须固定口径；按 CDS 利差减同期限 ASW 利差定义时，正值表示 CDS 更宽，融资、流动性与合约错配可使它持续偏离零"
  - "Bond-CDS basis under the CDS-minus-ASW convention"
  - "CDS-ASW basis"
student_os: knowledge-atom
atom_id: RM-CDS-003
atom_set: cds-pricing-and-basis
atom_type: measurement-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[CDS两腿定价]]"
  - "[[信用利差]]"
  - "[[嵌入期权的收益率比较边界]]"
related:
  - "[[CDS信用事件与结算]]"
  - "[[负基差交易]]"
part_of:
  - "[[CDS定价与基差.canvas|CDS定价与基差]]"
---

# 债券–CDS 基差必须固定口径；按 CDS 利差减同期限 ASW 利差定义时，正值表示 CDS 更宽，融资、流动性与合约错配可使它持续偏离零
<!-- bilingual-en:start -->
*The bond–CDS basis requires a fixed convention: CDS spread minus same-maturity ASW spread*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 本卡固定采用“CDS 利差减同期限资产互换利差（ASW）”的口径。正基差表示 CDS 更宽，负基差表示债券的 ASW 更宽。零基差只是在这一度量下两项报价相等，并不表示现金债与 CDS 的现金流、融资、流动性、对手方风险和信用事件结算完全相同。若把 ASW 换成 Z-spread 或相对国债利差，必须改名并重新解释数值，不能沿用本卡的符号含义。

## 定义先锁定比较对象
<!-- bilingual-en:start -->
*Fix the compared objects before interpreting the sign*
<!-- bilingual-en:end -->

在观察日 \(t\)，对到期日为 \(T\) 的同一信用，定义

\[
B_{\mathrm{ASW}}(t,T)
=s_{\mathrm{CDS}}(t,T)-s_{\mathrm{ASW}}(t,T).
\]

- \(s_{\mathrm{CDS}}\) 是与当前 CDS 价格一致的**平价等价运行利差**。标准合约若采用固定票息加期初 upfront，不能把固定票息直接代入；须按 [[CDS两腿定价]] 把 upfront 与两腿现金流还原成可比的平价利差。
- \(s_{\mathrm{ASW}}\) 是由该债券价格隐含的同期限资产互换利差。资产互换把固定息票债与利率互换组合成“浮动参考利率加一段利差”的资产；使组合按约定价格口径成立的那段利差才是 ASW。
- \(B_{\mathrm{ASW}}>0\)：CDS 保护价格按利差表示更宽；\(B_{\mathrm{ASW}}<0\)：现金债的 ASW 更宽。这里的“更宽”是报价比较，不等于哪一边必然错价。

“同一信用、同一期限”仍不够。可比报价还应尽量匹配参考实体、债务层级、币种、名义本金、观察时点、到期日，并明确 CDS 的信用事件、重组条款和可交割债务。债券若可赎回、可回售或可转换，还要先处理嵌入期权；否则差值混入了可选性。

## ASW、Z-spread 与相对国债利差不是同一个量
<!-- bilingual-en:start -->
*ASW, Z-spread, and government spread are not interchangeable*
<!-- bilingual-en:end -->

| 度量 | 它回答的问题 | 主要口径依赖 |
|---|---|---|
| ASW spread | 把债券与利率互换组合后，浮动参考利率之上需要多少利差？ | 债券价格、息票、互换曲线、日计数和资产互换报价约定 |
| Z-spread | 在整条零息即期曲线的每个期限点上加多少常数，才能把债券**合同现金流**折现到市场价格？ | 基准零息曲线、复利约定；有嵌入期权时不能替代 OAS |
| government spread | 债券到期收益率比选定政府债基准收益率高多少？ | 政府债基准券选择、两券期限差、政府债自身的稀缺性与流动性 |

因此，“CDS 减 Z-spread”或“CDS 减 government spread”可以是另一个研究变量，却不是本卡定义的 \(B_{\mathrm{ASW}}\)。尤其不能先用 ASW 给出“正基差”的解释，再悄悄用另一个利差计算符号。

## 最小算例：同一组报价可因口径而变号
<!-- bilingual-en:start -->
*Minimal example: the sign can change when the bond-spread measure changes*
<!-- bilingual-en:end -->

设五年期 CDS 的平价等价利差为 \(180\) bp，同期限债券的 ASW 为 \(140\) bp，则

\[
B_{\mathrm{ASW}}=180-140=+40\text{ bp}.
\]

按本卡口径，这是正基差，含义仅是 CDS 比 ASW 宽 \(40\) bp。假设同一债券同时报出 Z-spread \(155\) bp、相对政府债利差 \(190\) bp，那么机械相减会分别得到 \(+25\) bp 与 \(-10\) bp。三个结果来自同一市场快照；差别由度量口径造成。只有 \(+40\) bp 才能被称为本卡的 CDS–ASW 基差。这组假设数值用于辨别口径，不主张三种利差通常具有这种排序，也不自动推出可交易利润。

> [!question]- 最小自检
> CDS 为 \(130\) bp、同期限 ASW 为 \(190\) bp。按本卡定义，基差是多少？能否仅凭该数值断言存在 \(60\) bp 的无风险收益？
>
> **答案：** \(B_{\mathrm{ASW}}=130-190=-60\) bp，是负基差；不能。\(60\) bp 只是未扣融资、保证金、交易成本和合约错配前的报价差，实际策略见 [[负基差交易]]。

## 为什么基差可以持续偏离零
<!-- bilingual-en:start -->
*Why the basis can persist away from zero*
<!-- bilingual-en:end -->

只有在一个很强的理想化环境中，现金债和 CDS 才近似复制同一信用风险：可无摩擦融资和做空、两边流动性相同、信用事件与损失给付完全匹配、没有对手方风险、保证金成本或资产负债表约束。在现实中，至少有以下楔子：

1. **融资与中介资本。** 买入债券通常要占用现金或回购融资，还要承担 haircut 和续作；中介机构资产负债表受限时，即使看见差价也未必有能力压平。
2. **相对流动性。** 债券是分券、分期限的现金工具，CDS 则可能在标准期限集中交易。买卖价差、市场深度和价格发现速度不同，任一市场的流动性冲击都可能改变基差。
3. **信用事件和交割错配。** CDS 只按合同覆盖的信用事件结算；可交割债务、重组期限限制、拍卖最终价和最便宜可交割（CTD）选择，会使 CDS 给付不等于投资者手中那只债券的实际损失。合同边界见 [[CDS信用事件与结算]]。
4. **对手方与抵押品。** 现金债是发行人风险，CDS 保护还叠加保护卖方、净额、抵押品和保证金机制。保护卖方与参考实体共同恶化时，名义上的信用对冲可能最弱。
5. **供需与技术流。** 债券新发行、指数调仓、结构化产品对保护的买卖、监管或会计需求，都可只冲击其中一个市场。
6. **曲线与模型选择。** 到期日不完全相同、固定票息加 upfront 的换算、回收率假设、利率曲线和债券可选性处理，也会把“模型差”混入“市场差”。

压力期并没有固定符号。融资与现金债流动性冲击可把 ASW 推得比 CDS 更宽，形成大幅负基差；保护需求或 CDS 市场自身的约束也可能把 CDS 推宽。ECB 记录的雷曼违约后投资级基差转为显著负值，是“危机不必为正”的历史例子，而不是“危机必为负”的普遍定律。

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 基差是**相对价格度量**，不是违约概率、预期损失或完整损益。
- \(B_{\mathrm{ASW}}=0\) 不证明两项合约可以逐状态复制；债券本金、息票和回收与 CDS 的保费腿、保护腿及结算规则仍不同。
- ASW 是债券价格相对于互换曲线的报价，不是某一投资者实际可获得的回购利率。评估交易必须另行加入实际融资。
- 多只债券对应同一参考实体时，债券选择会改变 ASW；CDS 的可交割集合和 CTD 选择又可能改变保护价值。
- “基差最终收敛”是交易假设，不是由定义推出的定理；到期前退出、违约或融资中断都可能先发生。

## 定位性来源与复核状态
<!-- bilingual-en:start -->
*Primary-source anchors and review status*
<!-- bilingual-en:end -->

- [ECB, *Financial Stability Review*, June 2009, Box 9, “The bond-CDS basis and the functioning of the corporate bond market,” pp. 77–79](https://www.ecb.europa.eu/press/financial-stability-publications/fsr/focus/2009/pdf/ecb~440f5aee6c.fsrbox200906_09.pdf)：直接把 bond–CDS basis 写为 CDS spread 与债券价格隐含 ASW spread 的差，并讨论融资、流动性、非完美替代与负基差持续。
- [Federal Reserve Board, FEDS 2011-18, “Cointegration Test with Stationary Covariates and the CDS-Bond Basis during the Financial Crisis,” §4](https://www.federalreserve.gov/pubs/feds/2011/201118/index.html)：定位资产互换由固定票息债券与利率互换组成、par ASW 的现金流含义、CDS 减 par ASW 的基差口径及正负基差的理想化交易方向。
- [ECB, *Financial Stability Review*, November 2020, Chart 2.7 notes](https://www.ecb.europa.eu/press/financial-stability-publications/fsr/html/ecb.fsr202011~b7be9ae1f1.en.html)：直接定义 Z-spread 为加到即期收益率曲线各点、使债券现金流现值等于价格的常数利差；[Federal Reserve Board, *Financial Stability Report*, November 2018, §1](https://www.federalreserve.gov/publications/2018-november-financial-stability-report-asset-valuation-pressures.htm)：定位公司债相对同期限 Treasury 的收益率利差，补齐三种债券利差的来源边界。
- [Federal Reserve Bank of New York, *Economic Policy Review* 24(2), “Trends in Credit Basis Spreads,” 2018, §1.2 and Exhibit 1/Table 4](https://www.newyorkfed.org/medialibrary/media/research/epr/pdf2/epr_2018_vol24no2.pdf)：定位平价等价 CDS 利差、CDS–现金债交易机制、回购 haircut、保证金和中介资产负债表成本。
- [Federal Reserve, “Sovereign CDS and Bond Pricing Dynamics in Emerging Markets: Does the Cheapest-to-Deliver Option Matter?”, IFDP 912, 2007](https://www.federalreserve.gov/pubs/ifdp/2007/912/default.htm)：在主权样本中定位 CTD 选择与相对流动性如何进入 CDS–债券价格关系；其主权结论不能无条件外推到所有公司债。
- [ISDA, *2014 ISDA Credit Derivatives Definitions* and the Credit Derivatives Physical Settlement Matrix](https://www.isda.org/book/2014-isda-credit-derivative-definitions)：定位信用事件、义务、可交割债务及结算选择的合同框架。

> [!success] 复核状态
> 独立模型复核通过：平价等价 CDS 利差、ASW 口径、基差符号和交易方向已分别核对；实际应用仍须重新匹配期限、层级、币种、重组和可交割条款。
