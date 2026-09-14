---
aliases:
  - "QE 增加准备金不等于银行按固定倍数增加贷款或广义货币"
  - QE reserves are not a fixed money multiplier
  - QE 与货币乘数边界
student_os: knowledge-atom
atom_id: MB-MONEY-022
atom_set: money-liquidity-aggregates
atom_type: failure-mode
status: source-checked
mastery_state: unassessed
requires:
  - "[[美国货币基础、M1 与 M2]]"
  - "[[央行购债结算链]]"
related:
  - "[[货币乘数的因果边界]]"
  - "[[央行购券与贷款]]"
part_of:
  - "[[货币、流动性与货币口径.canvas]]"
---

# QE 增加准备金不等于银行按固定倍数增加贷款或广义货币
<!-- bilingual-en:start -->
*QE-created reserves do not make banks expand lending or broad money by a fixed multiple*
<!-- bilingual-en:end -->

> [!summary] 原子失败模式
> 在标准银行准备金结算链下，量化宽松购债会创造准备金；若卖方是非银行，还可初始增加其银行存款。但准备金只能由合格机构持有并通过央行账户在这些机构之间转移，不能像一袋可贷给公众的现成资金那样被固定倍数“放大”为贷款。
> <!-- bilingual-en:start -->
> Under the standard settlement chain through bank reserve accounts, quantitative-easing purchases create reserves and, when the seller is a non-bank, can initially raise that seller's bank deposit. But reserves are held by eligible institutions and transferred among them through central-bank accounts; they are not a bag of pre-existing funds that is lent to the public and mechanically multiplied into loans.
> <!-- bilingual-en:end -->

银行愿不愿扩大贷款取决于有信用需求、贷款定价与风险、资本、监管、融资和预期盈利；放贷本身通常同步创造贷款资产与存款负债，跨行支付再需要准备金结算。QE 可通过利率、资产价格、组合再平衡、预期和融资条件影响信贷与支出，但没有“准备金增加一单位必然带来固定多单位广义货币”的会计定律。
<!-- bilingual-en:start -->
Bank lending depends on creditworthy demand, pricing and risk, capital, regulation, funding, and expected profitability. Lending normally creates a loan asset and deposit liability together, with reserves needed later for interbank settlement. QE may affect credit and spending through interest rates, asset prices, portfolio rebalancing, expectations, and financing conditions, but there is no accounting law by which one extra unit of reserves must create a fixed number of units of broad money.
<!-- bilingual-en:end -->

## 易错边界

- 非银行卖方获得存款是购债结算的直接结果，不是银行先拿准备金再乘数放贷的结果。
- 银行卖方只把证券换成准备金时，广义货币不必在初始步骤增加。
- “不机械增加贷款”不等于“QE 没有效果”；它否定的是固定数量因果链，不是否定利率和组合渠道。
<!-- bilingual-en:start -->
- A non-bank seller's new deposit is a direct settlement result, not the outcome of a bank first receiving reserves and then multiplying loans.
- When the selling institution is a bank exchanging securities for reserves, broad money need not rise in the initial step.
- “Does not mechanically increase lending” does not mean “QE has no effects”; it rejects a fixed quantity chain, not interest-rate or portfolio channels.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“QE 后银行准备金增加十倍，所以贷款必增十倍”不是资产负债表恒等式？
>
> **答案：** 准备金是银行结算资产，贷款决策受需求、风险、资本、定价和监管约束；准备金与贷款之间不存在固定倍数的逐笔转换。
> <!-- bilingual-en:start -->
> Why is “bank reserves rose tenfold after QE, so loans must rise tenfold” not a balance-sheet identity?
>
> **Answer:** Reserves are settlement assets, while lending is constrained by demand, risk, capital, pricing, and regulation. There is no fixed one-for-many conversion from reserves into loans.
> <!-- bilingual-en:end -->

## 来源与核验

- [[02_Economy/03_货币银行学/4_中央银行与货币政策的实施/14_货币供给过程.md#简单存款乘数：条件化的教学模型|课程：简单乘数边界]]：核对教学模型依赖无现金漏损、无额外准备和充分再贷等强假设。
- [Bank of England, Money creation in the modern economy](https://www.bankofengland.co.uk/quarterly-bulletin/2014/q1/money-creation-in-the-modern-economy)：核对 QE、非银行卖方存款、组合再平衡及准备金不能被机械乘数转成贷款。
- [Federal Reserve Board, IORB FAQs](https://www.federalreserve.gov/monetarypolicy/iorb-faqs.htm)：核对准备金的持有人、结算与充足准备金政策框架。
<!-- bilingual-en:start -->
- The local multiplier critique was checked for the restrictive assumptions behind the teaching model.
- The Bank of England article was checked for QE settlement, non-bank deposits, portfolio rebalancing, and rejection of mechanical reserve multiplication.
- The Federal Reserve IORB FAQ was checked for reserve holders and the ample-reserves framework.
<!-- bilingual-en:end -->
