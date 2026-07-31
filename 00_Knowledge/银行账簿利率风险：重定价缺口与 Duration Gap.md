---
aliases:
  - "Interest Rate Risk in the Banking Book"
  - "IRRBB"
  - "Repricing Gap and Duration Gap"
  - "银行账簿利率风险"
status: source-checked
---

# 银行账簿利率风险：重定价缺口与 Duration Gap
<!-- bilingual-en:start -->
*Interest-Rate Risk in the Banking Book: Repricing Gap and Duration Gap*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 同时衡量利率变化对银行短期净利息收入和长期权益经济价值的影响，并纳入存款、提前还款和基差等行为选择权。
> **具体锚点：** 长期固定利率贷款由可快速重定价存款融资时，加息会使负债成本先上升、净利息收入承压，同时贷款经济价值下降。
> **核心难点：** 无到期存款和提前还款没有固定现金流；重定价缺口看收入时点，Duration Gap 看经济价值，二者可能方向不同。
> **为什么重要：** 一家账面资本充足的银行仍可能因持续净息差压缩或经济价值损失变得脆弱，单一到期表无法揭示这种风险。
> **继续：** 先分别建立 NII 与 EVE 视角，再对存款 beta、流失、提前还款和曲线扭曲做共同压力；证券久期基础见 [[债券久期、凸性与收益率曲线风险]]。
> <!-- bilingual-en:start -->
> **What it solves:** It jointly measures how rate changes affect a bank's short-run net interest income and long-run economic value of equity while incorporating behavioral options in deposits, prepayment, and basis.
> **Concrete anchor:** When long-term fixed-rate loans are funded by deposits that reprice quickly, a rate increase raises liability cost first and compresses net interest income while also reducing the loans' economic value.
> **Central difficulty:** Non-maturity deposits and prepayments have no fixed cash-flow schedule. Repricing gap examines earnings timing, while duration gap examines economic value, and the two can point in different directions.
> **Why it matters:** A bank with adequate book capital can still become fragile through persistent margin compression or economic-value loss; a simple maturity table does not reveal the risk.
> **Continue:** Build NII and EVE views separately, then jointly stress deposit beta, runoff, prepayment, and curve twists. For security-duration foundations, see [[债券久期、凸性与收益率曲线风险|Bond Duration, Convexity, and Yield-Curve Risk]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
> - 本库货币银行学课程笔记：支持课程范围、案例和课堂顺序。
> - [Federal Reserve Education](https://www.federalreserveeducation.org/) 与各专题官方说明：核验中央银行、货币政策、银行体系与金融市场机制。
> - Basel Committee《[Interest rate risk in the banking book](https://www.bis.org/bcbs/publ/d368.htm)》：核验 EVE、NII、行为与基差风险框架。
> <!-- bilingual-en:start -->
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
> - The vault's Money and Banking course notes support course scope, examples, and lecture sequence.
> - [Federal Reserve Education](https://www.federalreserveeducation.org/) and its official topic pages verify central banking, monetary policy, banking-system, and financial-market mechanisms.
> - The Basel Committee's “[Interest rate risk in the banking book](https://www.bis.org/bcbs/publ/d368.htm)” verifies the EVE, NII, behavioral-risk, and basis-risk framework.
> <!-- bilingual-en:end -->

## 两种视角先分开
<!-- bilingual-en:start -->
*Separate the two perspectives first*
<!-- bilingual-en:end -->

- **净利息收入（NII）视角：** 在给定短期规划期内，资产收益和负债成本按何时重定价变化。
- **权益经济价值（EVE）视角：** 所有资产、负债与表外现金流现值怎样随整条曲线变化。
<!-- bilingual-en:start -->
- **Net interest income perspective:** Over a specified short planning horizon, asks when asset yields and liability costs reprice.
- **Economic value of equity perspective:** Asks how the present value of all assets, liabilities, and off-balance-sheet cash flows changes with the full curve.
<!-- bilingual-en:end -->

收入视角关注未来若干季或年的会计/经营盈利，经济价值视角把剩余全期限现金流一次性折现。短端上升可能先改善浮息资产收入，却因长期固定资产跌价而损害 EVE；不能要求两者总是同号。
<!-- bilingual-en:start -->
The earnings view focuses on accounting or operating profit over coming quarters or years, while the economic-value view discounts all remaining cash flows at once. A short-rate rise can initially improve floating-rate asset income while damaging EVE through losses on long fixed-rate assets; the two need not have the same sign.
<!-- bilingual-en:end -->

## 重定价缺口与净利息收入
<!-- bilingual-en:start -->
*Repricing gap and net interest income*
<!-- bilingual-en:end -->

按时间桶比较 rate-sensitive assets 与 liabilities 可近似短期收入敏感度，但忽略提前偿还、无到期存款 beta、基差和曲线形状。行为假设应压力测试。
<!-- bilingual-en:start -->
Comparing rate-sensitive assets and liabilities by time bucket approximates short-run earnings sensitivity but ignores prepayment, non-maturity-deposit beta, basis, and curve shape. Behavioral assumptions should be stressed.
<!-- bilingual-en:end -->

若未来一年内重定价资产为 600、重定价负债为 800，简单 gap 为 −200。若相关利率均平行上升 100bp，静态近似 NII 变化为 $-200\times1\%= -2$。但若资产和负债参考不同利率、存款只传导 40bp 或客户流失，实际结果会不同。
<!-- bilingual-en:start -->
If 600 of assets and 800 of liabilities reprice within one year, the simple gap is −200. If relevant rates all rise in parallel by 100 basis points, the static approximate NII change is $-200\times1\%=-2$. But if assets and liabilities reference different rates, deposits pass through only 40 basis points, or customers leave, the actual result differs.
<!-- bilingual-en:end -->

## duration gap 与免疫
<!-- bilingual-en:start -->
*Duration gap and immunization*
<!-- bilingual-en:end -->

权益经济价值对小平行变动近似由资产/负债久期和规模差决定。免疫需匹配现值与久期，随着时间和利率变化动态再平衡；凸性、现金流和模型误差留下残余风险。
<!-- bilingual-en:start -->
For a small parallel shift, economic value of equity is approximately determined by differences in asset and liability duration and scale. Immunization requires matching present value and duration and rebalancing dynamically as time and rates change; convexity, cash flow, and model error leave residual risk.
<!-- bilingual-en:end -->

常用规模调整 gap 为 $DGAP=D_A-(L/A)D_L$，经济价值变化近似 $\Delta E\approx-A\,DGAP\,\Delta y/(1+y)$。若资产 1,000、负债 900，$D_A=5$、$D_L=2$，则 $DGAP=5-0.9\times2=3.2$；利率上升会显著降低权益经济价值。
<!-- bilingual-en:start -->
A common scale-adjusted gap is $DGAP=D_A-(L/A)D_L$, with economic-value change approximately $\Delta E\approx-A\,DGAP\,\Delta y/(1+y)$. If assets are 1,000, liabilities 900, $D_A=5$, and $D_L=2$, then $DGAP=5-0.9\times2=3.2$; a rate rise materially reduces economic value of equity.
<!-- bilingual-en:end -->

## 无到期存款与行为选择权
<!-- bilingual-en:start -->
*Non-maturity deposits and behavioral options*
<!-- bilingual-en:end -->

合同上可随时提款的存款，经济上可能有稳定核心余额；其利率也常不按市场一比一变化。模型需估计余额存续期、存款 beta、迁移到高息产品的速度和压力期流失。把全部活期存款设为隔夜或永久稳定都会产生巨大模型偏差。
<!-- bilingual-en:start -->
Deposits withdrawable on demand may have a stable behavioral core, and their rates often do not move one-for-one with market rates. Models must estimate balance life, deposit beta, migration speed into higher-paying products, and stressed runoff. Treating all demand deposits as overnight or permanently stable creates large model error.
<!-- bilingual-en:end -->

贷款提前还款也是客户持有的利率选择权：利率下降时固定利率贷款更可能提前偿还，银行失去高息资产并需要按较低利率再投资；利率上升时期限反而延长。这种负凸性必须进入情景而非只用合同到期。
<!-- bilingual-en:start -->
Loan prepayment is also a customer-held rate option. When rates fall, fixed-rate loans are more likely to prepay, and the bank loses high-yield assets and reinvests at lower rates; when rates rise, duration extends. This negative convexity belongs in scenarios rather than a contractual-maturity table alone.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 用一年重定价 gap 推断全部经济价值损失，混淆 NII 与 EVE。
- 假设所有存款利率和市场利率同步同幅变化，忽略 beta 与客户迁移。
- 用合同到期建模可提前还款贷款，低估利率下降时的现金流变化。
- 只做平行曲线，遗漏短长端分化和不同参考利率的基差。
<!-- bilingual-en:start -->
- Using a one-year repricing gap to infer total economic-value loss confuses NII with EVE.
- Assuming every deposit rate moves simultaneously and one-for-one with market rates ignores beta and customer migration.
- Modeling prepayable loans by contractual maturity understates cash-flow change when rates fall.
- Testing only parallel curves omits short–long divergence and basis between reference rates.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 银行存款没有合同到期，利率风险怎样处理？
<!-- bilingual-en:start -->
*How should interest-rate risk be handled for bank deposits without contractual maturity?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 需建立余额稳定性和存款利率 beta 的行为模型，并在不同流失/重定价情景下压力测试。
> <!-- bilingual-en:start -->
> Build behavioral models of balance stability and deposit-rate beta and stress them under different runoff and repricing scenarios.
> <!-- bilingual-en:end -->

### 用自己的话解释：NII gap 与 Duration Gap 为什么可能给出不同方向？
<!-- bilingual-en:start -->
*Explain in your own words: why can the NII gap and duration gap point in different directions?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> NII 看短期内哪些利率先重定价，Duration Gap 看全部未来现金流现值；浮息资产可迅速增加收入，却不能抵消长期固定现金流因贴现率上升而产生的价值损失。
> <!-- bilingual-en:start -->
> NII asks which rates reprice first over a short horizon; duration gap values all future cash flows. Floating-rate assets can quickly raise income yet fail to offset the present-value loss on long fixed cash flows when discount rates rise.
> <!-- bilingual-en:end -->

### 资产负债久期匹配后为什么仍需再平衡？
<!-- bilingual-en:start -->
*Why is rebalancing still needed after matching asset and liability duration?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 时间经过、利率变化、客户行为和现金流实现会改变现值、久期与凸性；一次匹配只在当前点对特定小冲击近似免疫。
> <!-- bilingual-en:start -->
> Passage of time, rate changes, customer behavior, and realized cash flows alter present values, durations, and convexities. One match immunizes only approximately at the current point against a specified small shock.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 本库货币银行学课程笔记：支持课程范围、案例和课堂顺序。
- [Federal Reserve Education](https://www.federalreserveeducation.org/) 与各专题官方说明：核验中央银行、货币政策、银行体系与金融市场机制。
- Basel Committee《[Interest rate risk in the banking book](https://www.bis.org/bcbs/publ/d368.htm)》：逐项核验 NII/EVE、重定价、基差、无到期存款与提前还款；缺口算例按资产负债表规模和久期公式复算。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- The vault's Money and Banking course notes support course scope, examples, and lecture sequence.
- [Federal Reserve Education](https://www.federalreserveeducation.org/) and its official topic pages verify central banking, monetary policy, banking-system, and financial-market mechanisms.
- The Basel Committee's “[Interest rate risk in the banking book](https://www.bis.org/bcbs/publ/d368.htm)” was checked for NII and EVE, repricing, basis, non-maturity deposits, and prepayment; the gap examples were recomputed from balance-sheet scale and duration formulas.
<!-- bilingual-en:end -->
