
# 1. 衍生产品信用风险的特点与基本概念
<!-- bilingual-en:start -->
*1. Counterparty Credit Risk in Derivatives: Key Features and Concepts*
<!-- bilingual-en:end -->

衍生产品交易的信用风险比传统贷款的信用风险更加复杂 。原因在于，贷款的风险敞口在发放时就确定（例如一笔5年期、1000万美元贷款在整个期间风险敞口基本为1000万），而衍生产品的未来风险暴露（违约发生时可能遭受的损失金额）是不确定的，会随市场变化而波动 。例如，银行与客户做一笔5年期利率互换：如果互换对银行而言现值为正，意味着客户欠银行钱，此时银行的信用风险敞口等于互换合约的当前价值；如果互换对银行而言价值为负（银行欠客户），则银行面对的信用风险敞口为零 。也就是说，衍生品交易的敞口动态变化，只有当交易对手的合约市值对我们为正时才存在信用暴露风险。
<!-- bilingual-en:start -->
Counterparty credit risk is more complex for derivatives than for a conventional loan. A loan's exposure is largely fixed when it is advanced: a five-year USD 10 million loan has roughly USD 10 million of principal exposure throughout. A derivative's future exposure is uncertain because its market value changes. In a five-year interest-rate swap, the bank is exposed when the swap has positive value to the bank—the client then owes the bank that amount. When the swap has negative value to the bank, the bank owes the client and its current counterparty exposure is zero. Exposure therefore exists only on the positive part of the contract's market value.
<!-- bilingual-en:end -->

为降低衍生品交易的信用风险，场外衍生品通常采用双边净额清算等机制 。国际掉期与衍生工具协会（ISDA）的主协议允许对同一对手的多笔交易进行净额结算：在某一方违约或发生提前终止事件时，双方所有未了结交易按约定归并为一笔交易进行净结算 。净额结算能显著降低信用风险 —— 将多笔交易的敞口由逐笔相加的大值降低为净敞口 。此外，设置**抵押品**（保证金）和**主协议补充条款**（如降级触发）等也是控制衍生品对手信用风险的重要方法，在后文中将进一步介绍。
<!-- bilingual-en:start -->
Over-the-counter derivatives commonly use bilateral netting to reduce counterparty credit risk. Under an International Swaps and Derivatives Association (ISDA) master agreement, all eligible outstanding trades with the same counterparty can be terminated and combined into a single net amount after default or another early-termination event. Netting replaces the sum of gross positive exposures with one net exposure. **Collateral**, margin arrangements, and contractual protections such as downgrade triggers provide additional risk mitigation.
<!-- bilingual-en:end -->

# 2. CVA与DVA的定义、公式及意义
<!-- bilingual-en:start -->
*2. Definitions, formulas, and significance of CVA and DVA*
<!-- bilingual-en:end -->

**信用价值调整**（[[对手方信用风险、CVA 与 DVA|CVA]], Credit Valuation Adjustment）是指衍生产品交易商对因交易对手可能违约所导致损失的**预期损失值**的估计 。CVA反映了交易对手信用风险对衍生品价值的影响，在计价时应当从假定无违约情况下的衍生品风险中性价值中减去CVA 。会计上，衍生品的公允价值=假设无违约的价格 - CVA + DVA，其中CVA总和的变动需要在当期损益中体现 。
<!-- bilingual-en:start -->
**Credit valuation adjustment** ([[对手方信用风险、CVA 与 DVA|CVA]], Credit Valuation Adjustment) is the dealer's estimate of expected loss caused by possible counterparty default. It captures the effect of counterparty credit risk on a derivative's value and is deducted from the risk-neutral value calculated under a no-default assumption. In the accounting expression used here, derivative fair value equals clean value minus CVA plus DVA, and changes in aggregate CVA pass through current profit or loss.
<!-- bilingual-en:end -->

## 2.1 CVA

**CVA计算公式：**假设与某对手方的交易组合最长到期时间为$T$，将区间$[0,T]$划分为若干时间段（如按季度、年份等）。记第$i$个时间段起止为$(t_{i-1}, t_i]$（$i=1,2,\dots,n$，且$t_0=0, t_n=T$）。令$q_i$为交易对手在第$i$时间段内发生违约的风险中性概率，$v_i$为若交易对手在第$i$段违约时，交易商对其净风险敞口在**该段中点**的**[[货币时间价值与贴现|现值]]**（考虑抵押品后）。另设$R$为违约时交易对手的回收率（即$1-R$为损失率）。若假定风险暴露$v_i$与违约概率$q_i$相互独立，则第$i$时间段违约所致的**预期损失现值**为$(1-R),q_i,v_i$ 。因此**整个交易组合的CVA**可表示为各期预期损失之和：
<!-- bilingual-en:start -->
**CVA formula:** Let the longest maturity in a counterparty portfolio be $T$ and partition $[0, T]$ into periods $(t_{i-1}, t_i]$, for $i=1,2,\ldots, n$, with $t_0=0$ and $t_n=T$. Let $q_i$ be the risk-neutral probability that the counterparty defaults during period $i$. Let $v_i$ be the **[[货币时间价值与贴现|present value]]** of the dealer's net positive exposure, after collateral, measured at that period's midpoint if default occurs in the period. Let $R$ be recovery, so $1-R$ is loss given default. If exposure and default are assumed independent, period-$i$ expected discounted loss is $(1-R)q_iv_i$. Portfolio CVA is the sum across periods:
<!-- bilingual-en:end -->

$CVA  =  (1−R)∑i=1nqi vi .\displaystyle \text{CVA} \;=\; (1-R)\sum_{i=1}^n q_i\,v_i$
上述$CVA$公式可以理解为：对手方每个期间违约的概率乘以违约时我们正向敞口的期望，再乘以违约损失率$(1-R)$，加总得到整个组合由于对手信用风险需要扣减的价值 。$q_i$可依据对手方信用利差推算得到，如假设违约强度$\lambda$恒定则$q_i \approx e^{-\lambda t_{i-1}} - e^{-\lambda t_i}$（即存活概率的下降）；实际计算时常根据对手各期限的信用利差曲线估计各期违约概率 。
<!-- bilingual-en:start -->
The formula says that CVA is the sum, across possible default periods, of the counterparty's risk-neutral probability of default multiplied by the expected positive exposure at default and the loss-given-default fraction $(1-R)$. The period probabilities $q_i$ can be inferred from the counterparty's credit-spread term structure. With a constant default intensity $\lambda$, for example, $q_i\approx e^{-\lambda t_{i-1}}-e^{-\lambda t_i}$, the fall in survival probability over the period.
<!-- bilingual-en:end -->

## 2.2 DVA

**债务价值调整**（[[对手方信用风险、CVA 与 DVA|DVA]], Debit Valuation Adjustment）则从**交易对手**的角度考量交易商本身违约的风险 。DVA定义为交易对手因交易商可能违约而预期会损失的价值，也就是交易商自身违约风险给交易商“带来的好处” 。在衍生品定价中会**加入DVA**（即加回DVA）：公允价值 = 无违约价值 - CVA + DVA 。例如，若交易商自身信用状况恶化（信用利差上升），则交易商违约概率上升会导致更高的DVA值，从而账面上增加交易商收益 。需要注意，DVA体现的是自身违约对本方的“利好”影响，其计算与CVA类似（可视为对手方针对本行计算的CVA，对应本行的负向风险敞口）。尽管DVA在账面上提升了衍生品价值，但其现实意义存在争议，后文将详细讨论。
<!-- bilingual-en:start -->
**Debit valuation adjustment** ([[对手方信用风险、CVA 与 DVA|DVA]]) is the adjustment for the dealer's own default risk. From the counterparty's perspective, it is the expected loss caused by the dealer's possible default; from the dealer's perspective, that same loss appears as a reduction in the value of what the dealer may have to pay. Hence the bilateral expression used here is fair value = clean value − CVA + DVA. If the dealer's credit spread widens, its own default probability and DVA rise, which can create an accounting gain. DVA is therefore economically similar to the CVA that the counterparty calculates on the dealer's negative exposure, but treating deterioration in one's own credit as a gain is controversial.
<!-- bilingual-en:end -->

## 3. 衍生品风险敞口的计算方法（含抵押品与补救期）
<!-- bilingual-en:start -->
*3. Calculating Derivative Exposure, Including Collateral and Cure Periods*
<!-- bilingual-en:end -->

**风险敞口**（Exposure）通常定义为在某一未来时刻，假如对手方违约时交易商可能遭受的损失。对于**无抵押**的交易组合，在任意时点$t$交易商的暴露可表示为交易组合的净市值的正部位，即$\max(V(t),,0)$（若$V(t)$为负则交易商不存在风险暴露） 。当双方有多笔交易时，需通过ISDA协议约定的**[[OTC 衍生品清算、保证金与 CCP 风险|净额结算]]**将多笔交易的正负价值抵消后计算净暴露 。
<!-- bilingual-en:start -->
**Exposure** is the loss a dealer would face at a future time if the counterparty defaulted then. For an **uncollateralised** portfolio at time $t$, exposure is the positive part of net market value, $\max(V(t),0)$; a negative $V(t)$ means the dealer owes the counterparty and has no current counterparty exposure. Under an ISDA agreement, **[[OTC 衍生品清算、保证金与 CCP 风险|netting]]** offsets positive and negative values across eligible trades before the positive part is taken.
<!-- bilingual-en:end -->

在考虑**抵押品（担保品）**的情况下，风险敞口需扣除对手方已交付的抵押品金额$C$。例如在时点$t$，若交易组合对交易商的净市场价值为$V(t)$，而对手方按保证金协议已交付价值为$C$的担保物，则交易商此时对该对手的风险敞口为 **$\max(V(t) - C,;0)$** 。也就是说，抵押品可以冲抵交易的正向价值，降低损失。但需注意反向情形：如果交易商向对手方提交了保证金，则当交易商违约时该保证金不予退还，可能使得对手方实际敞口增加 。一般而言，双方签订的信用支持附约（CSA）会规定每日或定期根据净敞口变动交换抵押品，从而动态降低敞口。
<!-- bilingual-en:start -->
With **collateral**, subtract the amount $C$ already posted by the counterparty from the dealer's positive portfolio value. At time $t$, exposure is therefore **$\max(V(t) - C,;0)$**. Collateral absorbs part of a positive mark-to-market and reduces loss on counterparty default. The reverse direction also matters: collateral posted by the dealer becomes part of the counterparty's exposure if the dealer defaults and the collateral is not returned. A Credit Support Annex (CSA) normally requires daily or periodic collateral transfers as net exposure changes.
<!-- bilingual-en:end -->

**补救期**（有时称“风险缓冲期”）是指当对手方停止支付担保品后，到交易最终平仓清算之间的一段时间 。在这段期间内，交易价值可能继续变化，而由于对手停止补充保证金，违约发生时可用的抵押品价值仍停留在最后一次正常补充时的水平 。因此，补救期的存在意味着即使有担保品，违约时敞口可能**高于**违约即时的担保覆盖，因为保证金未能完全及时反映交易价值的最新变动 。计算CVA时通常假设违约发生在每段的中点$t_i^*=(t_{i-1}+t_i)/2$，并考虑一个补救期长度$\delta$（如10天、20天等）来调整风险敞口：即需要同时考虑$t_i^*$时刻和提前$\delta$时刻（$t_i^* - \delta$）交易价值的变化来确定违约敞口 。
<!-- bilingual-en:start -->
The **cure period**, closely related to the margin period of risk, runs from the point at which the counterparty stops posting collateral until positions are closed out and settled. Portfolio value can continue to move while collateral remains at its last posted level. A collateralised portfolio can therefore have uncovered exposure at default. A CVA calculation may place default at the period midpoint $t_i^*=(t_{i-1}+t_i)/2$ and allow a cure period $\delta$, such as 10 or 20 days, by comparing value at $t_i^*$ with the collateral based on value at $t_i^*-\delta$.
<!-- bilingual-en:end -->

**有担保品及补救期的敞口示例：**设某双边交易协议要求每日按净敞口零起点交换抵押品，补救期为20天 。假定在时间$T$时交易对手B违约，A为交易商：
<!-- bilingual-en:start -->
**Example with collateral and a cure period:** suppose a bilateral agreement requires daily collateralisation of net exposure from a zero threshold and has a 20-day cure period. Counterparty B defaults at time $T$, and A is the dealer:
<!-- bilingual-en:end -->

- 情景①：违约时（$T$）交易组合对A的价值$V(T)=50$，而20天前的价值$V(T-!20d)=45$。由于B在违约前最后交付的担保品价值基于20天前的45，A实际持有抵押品45。违约时交易价值涨至50，超过抵押品**5**的部分即为A的未被担保风险敞口 。
<!-- bilingual-en:start -->
- Scenario 1: At the time of default ($T$), the value of the portfolio for A is $V(T)=50$, while its value 20 days prior was $V(T-!20d)=45$. Since B delivered its last collateral based on the value from 20 days ago at 45, A actually holds 45 units of collateral. At the time of default, the transaction value rises to 50, exceeding the collateral by 5 units, which represents A's uncovered risk exposure.
<!-- bilingual-en:end -->
    
- 情景②：违约时$V(T)=50$，而20天前$V(T-!20d)=55$。此时A持有担保品55，足以覆盖当前50的交易价值，因此A的敞口为**0**（抵押品有富余） 。
<!-- bilingual-en:start -->
- Scenario 2: At default $V(T)=50$, while 20 days earlier $V(T-!20d)=55$. A therefore holds collateral of 55, which fully covers current value of 50, so A's exposure is **zero** and collateral is in excess.
<!-- bilingual-en:end -->
    

上述例子表明，在违约前的补救期内交易价值若上升，会出现未被担保的敞口；反之若价值下降，则抵押品可能有余但不会返还（对违约方而言损失） 。因此补救期越长，抵押品覆盖的时滞越大，违约时未覆盖敞口可能越高。在CVA模拟中通常会通过在违约模拟时取交易价值的历史回溯（如提前$\delta$期的价值）来近似计入补救期的影响 。
<!-- bilingual-en:start -->
The examples show why collateral does not eliminate exposure during the cure period. If portfolio value rises after the last margin transfer, the increase is unsecured; if value falls, the dealer may hold excess collateral that is not immediately returned to the defaulting party. A longer cure period creates a larger lag between current value and collateral. CVA simulations commonly approximate this effect by comparing value at default with collateral based on value $\delta$ earlier.
<!-- bilingual-en:end -->

## 4. CVA计算中的Monte Carlo模拟应用
<!-- bilingual-en:start -->
*4. Application of Monte Carlo Simulation in CVA Calculations*
<!-- bilingual-en:end -->

由于衍生产品未来价值取决于市场风险因素（利率、汇率、股价、商品价格等）的随机演变，**蒙特卡罗模拟**是计算CVA时常用的工具 。基本思路是在**风险中性**假设下，对未来市场变量从当前时刻一直模拟到交易组合最长到期$T$ 。沿每条模拟路径，可在预设的时间网格上（例如每季度或每半年中点）计算交易商对交易对手的**暴露**（即组合价值为正时的数值，负时按0计） 。对大量模拟路径取平均，得到每个时间段中点的平均正向暴露，再贴现到当前即为$v_i$ 。重复此过程可估计出每一期的$v_i$，再结合对手各期违约概率$q_i$套用公式计算CVA。
<!-- bilingual-en:start -->
Because future derivative values depend on the stochastic evolution of rates, exchange rates, equity prices, commodity prices, and other market factors, **Monte Carlo simulation** is widely used for CVA. Under the **risk-neutral** measure, simulate market factors from today to the portfolio's longest maturity $T$. On each path and time-grid point, calculate positive portfolio value, setting negative value to zero. Average across paths to obtain expected positive exposure at each period midpoint and discount it to obtain $v_i$. Combine these $v_i$ values with period default probabilities $q_i$ to calculate CVA.
<!-- bilingual-en:end -->

进行CVA模拟时需注意采用**风险中性概率分布**和**无风险利率贴现**，以确保计算得到的CVA符合衍生品定价原理 。也就是说，模拟中的市场变量应按照风险中性测度演化（预期符合当前市场远期曲线或期权隐含分布），违约概率也取自风险中性违约分布；同时计算预期损失时对未来现金流的折现使用无风险利率 。这样算出的CVA可以被视作在风险中性世界中衍生品因对手违约风险造成的价值折减。
<!-- bilingual-en:start -->
CVA is a valuation adjustment, so the simulation uses a **risk-neutral distribution** and risk-neutral default probabilities. Market factors should be calibrated to current forward curves or option-implied distributions, and future losses are discounted consistently with the valuation framework. The result is the amount by which counterparty default risk reduces the derivative's value in the risk-neutral pricing measure.
<!-- bilingual-en:end -->

实际操作中，银行会对每个重要对手定期运行CVA的Monte Carlo仿真，生成所有交易组合的敞口分布和CVA估计值。由于模拟路径已生成并存储，**新增交易**的CVA影响也可方便地通过同一批路径计算（后面章节详述新增CVA计算） 。
<!-- bilingual-en:start -->
In practice, banks regularly run Monte Carlo CVA calculations for each important counterparty, producing exposure distributions and a CVA estimate for the entire netting set. Because the simulated market paths are stored, the incremental effect of a **new trade** can be calculated on the same paths rather than by rebuilding the simulation from scratch.
<!-- bilingual-en:end -->

## 5. 峰值敞口（Peak Exposure）与最大峰值敞口
<!-- bilingual-en:start -->
*5. Peak Exposure and Maximum Peak Exposure*
<!-- bilingual-en:end -->

除了计算平均意义下的CVA，银行还关心在低概率不利情形下的高敞口水平，即**峰值敞口**（Peak Exposure）。**峰值敞口**通常定义为在某个给定置信水平下（例如97.5%或99%分位数），衍生品组合在未来某时刻的高分位风险暴露值 。换言之，峰值敞口是暴露分布的某高百分位数对应的敞口水平。例如，若采用97.5%分位作为峰值标准，基于10000次蒙特卡罗模拟，在某时刻将所有模拟暴露按大小排序取第250大的值（因为10000×2.5%=250），这个值即为该时刻的峰值敞口 。
<!-- bilingual-en:start -->
Besides expected exposure, banks monitor high exposures in low-probability adverse scenarios. **Peak exposure** is a high percentile—for example, the 97.5th or 99th percentile—of the exposure distribution at a specified future date. With 10,000 simulations, the 97.5th percentile can be found by sorting exposures from largest to smallest and taking approximately the 250th-largest observation, because only 2.5% of simulated exposures lie above it.
<!-- bilingual-en:end -->

**最大峰值敞口**（Maximum Peak Exposure）则是对整个期限内各时点的峰值敞口取最大值 。它代表了在高置信水平下，整个交易期间最极端的潜在敞口大小，有助于风险管理者了解最坏情况下可能承受的最大信用暴露。
<!-- bilingual-en:start -->
**Maximum peak exposure** is the largest peak exposure across all future dates in the portfolio's life. It represents the most extreme high-quantile credit exposure over the full horizon and helps risk managers identify the worst point on the exposure profile.
<!-- bilingual-en:end -->

需要注意，**CVA的计算与峰值敞口的模拟有所区别**。【CVA】使用**风险中性**世界下的模拟（因CVA本质是定价调整，要符合无套利原则），而计算【峰值敞口】通常作为压力测试或风险限额评估，采用**真实世界**分布进行情景分析 。因此，CVA模拟中市场变量的预期跟随风险中性假设（如利率、价格按各自贴现率或期望漂移），而峰值敞口分析中市场变量按照实际历史或假设的现实分布演进。简单来说，CVA重在**定价**, 峰值敞口重在**风险管理**。
<!-- bilingual-en:start -->
**CVA and peak-exposure simulation serve different purposes.** CVA uses **risk-neutral** simulation because it is a no-arbitrage valuation adjustment. Peak exposure is normally a risk-limit or stress measure based on a **real-world** or scenario distribution. In short, CVA is primarily a **pricing** measure, while peak exposure is primarily a **risk-management** measure.
<!-- bilingual-en:end -->

## 6. 降级触发条款及案例（如AIG事件）
<!-- bilingual-en:start -->
*6. Downgrade Trigger Provisions and Case Studies (e.g., AIG Event)*
<!-- bilingual-en:end -->

**降级触发**（Downgrade Trigger）是指在衍生品交易的信用支持附属协议（CSA）中约定的一种条款：当交易一方的信用评级被下调到某一门槛以下时，该方须向对手方提供额外的抵押品 。此条款旨在提前缓释交易对手的信用恶化风险。==以AIG事件为例：许多AIG与投行的衍生品交易协议规定，当AIG的信用评级高于AA级时，无需为交易支付抵押品；一旦其评级跌破AA级，必须立即按敞口提供抵押 。2008年9月15日，AIG被三大评级机构同时降至AA以下，**降级触发条款被触发**，交易对手纷纷要求AIG补缴大额保证金。短时间内AIG面临巨额现金需求，流动性枯竭，最终只能靠政府紧急救助才免于破产 。==
<!-- bilingual-en:start -->
A **downgrade trigger** is a CSA provision requiring a party to post additional collateral when its credit rating falls below a specified threshold. The clause protects the counterparty before credit quality deteriorates further. In the AIG example, many contracts required little or no collateral while AIG remained above AA, but immediate collateral once it fell below that threshold. The 15 September 2008 downgrades activated these clauses across many contracts at once, producing enormous margin calls and a severe liquidity crisis that ultimately required government support.
<!-- bilingual-en:end -->

降级触发条款在保护交易对手方面作用明显，但也有局限。如果一家机构与众多对手都签有类似降级触发，当其评级被调降时，可能出现对手方**同时大量索取现金担保**的情形，瞬间引发流动性危机（AIG就是例子） 。另外，若发生跳级降等（如从A级直接跌至违约），降级触发可能来不及发挥作用，对交易对手无实质保护 。因此，降级触发需谨慎运用，通常只有在个别交易对手有限度地采用时才能有效，否则可能加剧系统性风险。
<!-- bilingual-en:start -->
Downgrade triggers protect counterparties but can also amplify stress. If an institution has similar clauses with many counterparties, a single downgrade can generate simultaneous demands for cash collateral and cause a liquidity crisis, as AIG illustrates. A sudden jump from an investment-grade rating to default may also occur too quickly for the trigger to provide meaningful protection. The clauses must therefore be used with care: when they are widespread, they can increase rather than reduce systemic risk.
<!-- bilingual-en:end -->

总体而言，降级触发条款为交易对手提供了一定保障：当对方信用恶化时可提前获得更多抵押缓冲。但这一机制对被降级方压力很大，可能形成“评级雪崩”效应，因此在风险管理中需要权衡条款设计和敞口集中度。
<!-- bilingual-en:start -->
In short, downgrade triggers give counterparties an earlier collateral cushion as credit quality weakens, but place acute funding pressure on the downgraded firm and can create a ratings-and-liquidity cascade. Clause design must therefore be considered alongside the concentration of similar obligations.
<!-- bilingual-en:end -->

## 7. 新增交易对CVA的影响与新增CVA计算
<!-- bilingual-en:start -->
*7. How a New Trade Changes CVA and How to Calculate Incremental CVA*
<!-- bilingual-en:end -->

当对手方与交易商之间增加一笔新交易时，该交易对组合整体CVA的影响取决于**新交易价值与现有组合价值的相关性**：
<!-- bilingual-en:start -->
The effect of a new trade on portfolio CVA depends on the **relationship between the new trade's value and the value of the existing netting set**.
<!-- bilingual-en:end -->

- **正相关情形：**如果新增交易的价值与现有组合价值呈正相关（即组合价值高时，新交易也通常价值高），那么这笔新交易会**增加整体CVA**。因为当组合对交易商有较大正敞口时，新交易也可能同时产生较大正敞口，扩大极端情况下的潜在损失。
<!-- bilingual-en:start -->
- **Positive correlation:** if the new trade tends to be valuable to the dealer when the existing portfolio is also valuable, both positions create positive exposure at the same time. The new trade therefore tends to **increase portfolio CVA**.
<!-- bilingual-en:end -->
    
- **负相关情形：**如果新交易价值与现有组合价值呈负相关（组合价值高时，新交易价值低，或反向），则新交易会**降低整体CVA**。因为新交易在组合原本敞口大的情况下往往是负价值（对交易商有利，抵消部分风险），起到对冲作用，从而减少预期损失。
<!-- bilingual-en:start -->
- **Negative correlation:** if the new trade tends to have negative value when the existing portfolio has large positive value, it offsets exposure within the netting set and can **reduce portfolio CVA**.
<!-- bilingual-en:end -->
    

>[!example] 例子
> **假设交易商和某对手方已有一笔5年期外汇远期（对手方将来从银行买入外汇，银行持有潜在敞口）。若该对手希望新增一笔3年期外汇远期：
><!-- bilingual-en:start -->
>**Suppose a dealer and counterparty already have a five-year foreign-exchange forward under which the counterparty will buy foreign currency from the bank. They now consider an additional three-year forward:**
><!-- bilingual-en:end -->

- 如果对手方在新增3年远期中仍是**买入外汇**的一方，则新交易与原先5年远期方向相同，二者价值正相关。新交易将使组合总体在外汇上涨时的正向敞口进一步增大，因此CVA会**上升**。
<!-- bilingual-en:start -->
- If the counterparty is again the **buyer of foreign currency**, the two forwards point in the same direction. Their values are positively correlated, so the combined positive exposure and CVA tend to **rise**.
<!-- bilingual-en:end -->
    
- 如果对手方在新增远期中改为**卖出外汇**（方向相反），则当外汇价格变动导致原交易有敞口时，新交易会产生相反价值，部分抵消风险，因此组合CVA会**下降**。
<!-- bilingual-en:start -->
- If the counterparty instead **sells foreign currency** in the new forward, its value moves in the opposite direction and partly offsets exposure on the original trade. Combined CVA can therefore **fall**.
<!-- bilingual-en:end -->
    

这一原理意味着：对于已经有大量交易往来的老客户，交易商因新增交易而增加的CVA可能小于新客户（无抵押、双边清算情形） 。因此老客户往往可获得更优惠的价格，因为交易商考虑到和该客户的已有组合，新交易带来的信用风险增量较小 。（在有集中清算的情况下，各交易独立承担清算所信用风险，上述优惠现象不适用 。）
<!-- bilingual-en:start -->
For an established client with many trades in the same unsecured bilateral netting set, a new trade may add less CVA than the same trade with a new client because existing positions provide offsets. This can support a better price for the established client. The logic does not apply in the same way to centrally cleared trades, where exposure is to the clearing house.
<!-- bilingual-en:end -->

**新增CVA的计算：**在实际计算中，银行通常在进行CVA模拟时**保存所有模拟路径的市场变量和组合价值****。当有新交易加入时，可直接利用已保存的市场情景，对每条模拟路径在对应时间节点为新交易重新定价，得到新交易在各情景下各时点的价值 。然后将此**附加价值**叠加到原组合每条路径的价值上，以更新计算敞口均值$v_i$的变化 。新旧$v_i$之差代入公式$(1-R)\sum q_i \Delta v_i$，即可得到**新增交易导致的CVA增量****。这种方法高效地复用了原有Monte Carlo模拟结果，无需重新完整模拟，仅对新交易进行定价计算即可。
<!-- bilingual-en:start -->
**Incremental CVA:** Banks normally retain the market-factor paths and existing portfolio values from a CVA simulation. When a new trade is proposed, revalue only that trade on the stored paths and dates, add its pathwise value to the existing portfolio, and recompute expected positive exposure. Substituting the change $\Delta v_i$ into $(1-R)\sum_iq_i\Delta v_i$ gives the new trade's incremental CVA. This reuses the original simulation rather than generating every path again.
<!-- bilingual-en:end -->

**举例：**原组合价值取决于黄金价格。在计算CVA的模拟中，第2.5年时某条路径下黄金价$=1572$/盎司，原组合价值=$240$万（无抵押则敞口也为240万），对应当期暴露现值$v_{20}=240$万贴现后约$230$万 。现在加入一笔与黄金相关的新交易：在相同情景下（第545条路径）第2.5年，新交易价值=$-420$万 。则该路径在2.5年时组合总价值由原$240$万变为$-180$万（$240-420=-180$万），敞口降为0（负的敞口按0计），所以新的$v_{20}$也降为0 。相比原先$v_{20}=230$万，新交易使该路径该时点暴露减少了230万 。类似地，可对所有路径和时间点计算新交易对$v_i$的变化均值，进而计算出CVA的减少量 。这种精细分析有助于定价时量化新交易的边际信用成本或收益。
<!-- bilingual-en:start -->
**Example:** Existing portfolio value depends on gold. On path 545 at year 2.5, gold is USD 1,572 per ounce and existing portfolio value is 2.40 million in the source's units; discounted exposure $v_{20}$ is approximately 2.30 million. A new gold trade is worth −4.20 million on the same path and date. Combined value becomes −1.80 million, so positive exposure falls to zero. Relative to the original 2.30 million discounted exposure, this path-date exposure falls by 2.30 million. Averaging such changes across every path and date yields the change in $v_i$ and hence the CVA reduction. This is how marginal counterparty-credit cost or benefit enters the new trade's price.
<!-- bilingual-en:end -->

## 8. CVA的市场风险（CVA Risk）及希腊值，Basel III对CVA风险资本要求
<!-- bilingual-en:start -->
*8. CVA Market Risk, Greeks, and Basel III Capital Requirements*
<!-- bilingual-en:end -->

CVA本身取决于市场风险因素和信用风险因素，因此具有显著的**[[市场风险、Greeks 与动态对冲|市场风险]]**属性，可以被看作一种衍生产品 。事实上，任何一个交易对手的CVA都比与该对手交易的任一具体衍生产品更复杂，因为CVA涉及该对手下所有交易的净风险敞口综合 。CVA随市场变化而波动，例如基础市场利率、汇率、商品价格变化会影响敞口$v_i$的大小，信用利差变化会影响违约概率$q_i$，从而引起CVA价值的变动 。
<!-- bilingual-en:start -->
CVA depends on both market and credit factors and therefore has substantial **[[市场风险、Greeks 与动态对冲|market risk]]** of its own. It can be treated as a derivative on the entire counterparty portfolio rather than on one trade. Interest rates, exchange rates, and commodity prices change exposure $v_i$, while the counterparty's credit spread changes default probabilities $q_i$; both channels move CVA.
<!-- bilingual-en:end -->

与传统衍生品类似，我们可以定义CVA对各种风险因子的敏感度（**希腊值**）。例如，CVA对利率的**[[市场风险、Greeks 与动态对冲|Delta]]**衡量利率变动引起的CVA变化，对汇率、大宗商品价格的Delta衡量对应市场价格变动对CVA的影响；CVA对信用利差的敏感度类似于**信用Vega**，因为违约概率$q_i$由信用利差曲线决定，利差平移将影响CVA 。在实际管理中，一些先进银行会对主要对手CVA进行风险因素分解，计算Delta、[[市场风险、Greeks 与动态对冲|Gamma]]、Vega等，以用于对冲和风险控制 。
<!-- bilingual-en:start -->
As with other derivatives, CVA sensitivities can be expressed as Greeks. Interest-rate, foreign-exchange, and commodity **delta** measure how exposure-driven CVA changes when the relevant market moves. Credit-spread sensitivity measures how changes in the counterparty's spread curve alter $q_i$ and hence CVA. Banks may decompose material CVA positions into delta, gamma, vega, and related factors for hedging and risk control.
<!-- bilingual-en:end -->

**[[巴塞尔银行资本与流动性监管|Basel]] III监管资本要求：**鉴于金融危机中CVA波动造成银行损失，巴塞尔委员会在2010年发布的Basel III框架中引入了**CVA风险资本**要求 。高级法（Advanced Approach）规定银行需计提因信用利差平行移动导致的CVA价值变动的市场风险资本 。也就是说，要考虑对手方信用利差上涨/下跌对CVA的冲击，将其纳入市场风险VaR和增量风险计算。然而，对于决定敞口$v_i$变化的其他市场风险因素（如利率、汇率等），Basel III **并未要求**额外资本，因为监管认为这些已在交易组合本身的市场风险资本中覆盖 。
<!-- bilingual-en:start -->
**[[巴塞尔银行资本与流动性监管|Basel]] III CVA capital:** Following crisis losses caused by CVA volatility, the 2010 Basel III framework introduced a CVA risk-capital charge. The advanced approach described in the course required capital for CVA changes driven by parallel movements in counterparty credit spreads, bringing that component into market-risk VaR and incremental-risk calculations. The course source states that other market factors affecting exposure $v_i$, such as rates and exchange rates, did not receive a separate CVA charge because supervisors treated them as covered by the trading portfolio's ordinary market-risk capital.
<!-- bilingual-en:end -->

此规定引发了一些业内争议。某些银行已经开发模型量化了CVA对各种市场因素的敏感性并进行对冲，但按照Basel规则，仅对信用利差的CVA风险给予资本要求，而忽略了对$v_i$的风险敞口 。如果银行去**对冲CVA因子风险**（比如对冲利率、汇率对敞口的影响），反而会增加市场风险资本占用 。原因在于：开展对冲交易会计入交易账簿常规风险资本，但这些对冲降低CVA波动的效果在CVA资本计提中未被认可（因为监管不考虑非信用因素导致的CVA风险） 。因此，有银行抱怨Basel规则**不鼓励全面对冲CVA**，甚至可能产生**额外资本惩罚**。这一矛盾也促使监管机构和业界不断探讨更完善的CVA风险资本计量方法。
<!-- bilingual-en:start -->
This treatment was controversial. Banks could model and hedge CVA sensitivity to rates, exchange rates, and other factors that determine $v_i$, but the course's Basel framework recognised mainly credit-spread-driven CVA risk. A market hedge could therefore add ordinary trading-book capital without receiving an offsetting reduction in the CVA capital charge. Banks argued that this discouraged comprehensive CVA hedging and could create an additional capital penalty, motivating later efforts to refine the framework.
<!-- bilingual-en:end -->

## 9. 错向风险（Wrong-Way Risk）与正向风险（Right-Way Risk）
<!-- bilingual-en:start -->
*9. Wrong-Way Risk and Right-Way Risk*
<!-- bilingual-en:end -->

计算CVA时通常假设违约概率$q_i$与敞口$v_i$独立，但在某些情况下二者可能具有相关性，从而出现**错向风险**或**正向风险**。定义如下：
<!-- bilingual-en:start -->
The simple CVA formula often assumes that default probability $q_i$ and exposure $v_i$ are independent. When they are related, the portfolio exhibits either **wrong-way risk** or **right-way risk**:
<!-- bilingual-en:end -->

- **错向风险（Wrong-Way Risk）**：当交易商对某对手方的**风险敞口较高**时，该对手违约的可能性也**更大**；反之敞口低时对手违约可能性低。即违约概率$q_i$与净敞口$v_i$呈**正相关**，这种不利情形称为错向风险 。
<!-- bilingual-en:start -->
- **Wrong-way risk:** The counterparty is more likely to default precisely when the dealer's exposure to it is high. Default probability $q_i$ and net exposure $v_i$ are positively associated, worsening loss.
<!-- bilingual-en:end -->
    
- **正向风险（Right-Way Risk）**：与上述相反，当交易商对对手的敞口高时，对手违约概率反而更**低**；而当敞口低（甚至交易商欠对手方钱）时，对手违约概率可能更**高**。即$q_i$与$v_i$呈**负相关**，此现象称为正向风险 。
<!-- bilingual-en:start -->
- **Right-way risk:** The counterparty is less likely to default when the dealer's exposure is high, and may be more likely to default when the dealer has little or no positive exposure. Default probability $q_i$ and exposure $v_i$ are negatively associated.
<!-- bilingual-en:end -->
    

错向风险情形下，最糟糕的情况（对手违约）往往发生在本行敞口大的时候，导致损失可能远超独立假设下的估计；正向风险则是一种有利相关，可部分缓解信用损失。
<!-- bilingual-en:start -->
Under wrong-way risk, counterparty default is most likely when the bank's exposure is large, so losses can greatly exceed an independence-based estimate. Right-way risk is the favourable opposite relationship and can reduce expected credit loss.
<!-- bilingual-en:end -->

**错向风险示例：**某对手方透过信用违约掉期（CDS）向银行**卖出**某高风险债券的违约保护（类似AIG销售巨额CDS合约的情形）。如果参考债券的信用状况恶化（信用利差上升），该CDS合约对银行而言价值增加（因为银行可能获得赔付的概率和额度提高），银行的正向敞口上升；但与此同时，由于参考债务违约风险上升通常也会降低卖保护方自身的信用，导致对手方本身违约概率上升 。此时敞口增大伴随对手违约概率增大，属于典型的**错向风险**。简而言之，银行“指望”对手方在市场不利时赔付，但越是不利对手越可能倒下，造成双重打击。
<!-- bilingual-en:start -->
**Wrong-way-risk example:** A counterparty sells the bank CDS protection on a risky reference bond, as in the broad AIG-type pattern. If the reference credit deteriorates, the CDS becomes more valuable to the bank and its positive exposure rises. The same shock can weaken the protection seller and raise its own default probability. The bank most needs payment exactly when the counterparty is least able to make it: exposure and counterparty default risk rise together.
<!-- bilingual-en:end -->

**正向风险示例：**在某些结构中对手方从银行**买入**违约保护（银行卖出保险）。如果参考债券信用利差上升，银行作为卖方将面临潜在赔付（合约价值对银行变负，敞口从银行角度降低甚至为零），同时对手方违约概率上升 。此时对银行来说，当对手方风险变大时反而**没有敞口**（或敞口很小，因合约价值为负是银行欠钱），即属于正向风险关系。又例如，若某公司与银行进行了大量相似交易，一个新的类似交易会引入错向风险：因为如果市场走势对公司不利，公司在所有类似交易上都亏损严重，财务恶化违约概率大增 。反之，如果公司进行交易是为了**对冲其自身风险**，则在交易价值对其不利时，公司在未对冲部分可能有收益，从而财务状况相对改善，违约可能性下降——这便是正向风险 。
<!-- bilingual-en:start -->
**Right-way-risk example:** A counterparty buys default protection from the bank. When the reference spread rises, the contract becomes a liability to the bank, so the bank's positive exposure to that counterparty falls toward zero even if the counterparty's own default probability rises. More generally, repeated unhedged trades in the same direction can create wrong-way risk by weakening the client when it owes the bank most. A trade that hedges the client's underlying business can instead create right-way risk: loss on the trade is offset by gain elsewhere in the client's business, improving its ability to perform when bank exposure is high.
<!-- bilingual-en:end -->

面对错向风险，定量化和缓释是难点之一。Basel II监管为弥补模型未覆盖的相关性风险，引入了**$\alpha$因子**（[[CAPM、系统风险与资本成本|Alpha]] Multiplier）放大CVA结果 。标准规定未考虑错向风险时计算的CVA要乘以$\alpha$系数（Basel II设定$\alpha=1.4$，银行若有自建模型可申请使用$\alpha$最低降至1.2） 。很多银行自己也会估计一个$\alpha$以反映组合的错向风险程度，通常在1.07～1.10之间 。此外，一些市场参与者开发了专门模型来直接建模$v_i$与$q_i$的相关性，从而更精确地刻画错向风险 。总体来说，识别并管理错向风险（如通过限额控制特定高度相关的交易、追加保证金要求等）是CVA管理中至关重要的一环。
<!-- bilingual-en:start -->
Wrong-way risk is difficult to quantify and mitigate. The Basel II framework introduced an **$\alpha$ multiplier** to increase CVA or exposure results for correlation not captured in the model. The source states a supervisory value of $\alpha=1.4$, with an approved internal-model floor of 1.2. It also reports some banks' own portfolio estimates around 1.07–1.10. More direct approaches model dependence between $v_i$ and $q_i$. Limits on highly dependent trades and stronger margin terms are practical mitigants.
<!-- bilingual-en:end -->

## 10. DVA的会计处理与争议
<!-- bilingual-en:start -->
*10. Accounting Treatment of DVA and the Controversy Around It*
<!-- bilingual-en:end -->

**DVA（Debit Value Adjustment）**代表交易商自身违约风险对衍生品定价的调整。会计准则允许在衍生品估值中**加入DVA**，即交易商将自身成为违约方的可能性视为对自己有利的因素：无违约情况下的理论价值 **减去CVA** 再 **加上DVA** 得出衍生品的账面价值 。DVA的大小等于交易对手为交易商违约所要求的补偿的现值，直观理解为：“如果我将来可能欠你钱但我破产不还，那对我现在就是收益”。
<!-- bilingual-en:start -->
**DVA (debit valuation adjustment)** adjusts a derivative for the dealer's own default risk. In the bilateral valuation convention used here, book value equals clean value minus CVA plus DVA. DVA represents the present value of the amount the dealer may not have to pay if it defaults while owing the counterparty. Put plainly: "If I might owe you money in the future but fail to pay in default, that possibility reduces the value of my liability today."
<!-- bilingual-en:end -->

这种**将自身违约计入利润**的处理引发了广泛争议：
<!-- bilingual-en:start -->
Recognising deterioration in one's own credit as a gain has generated two main objections:
<!-- bilingual-en:end -->

- **争议1:** 除非交易商真的违约，否则账面上的DVA收益无法锁定兑现 。例如银行由于信用恶化记入了一笔DVA收益，但只有当银行实质性违约逃废债务时，这笔收益才成为现实；若银行信用后来改善，之前确认的DVA收益还可能转回为损失。因此DVA收益对企业而言并没有真正可支配的经济价值。
<!-- bilingual-en:start -->
- **Controversy 1:** a DVA gain cannot normally be monetised while the dealer remains a going concern. It becomes economically realised only through default and non-payment. If credit quality later improves, the earlier gain can reverse into a loss. The firm therefore cannot treat DVA as freely available economic value.
<!-- bilingual-en:end -->
    
- **争议2:** DVA机制导致信用状况变差的公司账面利润反而上升，极具讽刺意味 。当衍生品交易商自身信用利差扩大（违约可能性上升）时，按照会计准则DVA增加，从而直接计入当期利润。这意味着公司的信用风险提高了，财务报表却出现盈利，混淆了利润信号，也可能削弱市场对财报的信任。
<!-- bilingual-en:start -->
- **Controversy 2:** A deterioration in the dealer's own credit can increase reported profit. When its credit spread widens and default becomes more likely, DVA rises and the accounting gain enters current earnings. Credit risk has **increased**, yet the financial statements show a gain, which obscures the economic signal and can weaken confidence in reported profit.
<!-- bilingual-en:end -->
    

鉴于上述问题，监管机构在资本要求中做了调整。巴塞尔协议提出在计算监管资本时，应当从核心资本中**扣除**DVA所带来的未实现收益 。简单说，**DVA增益不计入核心一级资本**，以防止银行通过自身信用恶化“平滑”利润或提振资本充足率 。这一做法承认了DVA收益的不可靠性。
<!-- bilingual-en:start -->
Because of these problems, the Basel framework requires unrealised DVA gains to be **deducted from regulatory capital**. In other words, a bank cannot use gains created by deterioration in its own credit to increase Common Equity Tier 1 capital or its reported capital ratio.
<!-- bilingual-en:end -->

目前业界对DVA的会计处理仍有不同观点。一些人士建议干脆不将自身违约计入衍生品日常估值，而仅在负债清偿时处理；也有人认为应引入“双边CVA”概念同时考虑双方违约。尽管如此，DVA在现行会计准则下仍是一项要求计量披露的内容，风险管理中则通常将DVA视为需剔除的指标，以更真实地反映交易的经济价值。
<!-- bilingual-en:start -->
Views on DVA remain divided. Some argue that own-default risk should be excluded from routine derivative valuation and recognised only when liabilities are settled; others favour bilateral valuation that explicitly allows either party to default. Under the accounting treatment described here, DVA must still be measured and disclosed, while risk managers often remove it when assessing the transaction's underlying economic value.
<!-- bilingual-en:end -->

## 11. 不同衍生品在CVA中的表现与计算差异
<!-- bilingual-en:start -->
*11. How Exposure Profiles Differ Across Derivative Types*
<!-- bilingual-en:end -->

各种衍生品的交易结构不同，导致其风险敞口随时间的分布特点不同，对CVA的贡献也有所差异：
<!-- bilingual-en:start -->
Different derivative structures produce different exposure profiles over time and therefore contribute differently to CVA.
<!-- bilingual-en:end -->

- **利率互换（IRS）：**通常本金不交换，仅交换利息差额。利率互换的预期风险敞口相对**较小且平稳**，通常在中期期限达到峰值 。原因是互换在起初时价值接近零，此后随着利率曲线变动慢慢累积敞口，一般在合约中段利率累积差异最大，从而暴露最大，然后逐渐降低。总体而言，利率互换因不涉及名义本金交换，违约时潜在损失主要是未来利息差的现值，远小于直接借贷本金金额。
<!-- bilingual-en:start -->
- **Interest-rate swap (IRS):** principal is normally not exchanged; only net interest payments change hands. The swap begins near zero value, exposure builds as rates move, often peaks around the middle of its life, and then declines as remaining cash flows run off. Because notional principal is not exchanged, default exposure is usually much smaller than the principal exposure on a loan.
<!-- bilingual-en:end -->
    
- **货币互换（Cross Currency Swap）：**涉及两种货币本金的交换，在到期日双方要互换名义本金。由于**末期要交换本金且汇率存在不确定性**，货币互换在到期时可能出现**巨大的敞口** 。因此货币互换的对手违约风险影响显著大于利率互换 。一般来说，货币互换敞口曲线在接近合约末期急剧上升（因为累积的利息差和最终本金交换风险并存），使得CVA计算时远期部分的贡献较大。
<!-- bilingual-en:start -->
- **Cross-currency swap:** the parties exchange principal in different currencies, including a final re-exchange of notional amounts. Exchange-rate uncertainty and the terminal principal exchange can create **large exposure near maturity**, so the later part of the exposure profile may contribute substantially to CVA.
<!-- bilingual-en:end -->
    
- **远期合约（Forward）：**例如远期外汇、远期商品交易，在到期时按约定价格交易标的资产。本金不在合同期内交换，但**最终交割时**根据市场价与合约价差结算盈亏，因此敞口多集中在**合约末期**。远期的风险敞口可以近似看作一个**期权**：因为只有当未来市场价偏向对交易商有利的一侧时才产生正向敞口，否则为零 。数学上，远期合约在某中间时点$t$的正向敞口的现值，相当于一份到$t$时行权价为合约价$K$的看涨期权的价格 。例如，对一个到期价$K$的多头远期，多头在$t$时的敞口为$\max(F_t - K,,0)$，其期望现值可用Black模型计算。这种期权特性意味着远期的预期敞口可用已知公式求解，从而简化CVA计算。
<!-- bilingual-en:start -->
- **Forward contract:** an FX or commodity forward settles at the agreed delivery price at maturity, so exposure is concentrated toward the end of the contract. Positive exposure has an option-like form because only favourable market moves count: for a long forward, exposure at time $t$ is $\max(F_t - K,,0)$. The expected present value of this positive part can be estimated with an option-pricing model such as Black's model, simplifying the exposure input to CVA.
<!-- bilingual-en:end -->
    

>[!example] 示例
> **假设某银行与矿业公司签订一笔2年期黄金远期合约，约定2年后按$1500$/盎司价格由银行买入100万盎司黄金。当前2年期黄金远期价格$F_0=1600$/盎司，矿业公司的违约概率：第1年2%，第2年3%（违约假设发生在每年年中），无风险利率5%，预期回收率30%。据此可计算远期合约的CVA及信用调整价值：
><!-- bilingual-en:start -->
>**Suppose a bank enters a two-year gold forward with a mining company and agrees to buy one million ounces at USD 1,500 per ounce at maturity. The current two-year forward price is $F_0=1600$ per ounce. The mining company's unconditional default probabilities are 2% in year one and 3% in year two, with default assumed at each year's midpoint. The risk-free rate is 5% and expected recovery is 30%. These inputs can be used to calculate the forward's CVA and credit-adjusted value:**
><!-- bilingual-en:end -->

- **无违约情形下远期合约的公允价值：**银行锁定了低于当前远期价的买入价，有利可图。按无风险计价，合约价值约为$(F_0 - K)e^{-rT} = (1600-1500)e^{-0.05\times2} = 100 \times e^{-0.1} \approx 90.48$（以万为单位则$=9048$万） 。
<!-- bilingual-en:start -->
- **Clean forward value:** The bank locked in a purchase price below the current forward price. Clean value is $(F_0-K)e^{-rT}=(1600-1500)e^{-0.05\times2}\approx90.48$ per ounce. On one million ounces, that is approximately **USD 90.48 million**.
<!-- bilingual-en:end -->
    
- **CVA计算：**根据模拟或公式得到每年中点敞口现值：$v_1 \approx 132.38$，$v_2 \approx 186.65$（单位万） 。套用离散CVA公式，$CVA=(1-0.3)\big(0.02\times 132.38 + 0.03\times 186.65\big) \approx 5.77$（万） 。因此CVA占无违约价值的约6.4%。
<!-- bilingual-en:start -->
- **CVA:** The source gives discounted midpoint exposures $v_1\approx132.38$ million and $v_2\approx186.65$ million. Hence $\text{CVA}=(1-0.3)[0.02(132.38)+0.03(186.65)]\approx5.77$ million, about 6.4% of clean value.
<!-- bilingual-en:end -->
    
- **信用调整后价值：**衍生品考虑对手违约后的价值 = 无违约价值 - CVA = $90.48 - 5.77 = 84.71$（万） 。这意味着由于对手方违约风险，远期合约对银行的价值减少了约$5.77$万。
<!-- bilingual-en:start -->
- **Credit-adjusted value:** Clean forward value minus CVA is $90.48-5.77=84.71$ million. Counterparty default risk therefore reduces the bank's value by about USD 5.77 million.
<!-- bilingual-en:end -->
    

从上述示例可见，不同衍生品的CVA计算需要结合其敞口分布特征：利率互换需关注中途敞口，货币互换需特别关注末期本金交换风险，而远期等合约可以利用期权模型估计中间时点敞口。在风险管理中，常绘制不同产品的**暴露曲线**以比较其信用风险轮廓，帮助制定相应的信用减值策略（如收取初始保证金、设置分段的名义本金交换安排等）。
<!-- bilingual-en:start -->
CVA must reflect each product's exposure profile: an interest-rate swap often peaks mid-life, a cross-currency swap can carry substantial terminal principal risk, and a forward's expected positive exposure can be estimated with an option model. Risk managers compare **exposure curves** across products and choose mitigants such as initial margin or staged exchanges of notional principal.
<!-- bilingual-en:end -->

---

## 模拟考试题
<!-- bilingual-en:start -->
*Practice Questions*
<!-- bilingual-en:end -->

为了检验对上述知识的掌握，以下提供几道模拟考题，并附上详尽解析：
<!-- bilingual-en:start -->
The following practice questions test the material covered above and include detailed solutions:
<!-- bilingual-en:end -->

**问题1：CVA计算** – 某银行与一对手方有一笔衍生品交易组合，未来两年内可能发生违约。已知该对手方在第1年违约的风险中性概率为5%，在第2年违约的风险中性概率为7%（假设违约只可能在年末发生，各年违约互斥）。若银行估计在对手方违约时的净风险敞口现值分别为：第1年末1,000万元，第2年末800万元。假设回收率$R=40%$。问：该组合的CVA是多少？若组合假定无违约时的公允价值为5,000万元，扣除CVA后的信用调整价值又是多少？
<!-- bilingual-en:start -->
**Question 1: CVA calculation.** A bank has a derivative portfolio with one counterparty. Mutually exclusive risk-neutral default probabilities are 5% at the end of year one and 7% at the end of year two. Discounted net exposures at default are RMB 10 million and RMB 8 million respectively, recovery is $R=40\%$, and clean portfolio value is RMB 50 million. Calculate CVA and credit-adjusted value.
<!-- bilingual-en:end -->

**解析：**CVA计算采用公式$\text{CVA}=(1-R)\sum_i q_i v_i$。题中给定两期违约概率和相应暴露：$q_1=5%=0.05$，$q_2=7%=0.07$；敞口现值$v_1=1000$万，$v_2=800$万；回收率$R=40%$则损失率$1-R=60%=0.6$。代入公式：
<!-- bilingual-en:start -->
**Analysis:** Use $\text{CVA}=(1-R)\sum_iq_iv_i$. Here $q_1=0.05$, $q_2=0.07$, $v_1=10$ million yuan, $v_2=8$ million yuan, and loss given default is $1-R=0.6$:
<!-- bilingual-en:end -->

CVA=0.6×(0.05×1000+0.07×800)万元=0.6×(50+56)=0.6×106=63.6万元.\text{CVA} = 0.6 \times (0.05 \times 1000 + 0.07 \times 800) \text{万元} = 0.6 \times (50 + 56) = 0.6 \times 106 = 63.6 \text{万元}.
<!-- bilingual-en:start -->
$\text{CVA}=0.6[0.05(10)+0.07(8)]\text{ million yuan}=0.636\text{ million yuan}$, or RMB 636,000.
<!-- bilingual-en:end -->

因此CVA约为63.6万元。无违约价值5,000万扣除CVA后，衍生品的信用调整后价值$=5000 - 63.6 = 4936.4$万元。CVA越高，表明交易对手信用风险越大、对价值的侵蚀越多。
<!-- bilingual-en:start -->
CVA is therefore RMB 636,000. Deducting it from clean value of RMB 50 million gives credit-adjusted value of RMB 49.364 million. A higher CVA means greater erosion of value by counterparty credit risk.
<!-- bilingual-en:end -->

**答案要点：**CVA = 63.6万元；信用调整后价值 ≈ 4936.4万元。
<!-- bilingual-en:start -->
**Answer:** CVA = RMB 636,000; credit-adjusted value $\approx$ RMB 49.364 million.
<!-- bilingual-en:end -->

---

**问题2：峰值敞口的概念与计算** – 请解释**峰值敞口（Peak Exposure）**和**最大峰值敞口（Maximum Peak Exposure）**的定义。假设某银行通过1万次模拟得到了未来某日期$t$的敞口分布，并计算得该日97.5%分位数的暴露为200万美元。如果在整个模拟期间各时点的97.5%峰值中最大值为250万美元，请问200万和250万分别代表什么含义？在CVA计算中是否使用峰值敞口？
<!-- bilingual-en:start -->
**Question 2: Peak exposure.** Define peak exposure and maximum peak exposure. A bank runs 10,000 simulations. Exposure at date $t$ has a 97.5th percentile of USD 2 million, while the largest 97.5th-percentile exposure across all simulated dates is USD 2.5 million. What does each number mean, and is peak exposure used directly in CVA?
<!-- bilingual-en:end -->

**解析：**峰值敞口是指在给定置信水平下某时点可能出现的**高端风险敞口**。具体来说，97.5%分位峰值敞口200万美元表示：根据模拟，有97.5%的情景下$t$时刻银行对该对手的敞口不超过200万；换言之，在**极端不利**的2.5%情景中，敞口可能达到200万或更高。最大峰值敞口则是把各未来时刻的峰值敞口进行比较，取其中**数值最大**者。例如250万意味着在所有时间点里，某一时刻的97.5%分位暴露达到250万，这是银行在高置信水平下可能面对的**最严重暴露**。
<!-- bilingual-en:start -->
**Analysis:** USD 2 million is the 97.5th percentile at the specified date: 97.5% of simulated exposures are no greater, while 2.5% are above it. Maximum peak exposure compares this percentile across dates; USD 2.5 million is the largest such value anywhere on the future exposure profile.
<!-- bilingual-en:end -->

CVA的计算**不直接使用**峰值敞口。CVA基于各时点**平均预期敞口**（风险中性下的期望值）加权违约概率求和得到，是一种**期望损失**度量。峰值敞口属于**极值风险指标**，用于监管和内部风险管理，反映**尾部情景**下的潜在敞口，常用于设定风险限额或计算经济资本，不会直接加总入CVA。需注意峰值敞口模拟通常采用实际分布（现实世界情景），而CVA采用风险中性假设模拟。两者用途不同：前者侧重高置信损失管理，后者侧重定价减值。不过，了解组合的峰值敞口有助于判断CVA模型假设下是否存在显著尾部风险未被充分捕捉，并可辅助制定额外的风险缓释措施。
<!-- bilingual-en:start -->
CVA does **not** directly use peak exposure. It weights **expected positive exposure** at each date under a risk-neutral distribution by default probability, producing expected loss. Peak exposure is a tail-risk measure used for limits, stress analysis, or economic capital and is generally simulated under a real-world or scenario distribution. It can reveal tail vulnerability that an expected-loss measure does not show, but it is not added into CVA.
<!-- bilingual-en:end -->

**答案要点：**峰值敞口是在高置信水平下某时点可能达到的高端敞口（如97.5%分位对应有2.5%概率敞口超出该值）；最大峰值敞口是所有时点峰值中的最大者。在题给例子中，200万美元是特定时刻97.5%分位敞口，250万美元是全时期内最高的97.5%分位敞口。CVA计算用的是预期敞口，不直接用峰值敞口，但峰值敞口用于衡量极端风险。
<!-- bilingual-en:start -->
**Answer:** USD 2 million is the 97.5th-percentile exposure at one date; USD 2.5 million is the maximum of those date-specific percentiles. CVA uses expected exposure, while peak measures extreme exposure for risk management.
<!-- bilingual-en:end -->

---

**问题3：错向风险识别** – 以下场景哪种属于**错向风险**（Wrong-Way Risk）？哪种属于正向风险（Right-Way Risk）？请简要解释理由。
<!-- bilingual-en:start -->
**Question 3: Identify wrong-way and right-way risk.** Which of Scenarios A and B below is wrong-way risk, and which is right-way risk? Explain why.
<!-- bilingual-en:end -->

**场景A：**银行从对手方买入一种债券的违约保护（对手方卖出CDS合约给银行）。如果债券发行人信用恶化，该合约将增值使银行有较大正敞口，但债券发行人的违约也可能拖累对手方的财务，增加对手方自身违约概率。
<!-- bilingual-en:start -->
**Scenario A:** The bank buys CDS protection from a counterparty on a bond issued by a third party. If the reference issuer's credit deteriorates, the CDS becomes more valuable to the bank and its positive exposure rises. The same deterioration may also weaken the protection seller and increase that counterparty's own probability of default.
<!-- bilingual-en:end -->

**场景B：**银行为某商品贸易公司提供远期汇率对冲。当汇率波动对贸易公司不利时，公司在远期上的损失会被其现货业务的额外利润部分弥补，从而公司反而更有能力履约。
<!-- bilingual-en:start -->
**Scenario B:** A bank provides an FX forward hedge to a commodity-trading company. A currency move that produces a loss on the forward is partly offset by additional profit in the company's underlying spot business, strengthening its capacity to meet the forward obligation.
<!-- bilingual-en:end -->

**解析：**
<!-- bilingual-en:start -->
**Analysis:**
<!-- bilingual-en:end -->

- **场景A**描述了经典的错向风险。银行的敞口（对手履约义务的价值）在标的信用恶化时变大，但此时对手方违约概率也升高，因为标的债券违约往往伴随卖保护方（如AIG案例）必须大额赔付、财务恶化 。敞口与违约概率同向上升，属于**错向风险**。银行最担心的情况正是债券违约导致自己敞口巨增，而对手方也因赔付义务沉重可能自身违约，银行无法得到赔付。
<!-- bilingual-en:start -->
- **Scenario A is wrong-way risk.** The bank's exposure rises as the reference credit deteriorates, while the protection seller's own default probability can rise because it faces larger CDS obligations and financial stress. Exposure and counterparty default probability therefore rise together, and the counterparty may fail when payment is most valuable to the bank.
<!-- bilingual-en:end -->
    
- **场景B**体现了正向风险的特征。贸易公司用远期合约对冲其现货业务风险：如果汇率走势导致公司在远期上亏损（银行敞口上升，因为公司欠银行款增多），则公司现货业务可能因相反的汇率变动受益，获得额外利润 。这额外收益增强了公司的财务状况，降低了其违约概率 。也就是说，当银行的信用敞口升高时，对手方反而更不易违约，这是**正向风险**的情况。
<!-- bilingual-en:start -->
- **Scenario B is right-way risk.** The trading company uses the forward to hedge its underlying business. A currency move that creates a loss on the forward, and hence greater exposure for the bank, may create an offsetting profit in the client's spot business. That profit strengthens the client and lowers its default probability when the bank's exposure is high.
<!-- bilingual-en:end -->
    

因此，场景A为错向风险，场景B为正向风险。
<!-- bilingual-en:start -->
Scenario A is therefore wrong-way risk, while Scenario B is right-way risk.
<!-- bilingual-en:end -->

**答案要点：**场景A错向风险（敞口大时对手违约概率也大，同方向变动）；场景B正向风险（敞口大时对手反而更稳健，违约概率下降，反方向相关）。
<!-- bilingual-en:start -->
**Key points:**<br>
Scenario A: **wrong-way risk**—exposure and counterparty default probability rise together.<br>
Scenario B: **right-way risk**—high exposure coincides with a stronger counterparty and lower default probability.
<!-- bilingual-en:end -->

---

**问题4：DVA的含义及会计争议** – 什么是DVA？某银行信用利差突然上升意味着其违约概率提高，从而DVA增大。请问这对银行当期利润有何影响？为何这一结果存在争议？监管对此有什么应对措施？
<!-- bilingual-en:start -->
**Question 4: Meaning of DVA and the accounting controversy.** What is DVA? If a bank's credit spread rises sharply, its default probability and DVA increase. How does this affect current profit, why is the result controversial, and how does regulation respond?
<!-- bilingual-en:end -->

**解析：** **DVA（Debit Value Adjustment）**是交易商自身违约风险带来的价值调整，数值上等于交易对手因银行可能违约而要求补偿的预期损失。对银行而言，DVA体现为一项“收益”调整：信用越差，未来不还款的可能性越高，当前负债的公允价值对银行而言越低，等效于银行获得收益。因此，当银行信用利差上升（违约风险加大）时，按照市值计价原则，银行衍生品负债的公允价值下降，DVA增加，这部分增加计入当期**利润**，会**提升**银行账面盈利 。
<!-- bilingual-en:start -->
**Analysis:** **DVA (debit valuation adjustment)** reflects the bank's own default risk. It is the expected loss that the counterparty would suffer if the bank defaulted while owing money. From the bank's perspective, a greater chance of non-payment reduces the fair value of its derivative liabilities. A wider bank credit spread therefore raises DVA and, under fair-value accounting, can create a current-period **gain** that **increases** reported profit.
<!-- bilingual-en:end -->

这一结果极具争议，因为它违背常识：银行财务状况恶化（信用变差）居然带来账面利润。首先，这种利润只是**纸面收益**，除非银行真的违约，否则DVA收益无法变现 ；其次，这可能向市场传递错误信号，粉饰公司的实际状况（信用风险上升本应是负面事件，却反映为正面盈利） 。因此不少分析师和监管者对此提出质疑，担心银行利用提升DVA来调节利润，或者投资者被迷惑。
<!-- bilingual-en:start -->
The result is controversial because worsening financial health produces reported profit. The gain is largely a **paper gain** that cannot normally be realised unless the bank actually defaults, and it may mislead investors by presenting increased credit risk as positive earnings. Analysts and regulators therefore worry that DVA obscures the bank's true condition and could be used to manage reported profit.
<!-- bilingual-en:end -->

监管机构的应对是在资本规则中**扣除DVA收益**。Basel III 要求银行在计算核心一级资本时，将DVA带来的未实现收益从盈余中扣减 ，避免其增厚资本。这意味着即使会计上确认了DVA利润，也不能用于满足监管资本要求。此举确保银行不能通过自身信用恶化来提高资本充足率，维护了资本指标的可靠性 。一些会计准则制定者也在考虑调整DVA的处理方式。但目前，DVA仍需计入利润表，只是监管上不予承认其对资本的积极贡献。
<!-- bilingual-en:start -->
Regulators respond by excluding the gain from capital. Under Basel III, unrealised DVA gains are deducted when Common Equity Tier 1 capital is calculated. A bank may recognise DVA in accounting profit, but cannot use deterioration in its own credit to satisfy regulatory capital requirements or improve its capital ratio.
<!-- bilingual-en:end -->

**答案要点：**DVA是自身违约风险的价值调整，信用变差时DVA升高会增加账面利润。这一利润无法真正实现且扭曲财报，因此有争议。监管要求扣除DVA收益在资本计算中的作用，以防信用恶化“虚增”资本。
<!-- bilingual-en:start -->
**Key points:** DVA is the valuation effect of the bank's own default risk. Credit deterioration can increase DVA and reported profit, even though the gain is not normally realisable and can distort the economic signal. Regulation therefore removes DVA gains from regulatory-capital calculations.
<!-- bilingual-en:end -->

##  20-3
>[!question] 
一家银行与某矿业公司签订 2 年期黄金远期合约，约定到期时银行以 1 500 美元/盎司买入 100 万盎司黄金。  
> - 当期 2 年期远期价 $F_0 = 1 600$ 美元/盎司  
> - 黄金对数波动率 $\sigma = 20\%$  
> - 无风险连续利率 $r = 5\%$  
> - 矿业公司 1 年内（年中点）无条件违约概率 $q_1 = 2\%$；第 2 年（年中点）无条件违约概率 $q_2 = 3\%$  
> - 违约回收率 $R = 30\%$  
> - 若违约发生于当年中点（距到期 0.5 年或 1.5 年），合约按 **正市值** 现金结算
> 求：  
> 1.   在两可能违约时点的 **正向曝险** $v_1, v_2$  
> 2.   信用估值调整 $\text{CVA}$  
> 3.   考虑信用风险后的远期合约价值  
<!-- bilingual-en:start -->
A bank enters a two-year gold forward with a mining company, agreeing to buy one million ounces at USD 1,500 per ounce at maturity.
- Current two-year forward price: $F_0=1600$ per ounce
- Gold log-price volatility: $\sigma=20\%$
- Continuously compounded risk-free rate: $r=5\%$
- Unconditional mining-company default probabilities: $q_1=2\%$ at the midpoint of year one and $q_2=3\%$ at the midpoint of year two
- Recovery rate: $R=30\%$
- If default occurs at a midpoint, 0.5 or 1.5 years from inception, the contract is cash-settled on its **positive market value**.

Find: (1) positive exposures $v_1$ and $v_2$ at the two possible default times; (2) the credit valuation adjustment; and (3) the forward's value after counterparty-credit adjustment.
<!-- bilingual-en:end -->

1  计算正向曝险
<!-- bilingual-en:start -->

&nbsp;
**1.** Calculate Positive Exposure<br>
<!-- bilingual-en:end -->

远期在违约时 $t_i$（距到期 $\tau_i = T-t_i$）的市值  
$$v_i = e^{-r\tau_i}\bigl[F_0\,N(d_{1,i}) - K\,N(d_{2,i})\bigr]$$  
$$d_{1,i}= \frac{\ln(F_0/K) + \tfrac12\sigma^{2}\tau_i}{\sigma\sqrt{\tau_i}},\qquad  
d_{2,i}=d_{1,i}-\sigma\sqrt{\tau_i}$$  
<!-- bilingual-en:start -->
At default time $t_i$, with $\tau_i=T-t_i$ remaining to maturity, the forward's positive market value is calculated as follows.
<!-- bilingual-en:end -->

| 违约时点 | $\tau_i$ | $d_{1,i}$ | $d_{2,i}$ | $v_i$ (USD m) |
|-----------|---------|-----------|-----------|---------------|
| 年 1 中点 | 0.5 | 0.527 | 0.386 | 132.38 |
| 年 2 中点 | 1.5 | 0.386 | 0.141 | 186.65 |
<!-- bilingual-en:start -->
| Default node | $\tau_i$ | $d_{1, i}$ | $d_{2, i}$ | $v_i$ (USD m) |
| --- | ---: | ---: | ---: | ---: |
| Midpoint of year 1 | 0.5 | 0.527 | 0.386 | 132.38 |
| Midpoint of year 2 | 1.5 | 0.386 | 0.141 | 186.65 |
<!-- bilingual-en:end -->

（单位：合约总名义，以百万美元计）
<!-- bilingual-en:start -->
Units are USD millions for the contract's full notional.
<!-- bilingual-en:end -->

2  计算 CVA  
<!-- bilingual-en:start -->

&nbsp;
**2.** Calculate CVA<br>
<!-- bilingual-en:end -->

$$(1-R)=0.70$$ 
$$\text{CVA}=(1-R)\bigl(q_1v_1+q_2v_2\bigr)  
           =0.70\bigl(0.02\times132.38 + 0.03\times186.65\bigr)\approx 5.77\,\text{m}$$ 
3  信用风险调整后的远期价值  
<!-- bilingual-en:start -->

&nbsp;
**3.** Calculate the Counterparty-Credit-Adjusted Forward Value<br>
<!-- bilingual-en:end -->

忽略信用风险的理论价值  
$$V_0 =(F_0-K)e^{-rT} =(1 600-1 500)e^{-0.05\times2}\approx 90.48\,\text{m}$$
计入 CVA 后  
$$V_{\text{adjusted}} = V_0 - \text{CVA} \approx 90.48 - 5.77 = 84.71\,\text{m}$$  
**要点总结**  
1.   先用 Black-Scholes–Merton 正向合约定价公式在每个潜在违约时点求正向曝险 $v_i$；  
2.   采用无条件违约概率 $q_i$ 与 $(1-R)$ 计算 $\text{CVA}$；  
3.   远期合约基准价值减去 $\text{CVA}$ 得到考虑对手信用风险后的公允价值。  
<!-- bilingual-en:start -->
First calculate clean value; then include CVA.
**Key points**
**1.** Use the Black--Scholes--Merton positive-forward-value formula to calculate $v_i$ at each potential default time.<br>
**2.** Combine unconditional default probability $q_i$ with loss given default $(1-R)$ to calculate CVA.<br>
**3.** Subtract CVA from clean forward value to obtain fair value after counterparty-credit risk.<br>
<!-- bilingual-en:end -->

## 20.13  
>[!question] 
将例 20-3 的计算进行扩展，假定违约可以发生在每个月的中间点。  
第 1 年每个月发生违约的概率为 0.001667，第 2 年每个月发生违约的概率为 0.0025。  
<!-- bilingual-en:start -->
Extend Example 20-3 by allowing default at the midpoint of every month. The unconditional probability of default in each month of year one is 0.001667, and the probability in each month of year two is 0.0025.
<!-- bilingual-en:end -->

>[!question] 
某银行与一家矿业公司签订 2 年期黄金远期合约，约定在到期日第 24 个月末，银行以 1 500 美元/盎司的价格买入 1 000 000 盎司黄金。  
— 当前 2 年期黄金远期价格 $F_0 = 1 600$ 美元/盎司  
— 黄金对数价格波动率 $\sigma = 20\%$（年化，连续复利）  
— 无风险连续复利利率 $r = 5\%$  
— 矿业公司违约回收率 $R = 30\%$  
违约可发生在 **每个月的中点**（即 0.5、1.5、2.5 … 23.5 个月，共 24 个节点）。  
-第 1 年（前 12 个月）每月 **无条件** 违约概率为 0.001667  
-第 2 年（后 12 个月）每月 **无条件** 违约概率为 0.0025  
若违约发生，合约按当时的正市场价值 **现金结算**（卖方向买方支付正向曝险的 70%，因 $(1-R)=70\%$）。  
要求：  
1. 对每个可能违约月 $t_i$，计算远期合约在该节点的正向曝险 $v_i$：   $$v_i = e^{-r\,(T-t_i)}\bigl[F_0\,N(d_{1,i}) - K\,N(d_{2,i})\bigr],\quad  
     d_{1,i} = \frac{\ln(F_0/K)+\tfrac12\sigma^2(T-t_i)}{\sigma\sqrt{T-t_i}},\quad  
     d_{2,i} = d_{1,i} - \sigma\sqrt{T-t_i}$$
2. 计算两年期 **信用估值调整**  
$$\text{CVA} = (1-R)\sum_{i=1}^{24} q_i\,v_i$$ 
   其中 $q_i$ 为对应月份的无条件违约概率。  
3. 给出考虑信用风险后的远期合约公允价值  
   $$V_{\text{adjusted}} = (F_0-K)e^{-rT} - \text{CVA}$$
提示：可将 1 000 000 盎司的名义直接乘入 $v_i$，所有货币单位均以美元计。  

**解答**
一、输入参数  
- 名义数量 `N = 1 000 000 oz`  
- 合约到期 `T = 24` 个月 = 2 年  
- 远期 / 执行价 `F₀ = 1 600`, `K = 1 500` (USD/oz)  
- 波动率 `σ = 20 %`（年化，连续复利）  
- 无风险利率 `r = 5 %`（连续复利）  
- 回收率 `R = 30 %` ⇒ `1‒R = 70 %`  
- 违约节点 `tᵢ = (i-0.5)/12`, `i = 1 … 24`

**无条件违约概率**

$$
q_i =
\begin{cases}
0.001667,& i = 1,\dots,12\$$2pt]
0.002500,& i = 13,\dots,24
\end{cases}
$$

---

 二、逐月正向曝险 $v_i$

$$
\begin{aligned}
d_{1,i} &= \frac{\ln(F_0/K)+\tfrac12\sigma^{2}(T-t_i)}
                {\sigma\sqrt{T-t_i}},\\
d_{2,i} &= d_{1,i}-\sigma\sqrt{T-t_i},\$$6pt]
v_i &= e^{-r\,(T-t_i)}
      \Bigl[F_0\,N(d_{1,i})-K\,N(d_{2,i})\Bigr]\times N .
\end{aligned}
$$

24 个节点结果  

| 月 $i$ | $t_i$ (年) | $v_i$ (mn USD) | $q_i$ | $(1-R)q_i v_i$ (mn USD) |
|:--:|:--:|------:|-------:|--------:|
|  1 | 0.0417 | 205.861 | 0.001667 | 0.240 |
|  2 | 0.1250 | 203.456 | 0.001667 | 0.237 |
|  3 | 0.2083 | 200.955 | 0.001667 | 0.234 |
|  4 | 0.2917 | 198.355 | 0.001667 | 0.231 |
|  5 | 0.3750 | 195.647 | 0.001667 | 0.228 |
|  6 | 0.4583 | 192.816 | 0.001667 | 0.225 |
|  7 | 0.5417 | 189.846 | 0.001667 | 0.222 |
|  8 | 0.6250 | 186.718 | 0.001667 | 0.219 |
|  9 | 0.7083 | 183.411 | 0.001667 | 0.215 |
| 10 | 0.7917 | 179.899 | 0.001667 | 0.211 |
| 11 | 0.8750 | 176.149 | 0.001667 | 0.206 |
| 12 | 0.9583 | 172.977 | 0.001667 | 0.202 |
| 13 | 1.0417 | 167.272 | 0.002500 | 0.293 |
| 14 | 1.1250 | 161.524 | 0.002500 | 0.282 |
| 15 | 1.2083 | 155.720 | 0.002500 | 0.272 |
| 16 | 1.2917 | 149.849 | 0.002500 | 0.262 |
| 17 | 1.3750 | 143.903 | 0.002500 | 0.252 |
| 18 | 1.4583 | 145.527 | 0.002500 | 0.255 |
| 19 | 1.5417 | 139.634 | 0.002500 | 0.244 |
| 20 | 1.6250 | 133.415 | 0.002500 | 0.233 |
| 21 | 1.7083 | 126.459 | 0.002500 | 0.221 |
| 22 | 1.7917 | 118.713 | 0.002500 | 0.208 |
| 23 | 1.8750 | 110.054 | 0.002500 | 0.193 |
| 24 | 1.9583 | 101.326 | 0.002500 | 0.177 |
三、信用估值调整 (CVA)

$$
\boxed{\text{CVA}
      = (1-R)\sum_{i=1}^{24} q_i v_i
      = \underline{\$5.603\ \text{million}}}
$$
 四、考虑信用风险后的远期价值  

1. **无信用风险价值**  
$$
V_{\text{clean}}
  = (F_0-K)\,e^{-rT}\,N
  = 100 \times e^{-0.10}\times 10^6
  = \underline{\$90.484\ \text{million}}
$$

2. **调整后公允价值**  
$$
\boxed{V_{\text{adjusted}}
       = V_{\text{clean}} - \text{CVA}
       = 90.484 - 5.603
       = \underline{\$84.881\ \text{million}}}
$$

---

> **一行记忆**：清洁价值 − CVA = 调整后价值。  

## 20.14  
>[!question] 
使用例 20-3 中的数据计算假设银行的 DVA。  
假设银行可能在每个月中的中点违约，两年内违约概率分布为每月 0.001。  
假设银行违约时，交易对手能得到的回收率为 40%。  

>[!question] 
某银行与一家矿业公司签订 2 年期黄金远期合约，约定在第 24 个月末（$T=2$）  
以 $K=1\,500$ 美元/盎司的价格买入 $Q=1\,000\,000$ 盎司黄金。  
已知市场与合约参数如下  

| 项目 | 数值 | 说明 |
|------|------|------|
| 现行 2 年期黄金远期价 $F_0$ | 1 600 美元/盎司 | |
| 黄金对数价格波动率 $\sigma$ | 20 %（年化） | |
| 无风险连续复利利率 $r$ | 5 % | 所有期限恒定 |
| 银行违约回收率 $R_{\text{bank}}$ | 40 % | 交易对手可回收 40 % 的负债 |
| 违约时清算 | 现金结算：<br>若远期市值对交易对手为正，则银行支付该正向市值 $\times (1-R_{\text{bank}})$ |
**违约假设**  
银行可能在每个月的中点违约（即 $t_i = 0.5,\,1.5,\dots ,23.5$ 个月，共 24 个节点）。  
- 第 1 年的每月**无条件**违约概率 $q_i = 0.001$  
- 第 2 年的每月**无条件**违约概率 $q_i = 0.001$  

> 要求  
> 1. 采用 Black–Scholes–Merton 正向合约定价框架，在每个潜在违约节点 $t_i$ 计算远期合约对 **交易对手** 的正向曝险  
>    $$v_i = e^{-r\,(T-t_i)}\bigl[F_0\,N(d_{1,i}) - K\,N(d_{2,i})\bigr],\quad  
>      d_{1,i} = \frac{\ln(F_0/K)+\tfrac12\sigma^{2}(T-t_i)}{\sigma\sqrt{T-t_i}},\quad  
>      d_{2,i} = d_{1,i}-\sigma\sqrt{T-t_i}$$  
>    并乘以名义数量 $Q$（单位：美元）。  
> 2. 计算银行 **债务估值调整**（DVA）：  
>    $$\text{DVA} = (1-R_{\text{bank}})\sum_{i=1}^{24} q_i\,v_i$$  
> 3. 给出计入 DVA 后远期合约的公允价值  
>    $$V_{\text{adjusted}} = (F_0-K)Q\,e^{-rT} - \text{DVA}$$  
> 4. 简述 DVA 的经济含义：为什么它代表银行因自身违约可能性而享有的“负债减免利益”。  
所有计算结果请保留至百万美元 2 位小数。  


Ⅰ. 参数与违约设置  

| 变量                        | 数值                          | 释义         |
| ------------------------- | --------------------------- | ---------- |
| 远期价 $F_0$               | 1 600                       | USD/oz     |
| 执行价 $K$                 | 1 500                       | USD/oz     |
| 名义 $Q$                  | 1 000 000                   | 盎司         |
| 波动率 $σ$                 | 20 %                        | 连续复利、年化    |
| 利率 $r$                  | 5 %                         | 连续复利、恒定    |
| 到期 $T$                  | 2                           | 年          |
| 回收率 $R_{\text{bank}}$   | 40 %                        |            |
| 损失率 $1-R_{\text{bank}}$ | 60 %                        |            |
| 违约节点 $t_i$              | $(i-0.5)/12,\;i=1\dots24$ | 月中点        |
| 无条件违约概率 $q_i$           | 0.001                       | 所有 24 个月相同 |

---

Ⅱ. 每月正向曝险的公式  
$$
\begin{aligned}
τ_i      &= T - t_i \$$4pt]
d_{1,i}  &= \frac{\ln(F_0/K)+\tfrac12σ^{2}τ_i}{σ\sqrt{τ_i}},\qquad 
d_{2,i}=d_{1,i}-σ\sqrt{τ_i} \$$6pt]
v_i &= e^{-rτ_i}\Bigl[F_0\,N(d_{1,i})-K\,N(d_{2,i})\Bigr]\times Q
\end{aligned}
$$

> - $e^{-rτ_i}$ 👉 贴现到 **违约节点**  
> - $N(d_{1,i}),\,N(d_{2,i})$ 👉 正态 CDF  
> - 乘以 $Q$ 👉 以名义规模计价  
> - 最终 $v_i>0$ 时，代表对**交易对手**的正向市值 (银行负债)  

---

Ⅲ. 计算示范（首月）  

| 步 | 计算 | 数值 |
|---|---|---|
| 1 | $t_1 = 0.5/12 = 0.041667$ 年, $τ_1 = 1.95833$ |  |
| 2 | $d_{1,1} = \dfrac{\ln(1600/1500)+0.5×0.2^2×1.95833}{0.2\sqrt{1.95833}} = 0.371$ |  |
| 3 | $d_{2,1} = 0.371 - 0.2\sqrt{1.95833} = 0.091$ |  |
| 4 | $N(d_{1,1}) = 0.645$, $N(d_{2,1}) = 0.536$ |  |
| 5 | $v_1 = e^{-0.05×1.95833}\bigl[1600×0.645 - 1500×0.536\bigr]×10^6$ | $→ 205.86$ |

---

Ⅳ. 24 个月完整数值  

| 月 $i$ | $t_i$ (年) | $d_{1,i}$ | $d_{2,i}$ | $v_i$ | $0.6 q_i v_i$ |
|:--:|:--:|:--:|:--:|------:|------:|
| 1  | 0.0417 | 0.371 | 0.091 | **205.86** | 0.12 |
| 2  | 0.1250 | 0.373 | 0.099 | 203.46 | 0.12 |
| 3  | 0.2083 | 0.376 | 0.108 | 200.96 | 0.12 |
| 4  | 0.2917 | 0.378 | 0.116 | 198.35 | 0.12 |
| 5  | 0.3750 | 0.381 | 0.126 | 195.65 | 0.12 |
| ⋯  | ⋯      | ⋯    | ⋯    | ⋯     | ⋯ |
| 24 | 1.9583 | 1.601 | 1.560 | 101.33 | 0.06 |
| **合计** | | | | **3 963.79** | **2.38** |

- $\sum v_i = 3 963.79$  
- $\sum 0.6 q_i v_i = 2.38$

---

Ⅴ. 债务估值调整 (DVA)

$$
\boxed{
\text{DVA}
  =(1-R_{\text{bank}})\sum_{i=1}^{24} q_i v_i
  =0.60 \times 0.001 \times 3\,963.79
  = \$2.38\ \text{million}}
$$

---

Ⅵ. 公允价值比较  

1. **不含信用风险**  
$$
V_{\text{clean}}
  =(F_0-K)Q\,e^{-rT}
  =100 \times 10^{6} \times e^{-0.10}
  = \$90.48\text{ million}
$$

2. **计入 DVA**  
$$
\boxed{
V_{\text{adjusted}}
  =V_{\text{clean}}-\text{DVA}
  =90.48 - 2.38
  = \underline{\$88.11\ \text{million}}}
$$
Ⅶ. 经济解释 (一段话)

DVA 量化了 **“若未来银行自身违约，负债可打折”** 的今天价值：  
当银行违约时，只需向交易对手支付正向曝险的 $1-R_{\text{bank}}=60\%$。  
因此，现时应将这部分可能减付的金额从负债公允价值中扣除，表现为 DVA。  
违约概率或损失率越高，DVA 越大，负债的“折扣”也越大。

> **一句话归纳**：  
> DVA = 自身违约可能 → 未来少还钱 → 今天负债要减值。

## 20.15  
>[!question] 
考虑某欧式看涨期权，期权标的资产为某不付股息的股票，股票的价格为 52 美元，期权执行价格为 50 美元，  
无风险利率为 5%，波动率为 30%，期权期限为 1 年。假定回收率为 0%，无担保品，无其他交易，且违约概率与期初价格无关。  
(a) 假定无违约风险，期权价值为多少？  
(b) 假定期权承销商在期权到期时有 2% 的违约概率，期权的价格为多少？  
(c) 假如期权买入方不是在交易开始时付费，而是在期权到期时付费（包括应计利息），如果期权承约人到期时有 2% 的违约概率，  
那么以上期权费的时间安排如何降低期权买方的违约损失？  
(d) 假如在 (c) 中期权买入方有 1% 的违约概率，这对期权卖出方的风险是什么？  
讨论该情形下违约的两面性，并求交易双方期权的价格分别为多少。  

| 变量 | 数值 | 说明 |
| :--- | :---: | :--- |
| $S_0$ | \$52 | 股票现价（不付股息） |
| $K$ | \$50 | 执行价 |
| $r$ | 5 % | 无风险连续复利 |
| $\sigma$ | 30 % | 年化波动率 |
| $T$ | 1 年 | 到期 |
| 回收率 | 0 % | 违约无回收 |
| 贴现因子 | $e^{-rT}=0.951229$ | |

1. 无违约风险下的期权价值  

$$
d_1=\frac{\ln(S_0/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt T}
     =\frac{\ln(52/50)+0.05+0.045}{0.30}\approx0.4474,
\qquad
d_2=d_1-\sigma\sqrt T\approx0.1474
$$  

$$
C_0=S_0N(d_1)-K e^{-rT}N(d_2)
   =52(0.6725)-50(0.9512)(0.5586)\approx\$8.39
$$  

2. 卖方到期违约概率 2 %  

$$C_{\text{buy}}=(1-0.02)\times8.39\approx\$8.22$$  

3. 改为到期支付权利金  

权利金到期金额 $X$ 需满足 $X e^{-rT}=8.22$  
$$X=8.22\,e^{0.05}\approx\$8.64$$  

到期若卖方违约（2 %），买方可不付 $X$，因此买方信用敞口明显降低；现值仍 \$8.22。  

4. 买方到期违约概率 1 %，卖方仍 2 %  

同时存活概率 $(1-0.02)(1-0.01)$，满足  
$$e^{-rT}(1-0.02)(1-0.01)\,E[\max(S_T-K,0)]
   =(1-0.01)X e^{-rT}$$  

简化得 $X e^{-rT}=0.98\times8.39=8.22$，因而 $X\approx\$8.64$，  
买卖双方期权现值均为 \$8.22。  

关键步骤  
* 用 Black–Scholes 计算无违约价值；  
* CVA ≈ ([[信用风险：PD、LGD、EAD 与评级迁移|违约概率]]) × $C_0$ × (1–回收率)；  
* 到期支付权利金可对冲违约风险；  
* 若买卖双方均有小概率违约且金额相抵，价格几乎不变。  

## 20.16  
>[!question] 
假设一家银行发行了 3 年期无风险固定收益券的收益率加 210 个基点的浮息票据，  
由布莱克–斯科尔斯–默顿公式得出的期权价格为 4.10 美元。  
如果你以银行作为期权卖方，你愿意支付的实际价格是多少？  

| 步骤 | 关键要点 | 简述 |
|------|----------|------|
| 1️⃣ 识别资金成本 | 银行自身的融资成本 = **[[CAPM、系统风险与资本成本|无风险利率]] + 210 bps** → 记作 $s = 2.10\%$ |
| 2️⃣ 理论定价输入 | B-S-M 给出的风险中性价值 $C_{BSM}= \$4.10$，已按无风险利率 $r$ 折现 |
| 3️⃣ 调整贴现因子 | 作为卖方使用 **自身资金成本** 计价：<br/> 连续复利：$DF = e^{-sT} = e^{-0.021\times3}=0.93896$<br/> 或年度复利：$DF = (1+s)^{-T}=(1.021)^{-3}=0.9392$ |
| 4️⃣ 求得愿付价格 | $C_{\text{internal}} = C_{BSM} \times DF \approx 4.10 \times 0.939 \approx \$3.85$ |
| 5️⃣ 结论 | **银行作为期权卖方，最多愿意支付 ≈ \$3.85** 来购入/对冲该期权（报价不应低于此） |

> **一句话记忆**：理论价 × 自身贴现系数 = 内部可接受价。
<!-- bilingual-en:start -->
A bank agrees to buy 1,000,000 ounces of gold from a mining company for USD 1,500 per ounce at the end of month 24. The current two-year forward price is $F_0=1600$, annualised log-price volatility is $\sigma=20\%$, the continuously compounded risk-free rate is $r=5\%$, and recovery is $R=30\%$. Default may occur at the midpoint of any month, giving 24 nodes at 0.5, 1.5, ..., 23.5 months. Each month in year one has unconditional default probability 0.001667; each month in year two has probability 0.0025. On default, cash settlement pays 70% of the positive market value.

The tasks are to calculate positive exposure $v_i$ at every monthly node, calculate two-year CVA using the corresponding unconditional $q_i$, and calculate the credit-adjusted fair value. The one-million-ounce notional is included directly, and all monetary results are in U. S. dollars.

**Solution: inputs and setup**

- Notional: $N=1{,}000{,}000$ oz
- Maturity: $T=24$ months, or two years
- Forward and delivery prices: $F_0=1600$ and $K=1500$ USD/oz
- Volatility: $\sigma=20\%$ per year
- Continuously compounded risk-free rate: $r=5\%$
- Recovery and loss rates: $R=30\%$ and $1-R=70\%$
- Default nodes: $t_i=(i-0.5)/12$, for $i=1,\ldots,24$
- Unconditional monthly default probability: 0.001667 for $i=1,\ldots,12$ and 0.0025 for $i=13,\ldots,24$

At each node let $\tau_i=T-t_i$. The positive forward exposure is obtained from the Black--Scholes--Merton positive-part formula,
$v_i=e^{-r\tau_i}[F_0N(d_{1, i})-KN(d_{2, i})]N$.

| Month $i$ | $t_i$ (years) | $v_i$ (USD m) | $q_i$ | $(1-R)q_iv_i$ (USD m) |
|:--:|:--:|--:|--:|--:|
| 1 | 0.0417 | 205.861 | 0.001667 | 0.240 |
| 2 | 0.1250 | 203.456 | 0.001667 | 0.237 |
| 3 | 0.2083 | 200.955 | 0.001667 | 0.234 |
| 4 | 0.2917 | 198.355 | 0.001667 | 0.231 |
| 5 | 0.3750 | 195.647 | 0.001667 | 0.228 |
| 6 | 0.4583 | 192.816 | 0.001667 | 0.225 |
| 7 | 0.5417 | 189.846 | 0.001667 | 0.222 |
| 8 | 0.6250 | 186.718 | 0.001667 | 0.219 |
| 9 | 0.7083 | 183.411 | 0.001667 | 0.215 |
| 10 | 0.7917 | 179.899 | 0.001667 | 0.211 |
| 11 | 0.8750 | 176.149 | 0.001667 | 0.206 |
| 12 | 0.9583 | 172.977 | 0.001667 | 0.202 |
| 13 | 1.0417 | 167.272 | 0.002500 | 0.293 |
| 14 | 1.1250 | 161.524 | 0.002500 | 0.282 |
| 15 | 1.2083 | 155.720 | 0.002500 | 0.272 |
| 16 | 1.2917 | 149.849 | 0.002500 | 0.262 |
| 17 | 1.3750 | 143.903 | 0.002500 | 0.252 |
| 18 | 1.4583 | 145.527 | 0.002500 | 0.255 |
| 19 | 1.5417 | 139.634 | 0.002500 | 0.244 |
| 20 | 1.6250 | 133.415 | 0.002500 | 0.233 |
| 21 | 1.7083 | 126.459 | 0.002500 | 0.221 |
| 22 | 1.7917 | 118.713 | 0.002500 | 0.208 |
| 23 | 1.8750 | 110.054 | 0.002500 | 0.193 |
| 24 | 1.9583 | 101.326 | 0.002500 | 0.177 |

The source obtains
$\text{CVA}=(1-R)\sum_{i=1}^{24}q_iv_i=\$5.603\text{ million}$.
Clean value is $(F_0-K)e^{-rT}N=\$90.484$ million, so credit-adjusted value is $90.484-5.603=\$84.881$ million. In one line: clean value minus CVA equals counterparty-credit-adjusted value.

**Question 20.14: the bank's DVA**

Using the same two-year gold forward, suppose the bank may default at each monthly midpoint with unconditional probability 0.001 per month, and the counterparty recovers 40%. Calculate the counterparty's positive exposure to the bank, the bank's DVA, the value after DVA, and explain the economic meaning of DVA.

The parameters remain $F_0=1600$, $K=1500$, $Q=1{,}000{,}000$, $\sigma=20\%$, $r=5\%$, and $T=2$. Bank recovery is 40%, so bank LGD is 60%. For $t_i=(i-0.5)/12$ and $\tau_i=T-t_i$, use
$v_i=e^{-r\tau_i}[F_0N(d_{1, i})-KN(d_{2, i})]Q$.
The discount factor places value at the default node, the normal CDF terms produce the positive part, and multiplication by $Q$ converts per-ounce value to contract value.

For the first month, $t_1=0.041667$, $\tau_1=1.95833$, $d_{1,1}\approx0.371$, $d_{2,1}\approx0.091$, $N(d_{1,1})\approx0.645$, and $N(d_{2,1})\approx0.536$, giving $v_1\approx\$205.86$ million. The source reports total monthly exposure $\sum v_i=\$3{,}963.79$ million and
$\text{DVA}=0.60\times0.001\times3{,}963.79=\$2.38$ million.

DVA measures the present value of the bank's possible reduction in what it ultimately pays if the bank itself defaults. A greater own-default probability or loss rate raises this liability discount. The source then reports an adjusted value of USD 88.11 million by subtracting DVA from USD 90.48 million. This line needs sign and arithmetic caution: $90.48-2.38=88.10$, while the usual bank-asset convention writes bilateral value as clean value minus CVA plus DVA. Verify the perspective and sign before using the reported number.

**Question 20.15: European call with bilateral default risk**

For a non-dividend-paying stock, $S_0=\$52$, $K=\$50$, $r=5\%$, $\sigma=30\%$, and $T=1$. Recovery and collateral are zero, there are no other trades, and default is independent of the initial stock price.

(a) With no default, Black--Scholes gives $d_1\approx0.4474$, $d_2\approx0.1474$, and $C_0\approx\$8.39$.

(b) If the option writer defaults at maturity with probability 2%, the source applies zero-recovery CVA directly and obtains approximately $0.98\times8.39=\$8.22$.

(c) If the buyer pays the premium, including accrued interest, only at maturity, no payment is made when the writer defaults. This payment timing offsets much of the buyer's exposure to writer default. The maturity premium $X$ is chosen so its present value is USD 8.22.

(d) If the buyer also has a 1% default probability, the writer faces the risk that the deferred premium is not paid. The source uses a simplified simultaneous-survival calculation and reports $Xe^{-rT}=0.98\times8.39=\$8.22$, so $X\approx\$8.64$ and both sides' stated present value is USD 8.22. This shortcut depends on the exercise's simplified default-timing and independence assumptions; a full bilateral model would distinguish positive and negative exposure and competing defaults.

**Question 20.16: funding-spread adjustment**

A bank funds at the three-year risk-free fixed-income yield plus 210 bp, and the Black--Scholes--Merton option value is USD 4.10. The source treats the bank's own funding spread as an additional discount: $s=2.10\%$, $e^{-sT}=e^{-0.021\times3}\approx0.93896$, and $4.10\times0.939\approx\$3.85$. It therefore gives about USD 3.85 as the bank's internal maximum price for buying or hedging the option. This is a funding-cost heuristic, not by itself a general arbitrage-free valuation rule; its use depends on the institution's valuation framework.
<!-- bilingual-en:end -->
