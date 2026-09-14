
# 1. 衍生产品信用风险的特点与基本概念
<!-- bilingual-en:start -->
*1. Counterparty Credit Risk in Derivatives: Key Features and Concepts*
<!-- bilingual-en:end -->

衍生产品交易的信用风险比传统贷款的信用风险更加复杂 。原因在于，贷款的风险敞口在发放时就确定（例如一笔5年期、1000万美元贷款在整个期间风险敞口基本为1000万），而衍生产品的未来风险暴露（违约发生时可能遭受的损失金额）是不确定的，会随市场变化而波动 。例如，银行与客户做一笔5年期利率互换：如果互换对银行而言现值为正，意味着客户欠银行钱，此时银行的信用风险敞口等于互换合约的当前价值；如果互换对银行而言价值为负（银行欠客户），则银行面对的信用风险敞口为零 。也就是说，衍生品交易的敞口动态变化，只有当交易对手的合约市值对我们为正时才存在信用暴露风险。
<!-- bilingual-en:start -->
Counterparty credit risk is more complex for derivatives than for a conventional loan. A loan's exposure is largely fixed when it is advanced: a five-year USD 10 million loan has roughly USD 10 million of principal exposure throughout. A derivative's future exposure is uncertain because its market value changes. In a five-year interest-rate swap, the bank is exposed when the swap has positive value to the bank—the client then owes the bank that amount. When the swap has negative value to the bank, the bank owes the client and its current counterparty exposure is zero. Exposure therefore exists only on the positive part of the contract's market value.
<!-- bilingual-en:end -->

为降低衍生品交易的信用风险，场外衍生品常使用双边净额和抵押品机制。ISDA 主协议可以约定违约或提前终止时把合格交易归并为一个净额，但合同名称本身不保证监管认可；还须证明相关交易、对手、破产程序和司法辖区下可依法执行。CSA 主要规定抵押品安排，也不自动创造跨协议净额权。只有在这些法律边界满足时，净额才可把逐笔正敞口降为认可净额集合的敞口。
<!-- bilingual-en:start -->
OTC derivatives commonly use bilateral netting and collateral. An ISDA Master Agreement can document close-out netting, while a CSA governs collateral, but regulatory recognition still requires legal enforceability for the relevant trades, counterparty, insolvency regime, and jurisdictions. Contract labels alone do not create recognised netting.
<!-- bilingual-en:end -->

# 2. CVA与DVA的定义、公式及意义
<!-- bilingual-en:start -->
*2. Definitions, formulas, and significance of CVA and DVA*
<!-- bilingual-en:end -->

**信用价值调整**（[[单边CVA|CVA]], Credit Valuation Adjustment）是对手方违约损失的价值调整。在与后文贴现、净额集、抵押品和 close-out 口径一致时，它从 clean value 中扣除。只有 CVA 和 DVA 的定义、违约顺序与 close-out 口径一致时，才可写成 $V_{\text{bil}}=V_0-\text{CVA}+\text{DVA}$。
<!-- bilingual-en:start -->
**Credit valuation adjustment** ([[单边CVA|CVA]], Credit Valuation Adjustment) is the valuation adjustment for loss caused by counterparty default. It is deducted from clean value under a discounting, netting, collateral and close-out framework used consistently throughout. The shorthand $V_{\text{bil}}=V_0-\text{CVA}+\text{DVA}$ is valid only when CVA and DVA use consistent definitions, default ordering and the same close-out convention.
<!-- bilingual-en:end -->

## 2.1 CVA

**一般单边公式：**令 $\tau_C$ 为对手方违约时间，$E_{\tau_C}$ 为同一 close-out、[[净额与抵押品|净额与抵押品]]口径下的违约正敞口，则
$$
\mathrm{UCVA}=\mathbb E^{\mathbb Q}\!\left[D(0,\tau_C)\,\mathrm{LGD}_C\,E_{\tau_C}\,\mathbf 1_{\{\tau_C\le T\}}\right].
$$
若将 $[0,T]$ 划为 $A_i=\{t_{i-1}<\tau_C\le t_i\}$，$q_i=\mathbb Q(A_i)$ 为该段无条件风险中性违约概率，并用 $t_i$ 或中点代表该段，则一般离散近似先写成
$$
\mathrm{UCVA}\approx\sum_{i=1}^n q_i\,
\mathbb E^{\mathbb Q}[D(0,t_i)\mathrm{LGD}_C E(t_i)\mid A_i].
$$
只有在 LGD 确定，且贴现敞口 $D(0,t_i)E(t_i)$ 与违约区间事件 $A_i$ 独立时，才进一步简化为；一个充分的特例是贴现因子确定且 $E(t_i)$ 与 $A_i$ 独立：
<!-- bilingual-en:start -->
**General unilateral formula:** Let $\tau_C$ be counterparty default time and let $E_{\tau_C}$ be positive exposure under the same close-out, [[净额与抵押品|netting and collateral]] convention. Then
$$
\mathrm{UCVA}=\mathbb E^{\mathbb Q}\!\left[D(0,\tau_C)\,\mathrm{LGD}_C\,E_{\tau_C}\,\mathbf 1_{\{\tau_C\le T\}}\right].
$$
Partition $[0,T]$ into $A_i=\{t_{i-1}<\tau_C\le t_i\}$ with unconditional risk-neutral period-default probability $q_i=\mathbb Q(A_i)$. Using $t_i$ or a midpoint to represent each period first gives the conditional approximation
$$
\mathrm{UCVA}\approx\sum_{i=1}^n q_i\,
\mathbb E^{\mathbb Q}[D(0,t_i)\mathrm{LGD}_C E(t_i)\mid A_i].
$$
Only with deterministic LGD and independence between discounted exposure $D(0,t_i)E(t_i)$ and the default-period event $A_i$ does this reduce further. A sufficient special case is deterministic discounting with $E(t_i)$ independent of $A_i$:
<!-- bilingual-en:end -->

$$
\mathrm{UCVA}\approx \mathrm{LGD}_C\sum_{i=1}^n q_i\,\mathbb E^{\mathbb Q}[D(0,t_i)E(t_i)]
=(1-R_C)\sum_{i=1}^n q_i v_i.
$$
有[[错向风险|错向风险]]时，不能把违约概率与无条件敞口均值直接相乘。$q_i$可由信用利差曲线推导；若强度 $\lambda$ 恒定，$q_i\approx e^{-\lambda t_{i-1}}-e^{-\lambda t_i}$。上式中的经济 LGD 应与实际 close-out 损失口径匹配；只有合同回收率 $R_C$ 正是该损失口径时，才可写成 $\mathrm{LGD}_C=1-R_C$，不应把监管 LGD 或其他回收率机械代入。
<!-- bilingual-en:start -->
$$
\mathrm{UCVA}\approx \mathrm{LGD}_C\sum_{i=1}^n q_i\,\mathbb E^{\mathbb Q}[D(0,t_i)E(t_i)]
=(1-R_C)\sum_{i=1}^n q_i v_i.
$$
With [[错向风险|wrong-way risk]], unconditional default probability and unconditional expected exposure cannot simply be multiplied. The $q_i$ values can be inferred from a credit-spread term structure; for constant intensity $\lambda$, $q_i\approx e^{-\lambda t_{i-1}}-e^{-\lambda t_i}$. Economic LGD must match the actual close-out loss claim. Writing $\mathrm{LGD}_C=1-R_C$ is justified only when contractual recovery $R_C$ matches that loss definition; regulatory LGD or a recovery estimate from another framework is not mechanically interchangeable.
<!-- bilingual-en:end -->

## 2.2 DVA

**债务价值调整**（[[DVA与双边估值|DVA]], Debit Valuation Adjustment）是本行自身违约风险对本行负敞口的调整。从银行视角，令 $V_t^+=\max(V_t,0)$、$V_t^-=\max(-V_t,0)$，$\tau_B$为银行违约时间、$\tau_C$为对手违约时间。在同一 close-out 定义下，first-to-default 双边式为
$$
\begin{aligned}
\mathrm{CVA}&=\mathbb E^{\mathbb Q}\!\left[D(0,\tau_C)\mathrm{LGD}_C V_{\tau_C}^{+}\mathbf1_{\{\tau_C<\tau_B,\,\tau_C\le T\}}\right],\\
\mathrm{DVA}&=\mathbb E^{\mathbb Q}\!\left[D(0,\tau_B)\mathrm{LGD}_B V_{\tau_B}^{-}\mathbf1_{\{\tau_B<\tau_C,\,\tau_B\le T\}}\right].
\end{aligned}
$$
因此 DVA 是对银行 clean value 的加项，但只有双方违约顺序、相依性和 close-out 都一致时，才能与 CVA 机械组合。监管 CVA 风险资本不把 own-default 影响算入 CVA；会计 DVA 与监管 CVA 是不同口径。
<!-- bilingual-en:start -->
**Debit valuation adjustment** ([[DVA与双边估值|DVA]]) is the adjustment for the bank's own default risk on the bank's negative exposure. From the bank's perspective set $V_t^+=\max(V_t,0)$ and $V_t^-=\max(-V_t,0)$, with bank and counterparty default times $\tau_B$ and $\tau_C$. Under one common close-out convention, the first-to-default formulas are
$$
\begin{aligned}
\mathrm{CVA}&=\mathbb E^{\mathbb Q}\!\left[D(0,\tau_C)\mathrm{LGD}_C V_{\tau_C}^{+}\mathbf1_{\{\tau_C<\tau_B,\,\tau_C\le T\}}\right],\\
\mathrm{DVA}&=\mathbb E^{\mathbb Q}\!\left[D(0,\tau_B)\mathrm{LGD}_B V_{\tau_B}^{-}\mathbf1_{\{\tau_B<\tau_C,\,\tau_B\le T\}}\right].
\end{aligned}
$$
DVA is therefore added to the bank's clean value, but it can be combined mechanically with CVA only when default ordering, dependence and close-out definitions are consistent. Regulatory CVA risk excludes own-default effects; accounting DVA and regulatory CVA are different quantities.
<!-- bilingual-en:end -->

## 3. 衍生品风险敞口的计算方法（含抵押品与 MPOR）
<!-- bilingual-en:start -->
*3. Calculating Derivative Exposure, Including Collateral and MPOR*
<!-- bilingual-en:end -->

**[[对手方敞口指标|风险敞口]]**（Exposure）是某一未来时点对手方当即违约时，本方在适用 close-out 口径下的正敞口。对无抵押组合，$E(t)=[V^{\text{net}}(t)]^+$；多笔交易只能在法律可执行的[[净额与抵押品|净额集]]内先抵销，再取正部分。
<!-- bilingual-en:start -->
**[[对手方敞口指标|Exposure]]** at a future date is the positive amount owed to the bank if the counterparty defaults then under the applicable close-out convention. For an uncollateralised portfolio, $E(t)=[V^{\text{net}}(t)]^+$. Trades are offset before taking the positive part only within a legally enforceable [[净额与抵押品|netting set]].
<!-- bilingual-en:end -->

以下简化式只适用于 VM 仍按 **collateralized-to-market（CTM）** 处理，且 $V^{\text{net}}(t)$ 是扣除该 VM 前的市值。定义有符号抵押品 $C(t)$：本方**收到并能在对手违约 close-out 中依法使用**的抵押品为正；本方提交且在对手违约时形成未隔离返还请求的抵押品为负。则不含其他 close-out 调整的简化敞口是
$$E(t)=[V^{\text{net}}(t)-C(t)]^+.$$
必须同时说清谁违约：若对手 B 向 A 提交抵押品后 B 违约，A 已持有的合格抵押品通常减少 A 的损失；若 A 向 B 提交抵押品而 B 违约，A 面临抵押品返还请求不能全额收回的风险。反之，A 提交后 A 自己违约时，B 已持有的抵押品通常减少 B 的损失，不是增加 B 敞口。合格隔离、bankruptcy-remote 的 posted IM 不能机械记为负 $C$；收到 IM 能否用于 close-out 也取决于法律与隔离安排。若 VM 是 **settled-to-market（STM）** 付款，它已经重置 $V^{\text{net}}$，不能又放进 $C$ 从同一市值扣一次。超额抵押品进入 close-out、净额和返还请求，不是自动没收。
<!-- bilingual-en:start -->
The following shortcut applies only when VM is treated as **collateralized-to-market (CTM)** and $V^{\text{net}}(t)$ is measured before deducting that VM. Use signed collateral $C(t)$: collateral **received and legally available in counterparty-default close-out** is positive, while posted collateral that creates an unsegregated return claim on counterparty default is negative. A simplified exposure before other close-out adjustments is
$$E(t)=[V^{\text{net}}(t)-C(t)]^+.$$
The identity of the defaulter matters. If B posted collateral to A and B defaults, eligible collateral already held by A normally reduces A's loss. If A posted to B and B defaults, A instead faces the risk that its collateral-return claim is not recovered in full. If A defaults after posting to B, collateral already held by B normally reduces B's loss; it does not increase B's exposure. Qualifying segregated, bankruptcy-remote posted IM is not mechanically negative $C$, and whether received IM is usable in close-out depends on the legal and segregation arrangement. If VM is a **settled-to-market (STM)** payment, it has already reset $V^{\text{net}}$ and must not be deducted again through $C$. Excess collateral enters close-out, netting and a return claim rather than being automatically forfeited.
<!-- bilingual-en:end -->

**变动保证金（VM）**覆盖当前按市值；**初始保证金（IM）**用于覆盖从最后一次 VM 交换到 close-out、替代交易或重新对冲完成期间的潜在变动。**Threshold** 是协议允许的未抵押额；**MTA** 是是否发起保证金调用的最低转移门槛，不是每次自动从抵押品中扣减；争议会使未支付或争议部分延后交换。

**保证金风险期（MPOR）**是从最后一次抵押品交换起，跨过随后可能发生的违约，并延续到违约组合 close-out、替代或重新对冲完成为止的未覆盖变动窗口。起点可以早于违约；它不是只从违约时刻向后数，也不是用已实现历史市值做回望。
<!-- bilingual-en:start -->
**Variation margin (VM)** covers current mark-to-market; **initial margin (IM)** is intended to cover potential changes from the last VM exchange until close-out, replacement or rehedging is completed. A **threshold** is permitted unsecured exposure. **MTA** is the minimum transfer amount that determines whether a margin call is made, not an automatic deduction from every collateral balance. A dispute delays exchange of the unpaid or disputed amount.

The **margin period of risk (MPOR)** starts at the last collateral exchange, spans any subsequent default, and continues until close-out, replacement or rehedging of the defaulted portfolio is completed. Its start can precede default; it is neither a window counted only from default nor a lookback to realised historical market values.
<!-- bilingual-en:end -->

**有抵押品及 MPOR 的敞口示例：**设 B 在最后一次 VM 交换时向 A 提交 45，随后 B 违约，并进入 20 天 MPOR：
<!-- bilingual-en:start -->
**Example with collateral and MPOR:** B posts VM of 45 to A at the final margin exchange, then B defaults and the portfolio enters a 20-day MPOR:
<!-- bilingual-en:end -->

- 若 close-out 时组合对 A 的价值上升到 50，则 $E=[50-45]^+=5$。这 5 是 MPOR 内市场变动造成的 gap exposure。
<!-- bilingual-en:start -->
- If close-out value to A rises to 50, then $E=[50-45]^+=5$. The five-unit gap exposure is caused by market movement during MPOR.
<!-- bilingual-en:end -->

- 若 close-out 时价值降到 40，则 A 的正敞口为 0，但多持有的 5 必须进入 close-out 净额和 B 破产财团的返还请求，不是 A 自动获得。
<!-- bilingual-en:start -->
- If close-out value falls to 40, A's positive exposure is zero, but the excess five enters close-out netting and B's estate has a return claim; A does not automatically keep it.
<!-- bilingual-en:end -->
模拟中应在每条未来路径上固定最后一次可用抵押品，再向后模拟到 close-out/重新对冲完成；这与把今天已实现的历史价值倒推 $\delta$ 天不同。
<!-- bilingual-en:start -->
On each simulated future path, hold the last available collateral balance fixed and evolve value forward until close-out or rehedging is complete. This is different from defining MPOR by looking backward through realised history from today.
<!-- bilingual-en:end -->

## 4. CVA计算中的Monte Carlo模拟应用
<!-- bilingual-en:start -->
*4. Application of Monte Carlo Simulation in CVA Calculations*
<!-- bilingual-en:end -->

由于衍生产品未来价值取决于市场风险因素（利率、汇率、股价、商品价格等）的随机演变，**蒙特卡罗模拟**是计算CVA时常用的工具 。基本思路是在**风险中性**假设下，对未来市场变量从当前时刻一直模拟到交易组合最长到期$T$ 。沿每条模拟路径，可在预设的时间网格上计算交易商对交易对手的正敞口。对大量路径取平均并一致贴现，得到各节点 $v_i$；在满足第 2.1 节的独立性等离散近似条件时，再与 $q_i$ 相乘。如有 WWR，应使用违约条件敞口或联合模型。
<!-- bilingual-en:start -->
Because future derivative values depend on stochastic market factors, **Monte Carlo simulation** is widely used for CVA. Under the **risk-neutral** measure, simulate factors to the longest maturity, calculate pathwise positive exposure, average and discount consistently to obtain each $v_i$. Multiply by $q_i$ only when the discrete-approximation conditions in Section 2.1 hold. With WWR, use default-conditional exposure or a joint model.
<!-- bilingual-en:end -->

估值 CVA 中的市场因子与违约概率通常在风险中性框架下校准。贴现曲线则必须与 clean valuation、抵押品计息和 close-out 口径一致，不应把“无风险利率”当作脱离抵押品框架的固定输入。
<!-- bilingual-en:start -->
CVA is a valuation adjustment, so the simulation uses a **risk-neutral distribution** and risk-neutral default probabilities. Market factors should be calibrated to current forward curves or option-implied distributions, and future losses are discounted consistently with the valuation framework. The result is the amount by which counterparty default risk reduces the derivative's value in the risk-neutral pricing measure.
<!-- bilingual-en:end -->

实际操作中，银行会对每个重要对手定期运行CVA的Monte Carlo仿真，生成所有交易组合的敞口分布和CVA估计值。由于模拟路径已生成并存储，**新增交易**的CVA影响也可方便地通过同一批路径计算（后面章节详述新增CVA计算） 。
<!-- bilingual-en:start -->
In practice, banks regularly run Monte Carlo CVA calculations for each important counterparty, producing exposure distributions and a CVA estimate for the entire netting set. Because the simulated market paths are stored, the incremental effect of a **new trade** can be calculated on the same paths rather than by rebuilding the simulation from scratch.
<!-- bilingual-en:end -->

## 5. CE、EE、EPE 与 PFE
<!-- bilingual-en:start -->
*5. CE, EE, EPE and PFE*
<!-- bilingual-en:end -->

沿用上一节 CTM/STM、抵押品可动用性和返还权边界，先定义路径敞口 $E(t)=[V^{\text{net}}(t)-C(t)]^+$。四个指标不可混用：

- **当前敞口（CE）：**$\mathrm{CE}=[V_0^{\text{net}}-C_0]^+$，是今天已实现的敞口。
- **预期敞口（EE）：**$\mathrm{EE}(t)=\mathbb E^{\mathbb M}[E(t)]$，是某一未来日期的敞口均值。
- **预期正敞口（EPE）：**$\mathrm{EPE}_H=\frac1H\int_0^H\mathrm{EE}(t)\,dt$，是区间 $[0,H]$ 内 EE 的时间平均。
- **潜在未来敞口（PFE）：**$\mathrm{PFE}_\alpha(t)=Q_\alpha^{\mathbb M}(E(t))$，是某一日期敞口分布的 $\alpha$ 分位数。
<!-- bilingual-en:start -->
Subject to the preceding CTM/STM, collateral-availability and return-claim boundaries, define pathwise exposure $E(t)=[V^{\text{net}}(t)-C(t)]^+$. The four measures are distinct:

- **Current exposure (CE):** $\mathrm{CE}=[V_0^{\text{net}}-C_0]^+$, today's realised exposure.
- **Expected exposure (EE):** $\mathrm{EE}(t)=\mathbb E^{\mathbb M}[E(t)]$, mean exposure at one future date.
- **Expected positive exposure (EPE):** $\mathrm{EPE}_H=\frac1H\int_0^H\mathrm{EE}(t)\,dt$, the time average of EE over $[0,H]$.
- **Potential future exposure (PFE):** $\mathrm{PFE}_\alpha(t)=Q_\alpha^{\mathbb M}(E(t))$, the $\alpha$-quantile of exposure at one future date.
<!-- bilingual-en:end -->

对离散日期取 $\max_k\mathrm{PFE}_\alpha(t_k)$，只是各日期特定分位数的最大值。它不是路径内最大敞口的分位数 $Q_\alpha(\max_t E(t))$，更不是“最坏情形”或无限置信水平的上界。例如 10,000 条路径上某日 97.5% PFE 约为第 250 大的当日敞口。
<!-- bilingual-en:start -->
For discrete dates, $\max_k\mathrm{PFE}_\alpha(t_k)$ is merely the maximum of date-specific quantiles. It is not the quantile of the pathwise maximum $Q_\alpha(\max_tE(t))$, nor is it a worst case or an upper bound at infinite confidence. With 10,000 paths, a 97.5% PFE at one date is approximately the 250th-largest exposure at that date.
<!-- bilingual-en:end -->

上式的测度 $\mathbb M$ 取决于用途，不是指标名称的一部分。估值用 CVA 通常在 $\mathbb Q$ 下与市场校准、贴现与抵押品框架一致；限额、经济资本或监管用 EE/PFE 可能使用真实世界、历史或压力测度。因此不能把“CVA=$\mathbb Q$、PFE=真实世界”当作定义。

Basel 中的 **Effective EPE** 还有额外单调化：离散网格上 $\mathrm{Effective\ EE}(t_k)=\max\{\mathrm{Effective\ EE}(t_{k-1}),\mathrm{EE}(t_k)\}$，再在一年窗口内做时间加权平均；若净额集合中最长合约剩余期限不足一年，就只平均到该最长期限。它不等于普通 EPE。Basel 的标准术语是 Effective EE 与 Effective EPE；当前 SA-CCR 公式中的 `PFE` 是监管 add-on，不是上文的模拟分位数 PFE，也不存在与之平行的标准 “Effective PFE”。
<!-- bilingual-en:start -->
The measure $\mathbb M$ depends on purpose; it is not part of the metric's definition. Valuation CVA normally uses $\mathbb Q$ with market calibration and a consistent discounting and collateral framework. EE or PFE for limits, economic capital or regulation may use real-world, historical or stressed measures. Hence “CVA equals $\mathbb Q$ and PFE equals real world” is not a definition.

Basel **Effective EPE** adds monotonicity. On a discrete grid, $\mathrm{Effective\ EE}(t_k)=\max\{\mathrm{Effective\ EE}(t_{k-1}),\mathrm{EE}(t_k)\}$, and Effective EPE is the time-weighted average of Effective EE over one year, or only to the longest remaining maturity in the netting set when that is shorter. It is not ordinary EPE. Basel's standard terms are Effective EE and Effective EPE. The `PFE` in current SA-CCR is a supervisory add-on, not the simulated quantile PFE above, and there is no parallel standard metric called “Effective PFE.”
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

当对手方与交易商之间增加一笔新交易时，必须在每条路径上先把新交易与原净额集合合并、应用抵押品和正部，再计算 $\Delta\mathrm{CVA}=\mathrm{CVA}_{\text{new}}-\mathrm{CVA}_{\text{old}}$。相关性只是一个驱动因素；方向、规模、非线性、抵押品以及与违约和折现的依赖共同决定结果，相关系数的符号本身不能保证 CVA 升降：
<!-- bilingual-en:start -->
The new trade must first be combined path by path with the existing netting set, collateral and positive-part operator, after which $\Delta\mathrm{CVA}=\mathrm{CVA}_{\text{new}}-\mathrm{CVA}_{\text{old}}$ is computed. Correlation is only one driver; direction, scale, nonlinearity, collateral and dependence with default and discounting also matter, so the sign of a correlation coefficient alone does not determine the CVA change.
<!-- bilingual-en:end -->

- **同向价值情形：**如果新增交易在原组合已对交易商有较大正敞口的路径上也通常为较大正值，它往往扩大路径净正敞口，因而**可能增加**组合 CVA；最终仍以重算后的增量为准。
<!-- bilingual-en:start -->
- **Aligned-value case:** if the new trade tends to have a large positive value on paths where the existing portfolio already has large positive exposure, it tends to enlarge pathwise net positive exposure and **may increase** portfolio CVA. The recomputed increment remains decisive.
<!-- bilingual-en:end -->
    
- **抵消价值情形：**如果新增交易恰在原组合正敞口较大的路径上对交易商为负值，它会压低净额后的正敞口，因而**可能降低**组合 CVA。这里的负价值对交易商本身并非“有利”，只是它在同一净额集合内抵消了另一笔正价值。
<!-- bilingual-en:start -->
- **Offsetting-value case:** if the new trade is negative to the dealer precisely on paths where the existing portfolio has large positive exposure, it lowers net positive exposure and can **reduce portfolio CVA**. The negative value is not favourable in isolation; it offsets another positive value within the same netting set.
<!-- bilingual-en:end -->
    

>[!example] 例子
> **假设交易商和某对手方已有一笔5年期外汇远期（对手方将来从银行买入外汇，银行持有潜在敞口）。若该对手希望新增一笔3年期外汇远期：
><!-- bilingual-en:start -->
>**Suppose a dealer and counterparty already have a five-year foreign-exchange forward under which the counterparty will buy foreign currency from the bank. They now consider an additional three-year forward:**
><!-- bilingual-en:end -->

- 如果对手方在新增 3 年远期中仍是**买入外汇**的一方，在题设方向、规模和其他条件不变的简化情形下，新交易会在原组合正敞口较大的路径上增加正价值，CVA 因而往往**上升**；最终仍须重算路径级增量。
<!-- bilingual-en:start -->
- If the counterparty is again the **buyer of foreign currency**, then under the illustration's fixed directions, scale and other assumptions, the new trade tends to add value on paths where the old portfolio already has positive exposure. CVA therefore tends to **rise**, subject to pathwise recomputation.
<!-- bilingual-en:end -->
    
- 如果对手方在新增远期中改为**卖出外汇**（方向相反），在同一简化条件下，新交易会在原组合正敞口较大的路径上产生抵消价值，组合 CVA 因而可能**下降**；这不是由“负相关”三个字自动推出的结论。
<!-- bilingual-en:start -->
- If the counterparty instead **sells foreign currency** in the new forward, then under the same simplified assumptions its value offsets the original trade on high-exposure paths. Combined CVA can therefore **fall**; the label “negative correlation” alone does not prove the result.
<!-- bilingual-en:end -->
    

这一原理意味着：对于同一无抵押双边净额集中的老客户，新交易的边际 CVA 可能小于新客户。集中清算会改变法律对手和适用的净额集，但 CCP 账户内仍可对合格组合净额和按组合计提保证金；不能绝对地说每笔交易都独立。边际价值取决于 CCP 的账户、净额与保证金规则。
<!-- bilingual-en:start -->
For an established client with trades in the same unsecured bilateral netting set, a new trade may add less CVA than the same trade with a new client. Central clearing changes the legal counterparty and applicable netting set, but eligible positions can still be netted and margined on a portfolio basis within a CCP account; cleared trades are not necessarily independent trade by trade. Marginal value depends on the CCP account, netting and margin rules.
<!-- bilingual-en:end -->

**新增CVA的计算：**在实际计算中，银行通常在进行 CVA 模拟时保存所有模拟路径的市场变量和组合价值。当有新交易加入时，可直接利用已保存的市场情景，对每条模拟路径在对应时间节点为新交易重新定价，得到新交易在各情景下各时点的价值。然后将此**附加价值**叠加到原组合每条路径的价值上，以更新敞口均值 $v_i$。新旧 $v_i$ 之差在满足离散分解条件时可代入 $(1-R)\sum q_i\Delta v_i$，得到新增交易导致的 CVA 增量。这种方法复用原有 Monte Carlo 路径，无需重新生成全部市场情景，但仍须对合并组合重做路径级净额、抵押品和正部计算。
<!-- bilingual-en:start -->
**Incremental CVA:** Banks normally retain the market-factor paths and existing portfolio values from a CVA simulation. When a new trade is proposed, revalue only that trade on the stored paths and dates, add its pathwise value to the existing portfolio, and recompute expected positive exposure. Substituting the change $\Delta v_i$ into $(1-R)\sum_iq_i\Delta v_i$ gives the new trade's incremental CVA. This reuses the original simulation rather than generating every path again.
<!-- bilingual-en:end -->

**举例：**在第 545 条路径的 2.5 年节点，原组合价值为 240 万，其贴现路径敞口约为 230 万；新交易价值为 $-420$ 万时，该路径合并价值为 $-180$ 万，正敞口为 0。这里的 230 万是**单条路径、单一日期**的贴现敞口，不是 $v_{20}$；只有对所有路径取期望后，才得到 $v_{20}=\mathbb E[D(0,t_{20})E(t_{20})]$。
<!-- bilingual-en:start -->
**Example:** On path 545 at year 2.5, existing portfolio value is 2.40 million and its discounted pathwise exposure is about 2.30 million. A new trade worth −4.20 million makes combined value −1.80 million, so positive exposure on that path and date is zero. The 2.30 million is a **single-path, single-date** discounted exposure, not $v_{20}$. Only averaging all paths gives $v_{20}=\mathbb E[D(0,t_{20})E(t_{20})]$.
<!-- bilingual-en:end -->

## 8. CVA的市场风险（CVA Risk）及希腊值，Basel III对CVA风险资本要求
<!-- bilingual-en:start -->
*8. CVA Market Risk, Greeks, and Basel III Capital Requirements*
<!-- bilingual-en:end -->

CVA本身取决于市场风险因素和信用风险因素，因此具有显著的**[[市场风险]]**属性，可以被看作一种衍生产品 。事实上，任何一个交易对手的CVA都比与该对手交易的任一具体衍生产品更复杂，因为CVA涉及该对手下所有交易的净风险敞口综合 。CVA随市场变化而波动，例如基础市场利率、汇率、商品价格变化会影响敞口$v_i$的大小，信用利差变化会影响违约概率$q_i$，从而引起CVA价值的变动 。
<!-- bilingual-en:start -->
CVA depends on both market and credit factors and therefore has substantial **[[市场风险|market risk]]** of its own. It can be treated as a derivative on the entire counterparty portfolio rather than on one trade. Interest rates, exchange rates, and commodity prices change exposure $v_i$, while the counterparty's credit spread changes default probabilities $q_i$; both channels move CVA.
<!-- bilingual-en:end -->

与传统衍生品类似，我们可以定义CVA对各种风险因子的敏感度（**希腊值**）。例如，CVA对利率的**[[Rho|Delta]]**衡量利率变动引起的CVA变化，对汇率、大宗商品价格的Delta衡量对应市场价格变动对CVA的影响；CVA对信用利差的敏感度类似于**信用Vega**，因为违约概率$q_i$由信用利差曲线决定，利差平移将影响CVA 。在实际管理中，一些先进银行会对主要对手CVA进行风险因素分解，计算Delta、[[Gamma]]、Vega等，以用于对冲和风险控制 。
<!-- bilingual-en:start -->
As with other derivatives, CVA sensitivities can be expressed as Greeks. Interest-rate, foreign-exchange, and commodity **delta** measure how exposure-driven CVA changes when the relevant market moves. Credit-spread sensitivity measures how changes in the counterparty's spread curve alter $q_i$ and hence CVA. Banks may decompose material CVA positions into delta, gamma, vega, and related factors for hedging and risk control.
<!-- bilingual-en:end -->

<!-- greeks-source-note:start GB-CVA01 -->
> [!note] 校注：利差水平与利差波动率是不同输入
> 原文的利率 Delta 使用“一阶因子敏感度”的广义叫法，具体定义须指明利率曲线与冲击口径，可参照 [[Rho]]。对信用利差**水平**的导数是 credit-spread delta，不是信用 Vega；[[Vega]] 针对波动率输入。[Basel MAR50.68–50.69](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/50/inforce/2023-01-01/published/2020-07-08) 也分别规定 reference credit-spread delta 与 vega。这里仅校正输入与名称，不把普通 Greek 报价单位当成监管敏感度的全部定义。
> <!-- bilingual-en:start -->
> Interest-rate delta here uses the broad convention of first-order factor sensitivity; specify its curve and bump, as in [[Rho|Rho]]. A derivative with respect to the credit-spread **level** is credit-spread delta; [[Vega|Vega]] concerns volatility. MAR50.68–50.69 distinguishes these inputs. This terminology clarification does not substitute ordinary Greek quote units for regulatory sensitivity definitions.
> <!-- bilingual-en:end -->
<!-- greeks-source-note:end -->

**监管资本的版本边界：**2010 年 Basel III 首次引入 CVA 风险资本，课程所述以信用利差 VaR/增量风险为核心的 Advanced Approach 属于历史方法。当前 Basel CVA 框架使用 BA-CVA、在获批条件下使用 SA-CVA，或对不重要 CVA 风险的交易采用 100% 对手方信用风险 RWA 的有限重要性处理；它更系统地处理信用利差、利率、汇率等影响 CVA 的风险因子。CVA 风险资本、对手方违约资本与会计 CVA 的区别见 [[三类 CVA 口径]]。
<!-- bilingual-en:start -->
**Version boundary for CVA capital:** The 2010 Basel III framework introduced CVA risk capital. The course's VaR/IRC-based Advanced Approach is historical. The current Basel framework uses BA-CVA, SA-CVA with approval, or the limited materiality treatment of assigning 100% of counterparty-credit-risk RWA to qualifying immaterial CVA positions. It more systematically captures credit-spread, interest-rate, foreign-exchange and other risk factors affecting CVA.
<!-- bilingual-en:end -->

在课程所述的 **2010 年历史 CVA 框架**中，资本计量主要围绕信用利差，银行即使对利率、汇率等影响敞口 $v_i$ 的因子进行对冲，也可能在交易簿增加市场风险资本，却不能在当时的 CVA 资本中得到同等认可。这一历史缺口曾引发“惩罚全面 CVA 对冲”的争议，也是后来重写 CVA 框架、扩大风险因子和合格对冲识别的重要背景。它不能再作为当前 BA-CVA/SA-CVA 一律忽略非信用因子的描述。
<!-- bilingual-en:start -->
Under the **historical 2010 CVA framework described in this course**, banks could model and hedge CVA sensitivity to rates, exchange rates and other drivers of $v_i$, while regulatory CVA capital mainly recognised credit-spread-driven risk. A market hedge could therefore add trading-book capital without an offsetting CVA-capital benefit. This historical criticism helped motivate later revisions; it is not a description of current BA-CVA or SA-CVA as ignoring all non-credit factors.
<!-- bilingual-en:end -->

## 9. 错向风险（Wrong-Way Risk）与正向风险（Right-Way Risk）
<!-- bilingual-en:start -->
*9. Wrong-Way Risk and Right-Way Risk*
<!-- bilingual-en:end -->

离散公式中 $q_i$ 和 $v_i$ 已是汇总数，不能把二者说成随机变量并计算相关。[[错向风险|错向风险]]的依赖对象是随机敞口 $E(t)$ 与对手违约事件、违约强度或共同信用状态：
<!-- bilingual-en:start -->
In the discrete shortcut, $q_i$ and $v_i$ are already aggregate numbers and are not random variables whose correlation can be measured. [[错向风险|Wrong-way risk]] concerns dependence between random exposure $E(t)$ and the counterparty's default event, default intensity or common credit state:
<!-- bilingual-en:end -->

- **错向风险（WWR）：**高随机敞口倾向与对手违约或高违约强度同时发生。**General WWR** 由共同宏观、行业或市场因子同时推高敞口和违约风险；**Specific WWR** 来自某交易、抵押品或参考实体与该对手之间的直接法律或经济联系。
<!-- bilingual-en:start -->
- **Wrong-way risk (WWR):** high random exposure tends to coincide with counterparty default or high default intensity. **General WWR** arises from common macroeconomic, industry or market factors that raise both exposure and default risk. **Specific WWR** arises from a direct legal or economic link between the trade, collateral or reference entity and that counterparty.
<!-- bilingual-en:end -->
    
- **正向风险（Right-Way Risk）：**高随机敞口倾向与较低对手违约风险同时发生，是有利的依赖关系。
<!-- bilingual-en:start -->
- **Right-way risk:** high random exposure tends to coincide with lower counterparty default risk, a favourable dependence relationship.
<!-- bilingual-en:end -->
    

错向风险情形下，最糟糕的情况（对手违约）往往发生在本行敞口大的时候，导致损失可能远超独立假设下的估计；正向风险则是一种有利相关，可部分缓解信用损失。
<!-- bilingual-en:start -->
Under wrong-way risk, counterparty default is most likely when the bank's exposure is large, so losses can greatly exceed an independence-based estimate. Right-way risk is the favourable opposite relationship and can reduce expected credit loss.
<!-- bilingual-en:end -->

**CDS 边界：**银行买入保护时，参考实体恶化会提高 CDS 对银行的价值和正敞口。但交易方向本身不足以构成 WWR；只有保护卖方的信用也因与参考实体的直接经济联系、赔付压力或共同冲击而同时恶化时，才是 WWR。向与参考实体无关且信用稳健的卖方买保护，不会仅因为“买 CDS”就自动成为 WWR。
<!-- bilingual-en:start -->
**CDS boundary:** when a bank buys protection, deterioration of the reference entity raises the CDS value and the bank's positive exposure. Trade direction alone is insufficient for WWR. It is WWR only if the protection seller's own credit also deteriorates because of a direct economic link to the reference entity, payment pressure, or a common shock. Buying protection from an unrelated, creditworthy seller is not automatically WWR merely because the bank bought CDS protection.
<!-- bilingual-en:end -->

**正向风险示例：**客户用远期对冲自身实体业务时，若远期亏损使银行正敞口上升，客户的现货业务可能同时获利并提高履约能力，这是有经济机制支撑的 right-way relationship。
<!-- bilingual-en:start -->
**Right-way-risk example:** if a client uses a forward to hedge its underlying business, a forward loss that raises the bank's positive exposure may coincide with profit in the client's spot business and greater ability to perform. That economic mechanism supports a right-way relationship.
<!-- bilingual-en:end -->

面对错向风险，定量化和缓释是难点之一。Basel 的 $\alpha$ 不是“把 CVA 乘大”的系数：在内部模型法（IMM）下，违约风险 EAD 以 $\alpha\times\text{Effective EPE}$ 形成，监管默认 $\alpha=1.4$；经批准的内部估计仍受 1.2 下限约束。当前 SA-CCR 也在 $EAD=1.4(RC+PFE)$ 中使用 1.4，但这里的 PFE 是监管 add-on，不是第 5 节的日期分位数 PFE；这是另一套非模型标准法公式。两者都不能替代对具体错向风险的识别和直接处理；还须用依赖模型、限额、抵押品和交易准入管理敞口与对手违约同时恶化的情形。
<!-- bilingual-en:start -->
Wrong-way risk is difficult to quantify and mitigate. Basel's $\alpha$ is not a multiplier on CVA. Under the internal model method (IMM), default-risk EAD is $\alpha\times\text{Effective EPE}$, with supervisory $\alpha=1.4$ and a 1.2 floor for an approved internal estimate. Current SA-CCR separately uses $EAD=1.4(RC+PFE)$, where PFE is a supervisory add-on rather than the date-specific quantile PFE in Section 5. Neither scaling convention replaces identification and direct treatment of specific wrong-way risk through dependence modelling, limits, collateral and transaction controls.
<!-- bilingual-en:end -->

## 10. DVA的会计处理与争议
<!-- bilingual-en:start -->
*10. Accounting Treatment of DVA and the Controversy Around It*
<!-- bilingual-en:end -->

**[[DVA与双边估值|DVA]]（Debit Value Adjustment）**是交易商自身违约风险对本方负敞口的调整，在银行视角下加回 clean value。只有 CVA 与 DVA 都使用 first-to-default、同一 close-out 和一致依赖模型时，才可写成 $V_{\text{bil}}=V_0-\mathrm{CVA}+\mathrm{DVA}$。直观上，DVA 是银行违约时可能少支付负债的现值。
<!-- bilingual-en:start -->
**[[DVA与双边估值|DVA]] (debit valuation adjustment)** is the adjustment for the dealer's own default risk on its negative exposure and is added from the bank's perspective. The shorthand $V_{\text{bil}}=V_0-\mathrm{CVA}+\mathrm{DVA}$ is valid only when both adjustments use first-to-default ordering, one close-out convention and a consistent dependence model. DVA is the present value of liability the bank may not pay in its own default.
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
    
- **远期合约（Forward）：**远期在中间敞口日 $t$ 的市值取决于当时市场远期价与合同交割价之差。对多头远期，正敞口是该市值的正部分。在本文黄金远期的简化 Black 设定中，从今天到敞口日的随机方差累积为 $\sigma^2t$，而该日市值再按交易剩余现金流的贴现口径计值。
<!-- bilingual-en:start -->
- **Forward contract:** value at an intermediate exposure date $t$ depends on the then-current market forward price relative to contractual delivery price. For a long forward, positive exposure is the positive part of that value. In the simplified Black setup used in the gold example below, variance accumulates from today to exposure date as $\sigma^2t$; the date-$t$ mark-to-market is then valued using the discounting convention for the remaining contractual cash flow.
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
    
- **CVA计算：**两个敞口日的贴现期望正敞口为 $v_1^+=132.379247$、$v_2^+=186.645238$ mn USD。在题设的独立性和确定回收率近似下，$\mathrm{UCVA}=0.7[0.02(132.379247)+0.03(186.645238)]=5.772859$ mn USD。
<!-- bilingual-en:start -->
- **CVA:** Time-zero discounted expected positive exposures are $v_1^+=132.379247$ and $v_2^+=186.645238$ million. Under the exercise's independence and deterministic-recovery approximation, $\mathrm{UCVA}=0.7[0.02(132.379247)+0.03(186.645238)]=5.772859$ million.
<!-- bilingual-en:end -->
    
- **信用调整后价值：**$V_{\text{adjusted}}=90.483742-5.772859=84.710882$ mn USD。
<!-- bilingual-en:start -->
- **Credit-adjusted value:** $V_{\text{adjusted}}=90.483742-5.772859=84.710882$ million.
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

**解析：**本题已给出贴现敞口，并隐含违约与敞口独立、合同回收率与经济 LGD 一致，因此可用离散近似 $\text{UCVA}=(1-R)\sum_i q_i v_i$。$q_1=0.05$，$q_2=0.07$，$v_1=1000$万，$v_2=800$万，$1-R=0.6$：
<!-- bilingual-en:start -->
**Analysis:** The question supplies discounted exposure and implicitly assumes independence between exposure and default and that contractual recovery matches economic LGD. The shortcut $\text{UCVA}=(1-R)\sum_iq_iv_i$ therefore applies. Here $q_1=0.05$, $q_2=0.07$, $v_1=10$ million yuan, $v_2=8$ million yuan, and $1-R=0.6$:
<!-- bilingual-en:end -->

$$
\mathrm{CVA}=0.6\times(0.05\times1000+0.07\times800)\text{万元}=63.6\text{万元}.
$$
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

**问题2：PFE 与跨日期最大 PFE** – 假设某银行通过 1 万次模拟得到未来日期 $t$ 的敞口分布，该日 97.5% PFE 为 200 万美元，而各网格日期的 97.5% PFE 最大值为 250 万美元。两个数分别表示什么？CVA 是否直接使用 PFE？
<!-- bilingual-en:start -->
**Question 2: PFE and maximum PFE across dates.** A bank runs 10,000 simulations. PFE at date $t$ has a 97.5th percentile of USD 2 million, while the largest date-specific 97.5% PFE is USD 2.5 million. What does each number mean, and is PFE used directly in CVA?
<!-- bilingual-en:end -->

**解析：**200 万美元是 $t$ 日的 $\mathrm{PFE}_{97.5\%}(t)$：97.5% 的该日模拟敞口不超过它。250 万是 $\max_k\mathrm{PFE}_{97.5\%}(t_k)$，即所有网格日期的 97.5% 分位数中最大的一个。它不是 $Q_{97.5\%}(\max_tE(t))$，也不是“最坏情形”。
<!-- bilingual-en:start -->
**Analysis:** USD 2 million is $\mathrm{PFE}_{97.5\%}(t)$: 97.5% of simulated exposures at that date are no greater. USD 2.5 million is $\max_k\mathrm{PFE}_{97.5\%}(t_k)$, the largest date-specific 97.5th percentile. It is neither $Q_{97.5\%}(\max_tE(t))$ nor a worst case.
<!-- bilingual-en:end -->

CVA 使用违约时随机正敞口的期望，不直接把 PFE 加总入公式。估值 CVA 通常使用 $\mathbb Q$，但 PFE 所用测度取决于限额、经济资本、监管或压力测试用途；PFE 本身不定义为真实世界测度。
<!-- bilingual-en:start -->
CVA uses expected random positive exposure at default; PFE is not added directly to the formula. Valuation CVA normally uses $\mathbb Q$, whereas the measure used for PFE depends on whether the purpose is limits, economic capital, regulation or stress testing. PFE is not defined as a real-world-measure quantity.
<!-- bilingual-en:end -->

**答案要点：**200 万是特定日期的 97.5% PFE；250 万是各日期 97.5% PFE 的最大值，但不是路径最大敞口的分位数或最坏情形。CVA 用期望敞口，不直接用 PFE。
<!-- bilingual-en:start -->
**Answer:** USD 2 million is 97.5% PFE at one date; USD 2.5 million is the maximum of those date-specific PFEs, not a path-maximum quantile or worst case. CVA uses expected exposure, not PFE directly.
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

- **场景 A 在题设的关联确实存在时**是错向风险：参考实体恶化提高银行的随机正敞口，同一冲击又通过直接经济联系或赔付压力提高保护卖方的违约风险。若题目只说“银行买 CDS”而没有卖方与参考实体的联系或共同冲击，则不能单凭交易方向判定 WWR。
<!-- bilingual-en:start -->
- **Scenario A is WWR given the stated link.** Reference-entity deterioration raises the bank's random positive exposure and, through a direct economic link or payment pressure, also raises the protection seller's default risk. If a question said only that the bank bought CDS protection, with no link or common shock affecting the seller, trade direction alone would not establish WWR.
<!-- bilingual-en:end -->
    
- **场景B**体现了正向风险的特征。贸易公司用远期合约对冲其现货业务风险：如果汇率走势导致公司在远期上亏损（银行敞口上升，因为公司欠银行款增多），则公司现货业务可能因相反的汇率变动受益，获得额外利润 。这额外收益增强了公司的财务状况，降低了其违约概率 。也就是说，当银行的信用敞口升高时，对手方反而更不易违约，这是**正向风险**的情况。
<!-- bilingual-en:start -->
- **Scenario B is right-way risk.** The trading company uses the forward to hedge its underlying business. A currency move that creates a loss on the forward, and hence greater exposure for the bank, may create an offsetting profit in the client's spot business. That profit strengthens the client and lowers its default probability when the bank's exposure is high.
<!-- bilingual-en:end -->
    

因此，在题设联动机制成立时，场景 A 为错向风险，场景 B 为正向风险。
<!-- bilingual-en:start -->
Given the stated dependence mechanisms, Scenario A is wrong-way risk and Scenario B is right-way risk.
<!-- bilingual-en:end -->

**答案要点：**场景 A 的共同冲击使随机敞口与对手违约风险同时上升，所以是 WWR；场景 B 中实体业务利润提高客户履约能力，所以是 right-way risk。
<!-- bilingual-en:start -->
**Key points:**<br>
Scenario A: **wrong-way risk** because the stated common mechanism raises exposure and counterparty default risk together.<br>
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
> - 若违约发生于起始后 0.5 年或 1.5 年，合约按 **正市值** 现金结算
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

在本题设定下，时点 $t_i$ 的随机方差从今天累积到 $t_i$，而 $v_i^+$ 是贴现到 0 时点的期望正敞口：
$$v_i^+ = e^{-rT}Q\bigl[F_0N(d_{1,i})-KN(d_{2,i})\bigr],$$
$$d_{1,i}=\frac{\ln(F_0/K)+\tfrac12\sigma^2t_i}{\sigma\sqrt{t_i}},\qquad d_{2,i}=d_{1,i}-\sigma\sqrt{t_i}.$$
<!-- bilingual-en:start -->
Variance accumulates from today to exposure date $t_i$. The time-zero discounted expected positive exposure is $v_i^+=e^{-rT}Q[F_0N(d_{1,i})-KN(d_{2,i})]$, with $d_{1,i}=[\ln(F_0/K)+\tfrac12\sigma^2t_i]/(\sigma\sqrt{t_i})$ and $d_{2,i}=d_{1,i}-\sigma\sqrt{t_i}$.
<!-- bilingual-en:end -->

| 违约时点 | $t_i$ | $d_{1,i}$ | $d_{2,i}$ | $v_i^+$ (USD m) |
|-----------|---------|-----------|-----------|---------------|
| 年 1 中点 | 0.5 | 0.527 | 0.386 | 132.38 |
| 年 2 中点 | 1.5 | 0.386 | 0.141 | 186.65 |
<!-- bilingual-en:start -->
| Default node | $t_i$ | $d_{1, i}$ | $d_{2, i}$ | $v_i^+$ (USD m) |
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
$$\text{UCVA}=(1-R)\bigl(q_1v_1^++q_2v_2^+\bigr)
           =0.70\bigl(0.02\times132.379247 + 0.03\times186.645238\bigr)\approx 5.772859\,\text{m}$$
3  信用风险调整后的远期价值  
<!-- bilingual-en:start -->

&nbsp;
**3.** Calculate the Counterparty-Credit-Adjusted Forward Value<br>
<!-- bilingual-en:end -->

忽略信用风险的理论价值  
$$V_0 =(F_0-K)Qe^{-rT} =(1 600-1 500)\times10^6e^{-0.05\times2}\approx 90.483742\,\text{m}$$
计入 CVA 后  
$$V_{\text{adjusted}} = V_0 - \text{UCVA} \approx 90.483742 - 5.772859 = 84.710882\,\text{m}$$
**要点总结**  
1.   方差累积到敞口日 $t_i$，计算贴现到 0 时点的 $v_i^+$；
2.   题设离散 UCVA 简化还隐含敞口与违约独立、回收率确定；
3.   远期合约基准价值减去 $\text{CVA}$ 得到考虑对手信用风险后的公允价值。  
<!-- bilingual-en:start -->
First calculate clean value; then include CVA.
**Key points**
**1.** Accumulate variance to exposure date $t_i$ and calculate time-zero discounted $v_i^+$.<br>
**2.** The discrete UCVA shortcut also assumes independence between exposure and default and deterministic recovery.<br>
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
- 第 1 年（前 12 个月）每月 **无条件** 违约概率为 0.001667
- 第 2 年（后 12 个月）每月 **无条件** 违约概率为 0.0025
若违约发生，按题设简化回收口径，买方对正敞口回收 30%、损失 70%，因此 $(1-R)=70\%$ 是 UCVA 的损失权重。
要求：  
1. 对每个可能违约月 $t_i$，计算贴现到 0 时点的期望正敞口 $v_i^+$：
   $$
   v_i^+ = e^{-rT}Q\bigl[F_0N(d_{1,i})-KN(d_{2,i})\bigr],\quad
   d_{1,i}=\frac{\ln(F_0/K)+\tfrac12\sigma^2t_i}{\sigma\sqrt{t_i}},\quad
   d_{2,i}=d_{1,i}-\sigma\sqrt{t_i}.
   $$
2. 计算两年期 **信用估值调整**  
$$\text{UCVA} = (1-R)\sum_{i=1}^{24} q_i\,v_i^+$$
   其中 $q_i$ 为对应月份的无条件违约概率。  
3. 给出考虑信用风险后的远期合约公允价值  
   $$V_{\text{adjusted}} = (F_0-K)Qe^{-rT} - \text{UCVA}$$
提示：方差积累到敞口日 $t_i$，不是剩余期限 $T-t_i$。$Q=1{,}000{,}000$ 盎司，所有结果以美元计。

**解答**
一、输入参数  
- 名义数量 `Q = 1 000 000 oz`
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
0.001667,& i = 1,\dots,12,\\[2pt]
0.002500,& i = 13,\dots,24
\end{cases}
$$

---

 二、逐月正向敞口 $v_i^+$

$$
\begin{aligned}
d_{1,i} &= \frac{\ln(F_0/K)+\tfrac12\sigma^{2}t_i}
                {\sigma\sqrt{t_i}},\\
d_{2,i} &= d_{1,i}-\sigma\sqrt{t_i},\\[6pt]
v_i^+ &= e^{-rT}Q\Bigl[F_0N(d_{1,i})-KN(d_{2,i})\Bigr].
\end{aligned}
$$

24 个节点结果  

| 月 $i$ | $t_i$ (年) | $v_i^+$ (mn USD) | $q_i$ | $(1-R)q_i v_i^+$ (mn USD) |
|:--:|:--:|------:|-------:|--------:|
|  1 | 0.0417 | 91.875 | 0.001667 | 0.107 |
|  2 | 0.1250 | 100.205 | 0.001667 | 0.117 |
|  3 | 0.2083 | 108.541 | 0.001667 | 0.127 |
|  4 | 0.2917 | 116.106 | 0.001667 | 0.135 |
|  5 | 0.3750 | 123.004 | 0.001667 | 0.144 |
|  6 | 0.4583 | 129.367 | 0.001667 | 0.151 |
|  7 | 0.5417 | 135.294 | 0.001667 | 0.158 |
|  8 | 0.6250 | 140.860 | 0.001667 | 0.164 |
|  9 | 0.7083 | 146.123 | 0.001667 | 0.171 |
| 10 | 0.7917 | 151.126 | 0.001667 | 0.176 |
| 11 | 0.8750 | 155.903 | 0.001667 | 0.182 |
| 12 | 0.9583 | 160.482 | 0.001667 | 0.187 |
| 13 | 1.0417 | 164.884 | 0.002500 | 0.289 |
| 14 | 1.1250 | 169.129 | 0.002500 | 0.296 |
| 15 | 1.2083 | 173.232 | 0.002500 | 0.303 |
| 16 | 1.2917 | 177.206 | 0.002500 | 0.310 |
| 17 | 1.3750 | 181.061 | 0.002500 | 0.317 |
| 18 | 1.4583 | 184.809 | 0.002500 | 0.323 |
| 19 | 1.5417 | 188.457 | 0.002500 | 0.330 |
| 20 | 1.6250 | 192.013 | 0.002500 | 0.336 |
| 21 | 1.7083 | 195.483 | 0.002500 | 0.342 |
| 22 | 1.7917 | 198.873 | 0.002500 | 0.348 |
| 23 | 1.8750 | 202.188 | 0.002500 | 0.354 |
| 24 | 1.9583 | 205.433 | 0.002500 | 0.360 |

$\sum_i v_i^+=3{,}791.653669$ mn USD。
三、信用估值调整 (CVA)

$$
\boxed{\text{UCVA}
      = (1-R)\sum_{i=1}^{24} q_i v_i^+
      = \underline{\$5.726408\ \text{million}}}
$$
 四、考虑信用风险后的远期价值  

1. **无信用风险价值**  
$$
V_{\text{clean}}
  = (F_0-K)\,e^{-rT}\,Q
  = 100 \times e^{-0.10}\times 10^6
  = \underline{\$90.484\ \text{million}}
$$

2. **调整后公允价值**  
$$
\boxed{V_{\text{adjusted}}
       = V_{\text{clean}} - \text{UCVA}
       = 90.483742 - 5.726408
       = \underline{\$84.757334\ \text{million}}}
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
| 违约时清算 | 银行视角的负敞口 $V^-=\max(-V,0)$ 是对手的正债权；债权人回收 40%，未偿 60% 构成银行债务减免 |
**违约假设**  
银行可能在每个月的中点违约（即 $t_i = 0.5,\,1.5,\dots ,23.5$ 个月，共 24 个节点）。  
- 第 1 年的每月**无条件**违约概率 $q_i = 0.001$  
- 第 2 年的每月**无条件**违约概率 $q_i = 0.001$  

> 要求  
> 1. 从银行视角计算每个节点贴现到 0 时点的期望负敞口
>    $$v_i^- = e^{-rT}Q\bigl[KN(-d_{2,i})-F_0N(-d_{1,i})\bigr],$$
>    $$d_{1,i}=\frac{\ln(F_0/K)+\tfrac12\sigma^2t_i}{\sigma\sqrt{t_i}},\qquad d_{2,i}=d_{1,i}-\sigma\sqrt{t_i}.$$
> 2. 计算银行 **债务估值调整**（DVA）：  
>    $$\text{DVA} = (1-R_{\text{bank}})\sum_{i=1}^{24} q_i\,v_i^-$$
> 3. 给出计入 DVA 后远期合约的公允价值  
>    $$V_{\text{clean+DVA}} = (F_0-K)Qe^{-rT} + \text{DVA}$$
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

Ⅱ. 每月银行负敞口的公式
$$
\begin{aligned}
d_{1,i}&=\frac{\ln(F_0/K)+\tfrac12\sigma^2t_i}{\sigma\sqrt{t_i}},\\
d_{2,i}&=d_{1,i}-\sigma\sqrt{t_i},\\
v_i^-&=e^{-rT}Q\left[KN(-d_{2,i})-F_0N(-d_{1,i})\right].
\end{aligned}
$$

这是银行视角的 put-like 期望负敞口；方差积累到 $t_i$，$e^{-rT}$ 把最终现金流口径折现到 0 时点。

Ⅲ. 首月校验

$t_1=0.041667$，$d_{1,1}=1.601277$，$d_{2,1}=1.560452$，因而 $v_1^-=1.391037$ mn USD。

Ⅳ. 24 个月完整数值（货币单位：mn USD）

| 月 $i$ | $t_i$ (年) | $d_{1,i}$ | $d_{2,i}$ | $v_i^-$ | $0.6q_iv_i^-$ |
|:--:|:--:|:--:|:--:|------:|------:|
|  1 | 0.0417 | 1.601 | 1.560 | 1.391 | 0.001 |
|  2 | 0.1250 | 0.948 | 0.877 | 9.721 | 0.006 |
|  3 | 0.2083 | 0.753 | 0.661 | 18.057 | 0.011 |
|  4 | 0.2917 | 0.652 | 0.544 | 25.622 | 0.015 |
|  5 | 0.3750 | 0.588 | 0.466 | 32.520 | 0.020 |
|  6 | 0.4583 | 0.544 | 0.409 | 38.883 | 0.023 |
|  7 | 0.5417 | 0.512 | 0.365 | 44.810 | 0.027 |
|  8 | 0.6250 | 0.487 | 0.329 | 50.377 | 0.030 |
|  9 | 0.7083 | 0.468 | 0.299 | 55.640 | 0.033 |
| 10 | 0.7917 | 0.452 | 0.274 | 60.642 | 0.036 |
| 11 | 0.8750 | 0.439 | 0.251 | 65.419 | 0.039 |
| 12 | 0.9583 | 0.428 | 0.232 | 69.998 | 0.042 |
| 13 | 1.0417 | 0.418 | 0.214 | 74.400 | 0.045 |
| 14 | 1.1250 | 0.410 | 0.198 | 78.645 | 0.047 |
| 15 | 1.2083 | 0.403 | 0.184 | 82.748 | 0.050 |
| 16 | 1.2917 | 0.398 | 0.170 | 86.722 | 0.052 |
| 17 | 1.3750 | 0.392 | 0.158 | 90.578 | 0.054 |
| 18 | 1.4583 | 0.388 | 0.146 | 94.325 | 0.057 |
| 19 | 1.5417 | 0.384 | 0.136 | 97.974 | 0.059 |
| 20 | 1.6250 | 0.381 | 0.126 | 101.529 | 0.061 |
| 21 | 1.7083 | 0.378 | 0.116 | 104.999 | 0.063 |
| 22 | 1.7917 | 0.375 | 0.107 | 108.389 | 0.065 |
| 23 | 1.8750 | 0.373 | 0.099 | 111.704 | 0.067 |
| 24 | 1.9583 | 0.371 | 0.091 | 114.949 | 0.069 |

$\sum_i v_i^-=1{,}620.043866$ mn USD，因此
$$
\boxed{\mathrm{DVA}=0.60\times0.001\times1{,}620.043866
=\$0.972026\ \text{million}.}
$$

Ⅴ. 公允价值与经济解释

$$
V_{\text{clean}}=(F_0-K)Qe^{-rT}=\$90.483742\ \text{million},
$$
$$
\boxed{V_{\text{clean+DVA}}=90.483742+0.972026
=\$91.455768\ \text{million}.}
$$

$R_{\text{bank}}=40\%$ 表示债权人可回收银行负债的 40%，余下 60% 是债权人损失，也是银行的潜在债务减免。银行视角下 DVA 加回 clean value。若只为展示算术而把 20.13 的校正 UCVA 与本题 DVA 机械叠加，得 $90.483742-5.726408+0.972026=85.729360$ mn USD；这不是完整双边估值，真正双边式必须使用 first-to-default 与同一 close-out。

## 20.15  
>[!question] 
考虑某欧式看涨期权，期权标的资产为某不付股息的股票，股票的价格为 52 美元，期权执行价格为 50 美元，  
无风险利率为 5%，波动率为 30%，期权期限为 1 年。假定回收率为 0%，无担保品，无其他交易，且卖方到期生存/违约状态与到期期权赔付 $H=(S_T-K)^+$ 独立。
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

在题设的生存状态与 $H$ 独立、零回收假设下，
$$C_{\text{buy}}=(1-0.02)\times8.39\approx\$8.22.$$

3. 改为到期支付权利金  

令 $H=(S_T-K)^+$。按本题的**特殊假设**，卖方违约时期权赔付与延后权利金两条腿都取消。卖方存活因子因此在两腿公允条件中相消：
$$
(1-p_S)e^{-rT}\mathbb E[H]=(1-p_S)e^{-rT}X
\quad\Longrightarrow\quad Xe^{-rT}=C_0\approx8.39.
$$
所以 $X=8.39e^{0.05}\approx\$8.82$，不是 \$8.64。延后支付使两腿在卖方违约时同时取消，从而降低买方的单边敞口。

4. 买方到期违约概率 1 %，卖方仍 2 %  

买方也可违约时，卖方面临延后权利金收不回的风险。一般 close-out 不能把两条腿分别乘生存率；应先净额为 $H-X$，再根据谁先违约处理 $(H-X)^+$ 与 $(H-X)^-$。若未给出双方违约时间、相依性、回收率和 close-out 约定，就不存在唯一的双边价格。

关键步骤  
* 用 Black–Scholes 计算无违约价值；  
* CVA ≈ ([[违约概率口径|违约概率]]) × $C_0$ × (1–回收率)；
* 在“卖方违约则两腿都取消”的特殊假设下，卖方存活因子相消，所以 $X\approx\$8.82$；
* 一般双边情形要对 $H-X$ 做 close-out 净额并使用 first-to-default，不能由边际违约概率直接得出唯一价格。

## 20.16  
>[!question] 
假设一家银行发行了 3 年期无风险固定收益券的收益率加 210 个基点的浮息票据，  
由布莱克–斯科尔斯–默顿公式得出的期权价格为 4.10 美元。  
如果你以银行作为期权卖方，你愿意支付的实际价格是多少？  

| 步骤 | 关键要点 | 简述 |
|------|----------|------|
| 1️⃣ 识别资金成本 | 银行自身的融资成本 = **[[无风险利率口径|无风险利率]] + 210 bps** → 记作 $s = 2.10\%$ |
| 2️⃣ 理论定价输入 | B-S-M 给出的风险中性价值 $C_{BSM}= \$4.10$，已按无风险利率 $r$ 折现 |
| 3️⃣ 调整贴现因子 | 作为卖方使用 **自身资金成本** 计价：<br/> 连续复利：$DF = e^{-sT} = e^{-0.021\times3}=0.93896$<br/> 或年度复利：$DF = (1+s)^{-T}=(1.021)^{-3}=0.9392$ |
| 4️⃣ 求得愿付价格 | $C_{\text{internal}} = C_{BSM} \times DF \approx 4.10 \times 0.939 \approx \$3.85$ |
| 5️⃣ 结论 | 在题设的融资成本启发式下，银行内部购入/对冲参考价约为 **\$3.85**；它不是一般无套利价格结论 |

> [!warning] 口径边界
> $4.10\times e^{-0.021\times3}\approx3.85$ 只是题目采用的 funding-cost heuristic（融资成本启发式）。是否、如何把自身融资成本纳入公允价值，取决于机构估值框架、复制策略、抵押品和会计口径；不能把“理论价再乘一次自身贴现因子”当作通用无套利定价规则。
<!-- bilingual-en:start -->
A bank agrees to buy 1,000,000 ounces of gold from a mining company for USD 1,500 per ounce at the end of month 24. The current two-year forward price is $F_0=1600$, annualised log-price volatility is $\sigma=20\%$, the continuously compounded risk-free rate is $r=5\%$, and recovery is $R=30\%$. Default may occur at the midpoint of any month. Each month in year one has unconditional default probability 0.001667; each month in year two has probability 0.0025. Under the exercise's simplified recovery convention, the buyer recovers 30% and loses 70% of positive exposure; 70% is the UCVA loss weight, not the cash recovery.

The tasks are to calculate time-zero discounted expected positive exposure $v_i^+$ at every monthly node, two-year UCVA and counterparty-credit-adjusted fair value.

**Solution: inputs and setup**

- Notional: $Q=1{,}000{,}000$ oz
- Maturity: $T=24$ months, or two years
- Forward and delivery prices: $F_0=1600$ and $K=1500$ USD/oz
- Volatility: $\sigma=20\%$ per year
- Continuously compounded risk-free rate: $r=5\%$
- Recovery and loss rates: $R=30\%$ and $1-R=70\%$
- Default nodes: $t_i=(i-0.5)/12$, for $i=1,\ldots,24$
- Unconditional monthly default probability: 0.001667 for $i=1,\ldots,12$ and 0.0025 for $i=13,\ldots,24$

Variance accumulates from today to exposure date $t_i$, not over $T-t_i$. The time-zero discounted expected positive exposure is
$$
v_i^+=e^{-rT}Q[F_0N(d_{1,i})-KN(d_{2,i})],\qquad
d_{1,i}=\frac{\ln(F_0/K)+\tfrac12\sigma^2t_i}{\sigma\sqrt{t_i}},\quad
d_{2,i}=d_{1,i}-\sigma\sqrt{t_i}.
$$

| Month $i$ | $t_i$ (years) | $v_i^+$ (USD m) | $q_i$ | $(1-R)q_iv_i^+$ (USD m) |
|:--:|:--:|--:|--:|--:|
| 1 | 0.0417 | 91.875 | 0.001667 | 0.107 |
| 2 | 0.1250 | 100.205 | 0.001667 | 0.117 |
| 3 | 0.2083 | 108.541 | 0.001667 | 0.127 |
| 4 | 0.2917 | 116.106 | 0.001667 | 0.135 |
| 5 | 0.3750 | 123.004 | 0.001667 | 0.144 |
| 6 | 0.4583 | 129.367 | 0.001667 | 0.151 |
| 7 | 0.5417 | 135.294 | 0.001667 | 0.158 |
| 8 | 0.6250 | 140.860 | 0.001667 | 0.164 |
| 9 | 0.7083 | 146.123 | 0.001667 | 0.171 |
| 10 | 0.7917 | 151.126 | 0.001667 | 0.176 |
| 11 | 0.8750 | 155.903 | 0.001667 | 0.182 |
| 12 | 0.9583 | 160.482 | 0.001667 | 0.187 |
| 13 | 1.0417 | 164.884 | 0.002500 | 0.289 |
| 14 | 1.1250 | 169.129 | 0.002500 | 0.296 |
| 15 | 1.2083 | 173.232 | 0.002500 | 0.303 |
| 16 | 1.2917 | 177.206 | 0.002500 | 0.310 |
| 17 | 1.3750 | 181.061 | 0.002500 | 0.317 |
| 18 | 1.4583 | 184.809 | 0.002500 | 0.323 |
| 19 | 1.5417 | 188.457 | 0.002500 | 0.330 |
| 20 | 1.6250 | 192.013 | 0.002500 | 0.336 |
| 21 | 1.7083 | 195.483 | 0.002500 | 0.342 |
| 22 | 1.7917 | 198.873 | 0.002500 | 0.348 |
| 23 | 1.8750 | 202.188 | 0.002500 | 0.354 |
| 24 | 1.9583 | 205.433 | 0.002500 | 0.360 |

The independently recomputed sum is $\sum_i v_i^+=\$3{,}791.653669$ million. Thus $\mathrm{UCVA}=\$5.726408$ million. Clean value is $(F_0-K)e^{-rT}Q=\$90.483742$ million, so counterparty-credit-adjusted value is $\$84.757334$ million.

**Question 20.14: the bank's DVA**

Using the same two-year gold forward, suppose the bank may default at each monthly midpoint with unconditional probability 0.001 per month, and the creditor recovers 40% of the bank's liability. Calculate bank-perspective negative exposure, DVA and clean value plus DVA.

The parameters remain $F_0=1600$, $K=1500$, $Q=1{,}000{,}000$, $\sigma=20\%$, $r=5\%$, and $T=2$. Bank recovery is 40%, so creditor loss and the bank's potential debt forgiveness are 60%. Use the put-like bank-negative-exposure formula
$$
v_i^-=e^{-rT}Q[KN(-d_{2,i})-F_0N(-d_{1,i})],
$$
with the same $t_i$-based $d_{1,i},d_{2,i}$ as in Question 20.13.

For the first month, $d_{1,1}=1.601277$, $d_{2,1}=1.560452$ and $v_1^-=\$1.391037$ million. Across 24 nodes, $\sum_i v_i^-=\$1{,}620.043866$ million and $\mathrm{DVA}=0.60\times0.001\times1{,}620.043866=\$0.972026$ million.

DVA is added from the bank's perspective: $V_{\mathrm{clean+DVA}}=90.483742+0.972026=\$91.455768$ million. A purely mechanical combination with corrected unilateral CVA gives $90.483742-5.726408+0.972026=\$85.729360$ million, but a genuine bilateral value requires first-to-default ordering and one common close-out convention.

**Question 20.15: European call with bilateral default risk**

For a non-dividend-paying stock, $S_0=\$52$, $K=\$50$, $r=5\%$, $\sigma=30\%$, and $T=1$. Recovery and collateral are zero, there are no other trades, and the writer's maturity survival/default state is assumed independent of the terminal payoff $H=(S_T-K)^+$.

(a) With no default, Black--Scholes gives $d_1\approx0.4474$, $d_2\approx0.1474$, and $C_0\approx\$8.39$.

(b) If the option writer defaults at maturity with probability 2%, independence between writer survival and $H$, together with zero recovery, gives approximately $0.98\times8.39=\$8.22$.

(c) Under the exercise's special assumption that writer default cancels both the option payoff $H=(S_T-K)^+$ and the deferred premium, the writer-survival factor cancels between the two legs. Hence $Xe^{-rT}=C_0\approx\$8.39$ and $X\approx\$8.82$, not USD 8.64.

(d) If the buyer can also default, the writer faces loss of the deferred premium. A general close-out first nets the two legs as signed amount $H-X$ and then applies positive and negative parts according to which party defaults first. Without default-time dependence, recovery and close-out assumptions, no unique bilateral price follows from the two marginal default probabilities alone.

**Question 20.16: funding-spread adjustment**

A bank funds at the three-year risk-free fixed-income yield plus 210 bp, and the Black--Scholes--Merton option value is USD 4.10. The source treats the bank's own funding spread as an additional discount: $s=2.10\%$, $e^{-sT}=e^{-0.021\times3}\approx0.93896$, and $4.10\times0.939\approx\$3.85$. It therefore gives about USD 3.85 as the bank's internal maximum price for buying or hedging the option. This is a funding-cost heuristic, not by itself a general arbitrage-free valuation rule; its use depends on the institution's valuation framework.
<!-- bilingual-en:end -->
