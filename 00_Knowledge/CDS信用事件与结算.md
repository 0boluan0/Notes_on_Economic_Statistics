---
aliases:
  - "CDS 赔付由合同覆盖的信用事件、可交割义务以及拍卖或实物结算条款共同决定，重组与最便宜可交割选择会改变保护价值"
  - "CDS credit events and settlement"
student_os: knowledge-atom
atom_id: RM-CDS-002
atom_set: cds-pricing-and-basis
atom_type: mechanism-boundary
status: source-checked
mastery_state: unassessed
related:
  - "[[违约损失口径]]"
  - "[[DVA与双边估值]]"
leads_to:
  - "[[CDS两腿定价]]"
part_of:
  - "[[CDS定价与基差.canvas|CDS定价与基差]]"
---

# CDS 赔付由合同覆盖的信用事件、可交割义务以及拍卖或实物结算条款共同决定，重组与最便宜可交割选择会改变保护价值
<!-- bilingual-en:start -->
*A CDS payout is jointly determined by covered credit events, deliverable obligations, and auction or physical-settlement terms; restructuring and the cheapest-to-deliver choice change protection value*
<!-- bilingual-en:end -->

> [!summary] “发生了信用问题”还不足以推出“CDS 赔多少”
> CDS 是合同给付，不是对保护买方一切实际信用损失的概括性保险。先判断交易文件指定的信用事件是否按定义发生在参考实体及相关义务上，再确定哪些债务可交割或进入拍卖，最后按物理、拍卖或其他现金结算条款计算金额。事件菜单、重组条款、到期限制与可交割篮子任一变化，都可能改变同一参考实体上的保护价值。

## Covered Credit Event 是合同概念

2014 ISDA Credit Derivatives Definitions 提供的信用事件类别包括 Bankruptcy、Obligation Acceleration、Obligation Default、Failure to Pay、Repudiation/Moratorium、Restructuring，以及对某些 2014 Definitions 交易适用的 Governmental Intervention。它们是可选并带详细条件的合同类别，不是每份 CDS 自动覆盖的七项清单。

一项事件要触发具体交易，至少还要检查：Confirmation 或适用 Transaction Type 是否把该事件列为 Applicable；事件是否满足定义中的门槛、宽限期、受影响义务和公开信息要求；以及 Determinations Committee 决议或双边通知、backstop date 等结算条件是否适用。因此评级下调、债券价格暴跌、延期谈判或媒体把某主体称为“违约”，都不自动等于该 CDS 已发生 Covered Credit Event。

### 参考实体与交易对手不是同一角色

- **Reference Entity（参考实体）**是合约引用的信用风险主体；其合同覆盖信用事件可能触发保护结算。
- **Counterparty（交易对手）**是保护买方或保护卖方本身。保护卖方违约会产生替换成本、抵押品、close-out 与 [[DVA与双边估值|CVA/DVA]] 问题，但不会因为它是卖方就自动构成参考实体的信用事件。

只有在某一法律实体同时处于两个角色，且各自合同条件分别满足时，两个问题才可能同时出现；分析时仍要分开。

## Reference Obligation 与 Deliverable Obligation

Reference Obligation（参考义务）用来锚定交易所引用的债务及其优先级等条款；Deliverable Obligations（可交割义务）则是信用事件后买方可以实际交付、或可进入拍卖最终清单的合格债务集合。其他债务是否合格，要逐项满足 Confirmation、Physical Settlement Matrix 与 Definitions 指定的 Obligation Category 和 Deliverable Obligation Characteristics，例如债务层级、币种、非或有性、可转让性与到期限制。不能因为债券由参考实体发行，就推断它一定可交割。

## 三种结算路径

### 物理结算（Physical Settlement）

保护买方交付总面值不超过 CDS 名义本金的合格可交割义务，保护卖方支付相应的合同面值金额。若忽略币种换算、应计、封顶和债务本金调整，一张面值为 $N$、市价为面值比例 $p_d$ 的债务被交割时，买方收到的净保护价值近似为

$$
\Pi_{\mathrm{physical}}\approx N(1-p_d).
$$

买方必须在交割日拿到可交割义务，但不必在买入 CDS 时已经持有它，也不必证明自己在该债务上发生同额实际损失。

### 拍卖结算（Auction Settlement）

Big Bang 把 Determinations Committee 与拍卖结算写入标准文件框架。拍卖先围绕最终可交割义务清单形成实物结算请求和市场报价，再给出统一的 Auction Final Price。若最终价格以面值比例 $F$ 表示，标准拍卖结算给保护买方的现金金额为

$$
\boxed{\Pi_{\mathrm{auction}}=N\max(1-F,0)}.
$$

参与者可通过 Physical Settlement Request 在拍卖中实现与净 CDS 头寸同方向、且不超过净头寸的债务买卖；普通 auction-settled CDS 则直接按 Final Price 现金结算。Final Price 是特定最终清单和拍卖规则下的合同市场价格，不是法院最终清偿率、某位买方的实际损失，也不是监管模型的“真实 recovery”。

### 非拍卖现金结算（Cash Settlement）

若交易约定 Cash Settlement 而非 Auction Settlement，或拍卖失败后触发现金结算 fallback，Calculation Agent 按 Confirmation 指定的 Reference Obligation、报价或 valuation method 确定价格。不能把所有“现金结算”都简写为拍卖 Final Price；应先读主结算方法和 fallback。

## 重组条款、到期限制与 CTD 选择权

Restructuring 是否覆盖以及覆盖哪一变体，由交易的 doc clause/Transaction Type 决定。常见简码区分 No Restructuring（XR）、Full/Old Restructuring（CR）、Modified Restructuring（MR）和 Modified Modified Restructuring（MM）。MR 与 MM 在买方触发重组时对可交割债务施加不同的 maturity limitation 和 transferability 条件，因此同一次重组可能按 CDS 到期日形成不同 maturity buckets；Full/Old Restructuring 不采用这组特别的 MR/MM 到期限制，但仍受一般可交割条件约束。

上述按 CDS 剩余期限进入 maturity bucket 的限制针对 Buyer 触发；Seller 先行触发时并非机械沿用同一 bucket 和 maturity limitation，而应按适用 Definitions、Restructuring Supplement 及交易文件中的 seller-trigger 规则处理。

在最终合格清单中，物理结算买方通常会选择市值最低的债务交割，以最大化 $N(1-p_d)$。这就是 cheapest-to-deliver（CTD，最便宜可交割）选择权。拍卖制度也以复制物理结算结果为目标，Final Price 因而受最终清单中 CTD 价值影响。可交割范围越宽、最低合格债务价格越低，保护一般越有价值；重组到期限制若排除低价长债，则会缩小这项选择权。

## 一个区分“实际损失、可交割范围与合同赔付”的算例

某 CDS 名义本金为 100。信用事件后，保护买方原本持有的债券市价为 62；另有同一参考实体、同一合格优先级的债券市价为 38。

- 若两张债券都可交割，买方可以在市场买入 38 的债券并物理交割，净保护价值约为 $100-38=62$。它不受“原持债只损失 38”约束。
- 若拍卖 Final Price 为 40，auction-settled CDS 支付 $100\times(1-0.40)=60$；该数既不等于原持债损失 38，也不必等于最终破产回收。
- 若事件是适用 MM 条款的重组，且市价 38 的长债超过该合约的 maturity limitation、只有市价 62 的债券合格，则物理结算净保护价值降为 $100-62=38$。

三种结果的差异来自合同的可交割集合与结算价格，而不是买方陈述自己“实际亏了多少”。

> [!question]- 最小自检
> 参考实体保持正常，但保护卖方破产；这是否自动触发参考实体上的 CDS 信用事件和 $N(1-F)$ 赔付？
>
> **答案：** 不会。卖方破产是交易对手违约，按主协议、净额、抵押品和 close-out 处理；只有参考实体发生该交易覆盖且满足认定条件的信用事件，才进入 CDS 保护结算。

## 边界

- Covered Credit Event 是“被该交易选中且满足定义与程序条件的事件”，不是任何信用恶化、技术性延迟或口语中的 default。
- Reference Obligation、Obligation、Deliverable Obligation 与拍卖 Final List 是不同层次；一项债务能用于判断事件，不必因此也能交割。
- 保护买方一般无需先有可保利益或证明实际损失；但选择物理结算就必须按时取得并交付合格义务，债务稀缺可能提高交割成本。
- Auction Final Price 与固定回收条款下的 contract recovery 都是结算输入，不等于 [[违约损失口径|Basel 经济 LGD]]。后者还纳入实际净回收时间、处置成本与折现，服务不同目的。
- Definitions 版本、Confirmation、Matrix、地区/主体 Transaction Type、事件日期与 fallback 会改变结果。本卡给出标准结构，不替代对具体交易文件的法律审阅。

## 来源与核验

- ISDA, [*2014 ISDA Credit Derivatives Definitions*](https://www.isda.org/book/2014-isda-credit-derivative-definitions) 与 [2014 Definitions FAQ](https://www.isda.org/a/eXEDE/isda-2014-credit-definitions-faq-v12-clean.pdf)：定位信用事件框架、Governmental Intervention、Standard Reference Obligation 与重组 doc clause；具体交易仍以其 Confirmation 和纳入的定义为准。
- ISDA, [*Guidelines for Smart Contracts: Credit Derivatives*，Credit Events、Deliverable Obligations、Physical Settlement 与 Auction Settlement](https://www.isda.org/a/ur4TE/Guidelines-for-Smart-Contracts-CDS.pdf)：定位核验可交割条件、CTD、物理交割、Auction Final Price 及 $N\max(1-F,0)$。
- ISDA, [*Big Bang Protocol*](https://www.isda.org/traditional-protocol/big-bang-protocol/) 与 [*Small Bang Protocol*](https://www.isda.org/traditional-protocol/small-bang-protocol/)：分别定位拍卖 hardwiring/DC 权限，以及重组拍卖、MR/MM maturity limitation 与不同 maturity buckets。
- Federal Reserve Board, [*Credit Default Swaps*（FEDS 2022-023），§§4–5.1，第 8–10 页](https://www.federalreserve.gov/econres/feds/files/2022023pap.pdf)：定位物理/现金/拍卖结算、事件菜单，以及没有参考实体敞口也可买保护。
- Basel Committee on Banking Supervision, [CRE36.76：economic loss definition](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/36/inforce/2023-01-01/published/2022-12-08)：仅用于定位 Basel LGD 的经济损失、折现与追收成本口径，避免把它与 CDS 合同回收混同。
- 作者逐项核验日：2026-08-30；事件角色、结算公式、CTD 算例与重组边界经独立模型复核通过；具体交易仍须按适用文件接受法律审阅。
