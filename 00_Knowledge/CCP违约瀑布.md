---
aliases:
  - "CCP 成员违约后的市场损失首先消耗违约成员自身资源，随后才依该 CCP 的规则触及指定的 CCP 自有资金和非违约成员共同资源；不存在适用于所有 CCP 的固定违约瀑布顺序"
  - "CCP default waterfall"
student_os: knowledge-atom
atom_id: RM-CCP-002
atom_set: otc-clearing-and-ccp-risk
atom_type: loss-allocation
status: source-checked
mastery_state: unassessed
requires:
  - "[[中央清算与多边净额]]"
  - "[[净额与抵押品]]"
related:
  - "[[保证金顺周期]]"
leads_to:
  - "[[CCP恢复与处置]]"
  - "[[QCCP资本处理]]"
part_of:
  - "[[OTC清算与CCP风险.canvas|OTC清算与CCP风险]]"
---

# CCP 成员违约后的市场损失首先消耗违约成员自身资源，随后才依该 CCP 的规则触及指定的 CCP 自有资金和非违约成员共同资源；不存在适用于所有 CCP 的固定违约瀑布顺序
<!-- bilingual-en:start -->
*CCP default waterfall*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 清算成员违约后，CCP 一方面要控制并转移违约组合，恢复买卖义务相配的账簿；另一方面要按规则分配平仓、对冲、拍卖和未履约造成的损失。违约瀑布只描述后一个问题。通常先使用该违约成员在相关清算服务、账户和法律安排下可动用的 IM 与违约基金出资，剩余损失才依 CCP 规则触及指定的 CCP 自有资金、非违约成员共同违约基金及后续承诺资源。CCP 自有资金位于共同资源之前、同时还是之后，以及恢复工具何时启动，都必须查实际 rulebook。

## 先确定要分配的损失
<!-- bilingual-en:start -->
*First determine the loss to allocate*
<!-- bilingual-en:end -->

CCP 宣布成员违约时，最后一次已经结算的 variation margin（VM，变动保证金）通常已经把此前的市值变化转移出去。之后的经济损失可概括为：未支付的到期义务，加上从违约到完成对冲、移转、拍卖或平仓期间的市场变化与执行成本，再减去可依法实现的违约成员资源。

VM 不应被随手列成一笔预先放在瀑布里的违约缓冲：

- 已结算 VM 已经重置或抵销了此前的当前敞口，再把它加进违约资源会重复计算。
- 到期但未支付的 VM 是违约时未履行义务的一部分，不是 CCP 已持有的保护。
- 极端情况下扣减非违约成员的 variation margin gains，属于规则预先约定的恢复型损失分配工具；它不等于违约成员事先提供的 IM 或违约基金出资。

因此，计算瀑布前要先固定估值时点、VM 是否已结算、close-out 价格、执行成本与抵押品可用性，不能把“保证金”统称为一个数字。

## 各层资源的经济角色不同
<!-- bilingual-en:start -->
*Each layer has a different role*
<!-- bilingual-en:end -->

| 资源 | 谁提供 | 主要角色 | 不能混淆之处 |
|---|---|---|---|
| 违约成员 IM | 违约成员 | 覆盖自最后一次 VM 到完成处置期间的潜在市场损失 | 只按相关账户、客户隔离、法律与 CCP 规则中可动用的部分计算 |
| 违约成员违约基金出资 | 违约成员 | 在其 IM 不足时继续吸收该成员造成的压力损失 | 与普通 IM、非违约成员共同基金不是同一资源 |
| 指定 CCP 自有资金 | CCP | 让 CCP 自身承担预先约定的一部分损失并对风险管理形成激励，常称 skin in the game | 金额、分层和位次依规则而异；不等于 CCP 全部权益资本 |
| 非违约成员共同违约基金 | 其他成员 | 共同分担穿透违约成员自身资源后的损失 | 属于互助化资源；通常按具体清算服务隔离，不能任意跨服务调用 |
| 追加出资或其他恢复资源 | CCP、母公司或非违约成员，依规则而定 | 在预筹资源不足时提供额外承诺资源或分配剩余损失 | 是否属于普通瀑布、恢复阶段或处置阶段要按规则和法律区分 |

PFMI 只给出可能组成瀑布的资源类别和风险覆盖原则，并没有规定全球统一的完整层序。尤其，CCP 自有资金可以在非违约成员出资之前、同时或之后分层使用；某一交易所或 CCP 的图示不能冒充所有 CCP 的规则。

## 违约管理流程不等于损失瀑布
<!-- bilingual-en:start -->
*Default management is not the waterfall*
<!-- bilingual-en:end -->

成员违约会同时触发操作流程与财务流程：

1. **控制账户与客户保护。** 冻结或限制违约成员活动，确认客户隔离，并在条件允许时把客户头寸与相应抵押品移转给另一清算成员（porting）。
2. **降低市场风险。** CCP 可先对违约组合做临时 hedge，避免暴露继续扩大；是否对冲、何时对冲和用什么工具取决于组合与市场流动性。
3. **转移或终止头寸。** 通过拍卖、市场出售、买入补回或其他规则，把违约成员头寸转给非违约参与者，目标是恢复 CCP 的 matched book。
4. **确认并分配损失。** 处置价格和成本确定后，按适用瀑布消耗资源。

前 3 步回答“怎样重新形成配平账簿并继续关键服务”，第 4 步回答“损失由谁承担”。拍卖本身不是一层资金；porting 也不是把损失转给客户。操作失败会扩大最终损失，却不能用一个更长的资金列表替代可执行的违约管理方案。

## 可复算示意：顺序必须作为规则假设写出
<!-- bilingual-en:start -->
*Reproducible illustration: state the rulebook order explicitly*
<!-- bilingual-en:end -->

假设某 CCP 的一个清算服务在完成对冲和拍卖后，确认需分配的损失为 26。最后一次已支付 VM 已在组合价值中反映，不再列作资源。该服务的规则**明确规定**：

1. 违约成员 IM 12；
2. 违约成员违约基金出资 4；
3. 第一层指定 CCP 自有资金 3；
4. 非违约成员共同违约基金可用金额 7，承担其后损失。

逐层计算：

$$
26-12=14,
$$

$$
14-4=10,
$$

$$
10-3=7.
$$

于是共同违约基金承担 7，预筹资源在这个算例中恰好足够。若此前 CCP 已收取并支付 VM 9，不能再写成 $26-9$，因为 26 已是在该 VM 结算之后确认的损失。若有一笔 VM 5 到期未付，它应进入违约义务和损失基数，而不是当作“可用资源 5”。

这个算例只验证给定规则下的算术。另一 CCP 可能把自有资金拆成两个 tranche，或让某层与成员基金并列；除“先动用依法可用的违约成员自身资源”这一基本方向外，后续次序必须重新读取规则。

## 从瀑布进入恢复与处置的边界
<!-- bilingual-en:start -->
*Boundary to recovery and resolution*
<!-- bilingual-en:end -->

正常违约管理的预筹资源或流动性安排已经耗尽、或已判断可能不足，或者 CCP 无法及时恢复 matched book、补充资源并持续关键服务时，就可能启动更强的恢复安排。恢复工具可能涉及追加出资、损失分摊、variation margin gains haircutting、强制分配或合约终止；其使用条件、上限和顺序应事先写入规则。若恢复不可行、不够及时、预期不足或会危及金融稳定，法定处置当局可以在恢复工具尚未全部实施或耗尽前按适用法律介入。

因此，“违约瀑布耗尽”不等于 CCP 自动破产，也不等于可以任意改变合同。正常瀑布、恢复与法定处置的触发点和权力来源分别见 [[CCP恢复与处置]]。

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 只能动用实际规则、账户隔离与法律安排允许的违约成员资源；客户资产、隔离 IM 或另一清算服务的资源不当然可用。
- IM、违约基金、CCP 自有资金和非违约成员资源承担不同风险层；把它们都叫“保证金”会丢失谁出资、何时使用和是否共同化。
- VM 是当前市值结算机制；未付 VM 会形成违约义务，已付 VM 已影响损失基数，VM gains haircutting 则是另一阶段的恢复工具。
- Hedging、auction、close-out 与 porting 是恢复 matched book 或保护客户的操作，不是默认资金层。
- 资本监管是否把某 CCP 认定为 QCCP、成员交易敞口和违约基金出资如何计提，另见 [[QCCP资本处理]]；不能从瀑布位次直接推导资本风险权重。

> [!question]- 最小自检
> 某笔已支付 VM 10 已把截至昨日的市值变化结算完；成员今日违约后，CCP 从昨日结算点起的新增市场损失与处置成本合计 18。能否把可用资源写成“VM 10 + IM 8”，从而断言损失已全部覆盖？
>
> **答案：** 不能。已支付 VM 10 已经进入昨日的结算状态，不能再次当作预筹违约资源。应从今日确认的 18 开始，按账户和法律可用性先使用 IM 8，剩余 10 再依该 CCP 的违约基金、指定自有资金和共同资源规则分配。

## 定位性来源与复核状态
<!-- bilingual-en:start -->
*Primary-source anchors and review status*
<!-- bilingual-en:end -->

- CPMI–IOSCO, [*Principles for financial market infrastructures*](https://www.bis.org/cpmi/publ/d101a.pdf)，Principle 4、para 3.4.17 与 Principle 13：定位 CCP 信用资源、可能的瀑布组成，以及参与者违约规则与处置程序。
- CPMI–IOSCO, [*Resilience of central counterparties: Further guidance on the PFMI*](https://www.bis.org/cpmi/publ/d163.htm)：定位保证金、压力测试、资源覆盖和 CCP 自有资金贡献的治理边界。
- CPMI–IOSCO, [*Recovery of financial market infrastructures*](https://www.bis.org/cpmi/publ/d162.htm)：定位预筹资源不足后的恢复工具、补充资源、损失分配与恢复 matched book。
- CPMI–IOSCO, [*Central counterparty default management auctions: Issues for consideration*](https://www.bis.org/cpmi/publ/d192.pdf)：定位违约组合对冲、拍卖与客户 porting 的操作关系；这些流程不能与资金瀑布混写。

> [!warning] 复核状态
> 本轮已通过独立的来源、定义、公式/算例与边界复核，因此状态为 `source-checked`；用户掌握度仍为 `unassessed`。用于实际机构决策时，仍应以至少一个实际 CCP rulebook 核对该服务的账户隔离、资源位次、CCP 自有资金 tranche、拍卖和恢复触发点，并按实际数据重算瀑布；不得把本卡的示例顺序当成普遍规则，也不得把 `source-checked` 当成模型批准或法律意见。
