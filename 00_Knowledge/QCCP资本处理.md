---
aliases:
  - "QCCP 资格不会把银行对 CCP 的风险设为零；Basel 将交易敞口、抵押品与违约基金出资分别计量，清算成员自营 QCCP 交易敞口适用 2% 风险权重，而非 QCCP 不适用这一优惠口径"
  - "Basel capital treatment of QCCP exposures"
student_os: knowledge-atom
atom_id: RM-CCP-005
atom_set: otc-clearing-and-ccp-risk
atom_type: regulatory-treatment
status: source-checked
mastery_state: unassessed
requires:
  - "[[SA-CCR]]"
  - "[[中央清算与多边净额]]"
  - "[[CCP违约瀑布]]"
related:
  - "[[三类 CVA 口径]]"
  - "[[Basel 国际标准与本地法]]"
part_of:
  - "[[OTC清算与CCP风险.canvas|OTC清算与CCP风险]]"
---

# QCCP 资格不会把银行对 CCP 的风险设为零；Basel 将交易敞口、抵押品与违约基金出资分别计量，清算成员自营 QCCP 交易敞口适用 2% 风险权重，而非 QCCP 不适用这一优惠口径
*QCCP status changes capital treatment; it does not eliminate CCP risk*

> [!summary] 先分敞口，再匹配规则
> Basel CRE54 不把“对 CCP 的全部金额”乘同一个系数。银行先按对手方信用风险方法计算交易敞口 EAD，再处理提交抵押品是否 bankruptcy-remote，最后对违约基金出资使用独立公式。清算成员为自营目的产生的 QCCP 交易敞口适用 2% 风险权重；2% 既不是 EAD 系数，也不能套到违约基金或所有客户清算关系。

QCCP 是满足 Basel 定义、获得相应许可并被有关监管或监督机关允许对相关产品开展 CCP 业务的中央对手方。资格带来审慎优惠，不是“不会违约”的信用评级。银行仍须监控 CCP 交易敞口、抵押品、违约基金、未缴承诺和接管其他成员头寸等或有义务。

## 三类金额分别进入不同计算

| 银行对 CCP 的项目 | Basel 计算入口 | QCCP 基本处理 | 关键边界 |
|---|---|---|---|
| trade exposure（交易敞口） | 先按适用 CCR 方法得到 EAD，例如 [[SA-CCR]] 或获批 IMM | 清算成员自营 QCCP 交易敞口的风险权重为 2% | $2\%$ 乘 EAD 得到 RWA，不是把名义本金或全部 CCP 资产乘 2% |
| posted collateral（提交抵押品） | 看是否已包含在交易敞口，以及持有者、隔离和破产隔离安排 | 未 bankruptcy-remote 时按 CRE54.18–54.23 计量；由托管人持有且对 CCP bankruptcy-remote 时，该抵押品本身的 CCR 风险权重或 EAD 可为零 | 这里只免该抵押品的 CCP 对手信用资本，不代表没有托管、市场、流动性或操作风险 |
| default-fund contribution（违约基金出资） | 使用 $K_{CCP}$、成员 EAD、CCP 与成员预缴资源等形成的独立风险敏感公式 | 按 CRE54.24–54.39 单独计算，并有相应 floor 与信息要求 | 不能把违约基金出资当成普通交易敞口直接乘 2% |

因此，对清算成员自营 QCCP 交易敞口，最小结构是

$$
RWA_{\text{trade}}=2\%\times EAD_{\text{trade}}.
$$

这里的 $EAD_{\text{trade}}$ 仍由 CCR 框架形成，包含当前重置成本、潜在未来敞口、认可净额和抵押品等输入；“风险权重低”不等于“敞口为零”。最低资本要求还要按适用资本比率作用于 RWA，并受本地实施和其他缓冲约束。

## 客户清算不能自动套用 2%

- 清算成员对客户的敞口原则上按双边交易计量，包括适用的 CVA 风险；中央清算并未把客户本身变成 QCCP。
- 银行作为清算客户时，只有满足 CRE54.14–54.15 的保护与可移转条件，才可对相应清算成员或 CCP 敞口采用类似 QCCP 交易敞口处理。条件包括客户交易被识别、抵押品得到保护，以及清算成员违约时头寸和抵押品高度可能按市场价值继续或移转。
- 若不能防止清算成员与其另一客户联合违约造成损失，但其他条件满足，CRE54.16 规定 4% 风险权重；若相关条件不满足，CRE54.17 要求按对清算成员的双边交易计量。
- 清算成员只有在合同上须补偿客户因 QCCP 违约导致的交易价值损失时，相关 CCP leg 才按 CRE54.7 的边界进入其 2% 处理。不能只因交易“经过 CCP”就机械加上或免除一条敞口。

## 非 QCCP 不享受同一优惠

对 non-qualifying CCP：

- 交易敞口按该对手类别适用信用风险标准法，而不是 2% QCCP 风险权重；
- 违约基金出资，包括受约束可能被要求缴付的 funded 与 unfunded contribution，通常适用 1250% 风险权重；无限承诺计入多少由国家监管者在监督审查中确定。

这一区分说明“CCP”只是基础设施类型，“QCCP”才是 CRE54 优惠处理的资格条件。即使是 QCCP，违约基金公式也会把 CCP 总体成员敞口、CCP 自有预缴资源和成员预缴资源纳入，而非假设共同损失为零。

## 可复算例：2% 只作用于交易 EAD

某银行作为清算成员，为自营交易面对 QCCP：

- 按 SA-CCR 得到交易敞口 $EAD_{\text{trade}}=100$；
- 另有由托管人持有且对 CCP bankruptcy-remote 的合格抵押品 30；
- 预缴违约基金出资 10。

交易敞口的 RWA 为

$$
0.02\times 100=2.
$$

在 CRE54.21 条件真正满足时，抵押品 30 本身不再产生 CCP 对手信用资本；但违约基金 10 仍必须进入 CRE54.24–54.39 的独立公式，不能算成 $0.02\times10=0.2$。

若同样金额面对非 QCCP，交易敞口须按该对手类别的信用风险标准法计量，违约基金 10 的 RWA 则为

$$
12.50\times 10=125,
$$

而不是 0.2。该算例只展示分类顺序；没有给出 QCCP 违约基金公式所需的全体成员与 CCP 资源数据，因此不能虚构其最终资本数值。

## 边界

- 2% 是清算成员自营 QCCP **交易敞口的风险权重**，不是名义本金折扣、EAD 乘数、违约概率或全部 CCP 敞口的统一权重。
- bankruptcy-remote collateral 的零处理只针对该抵押品的 CCP 对手信用风险，并要求托管和法律隔离条件成立；不得外推为该资产“无任何风险”。
- 违约基金承担共同损失，必须单独计量；QCCP 资格不把 funded、unfunded 或 assessment 风险消除。
- 客户、清算成员、高层客户和 CCP 之间每一条法律关系要分别识别。只有满足客户保护、可执行法律审查和高度可能 porting 等条件时，才适用相应 2% 或 4% 处理。
- Basel 是国际最低标准；QCCP 认定、报告日期和本地资本规则仍须按 [[Basel 国际标准与本地法|当地实施]] 核对。

> [!question]- 最小自检
> 银行对 QCCP 有交易 EAD 80、未破产隔离的提交抵押品 20 和违约基金出资 10。能否把三项相加后统一计算 $2\%\times(80+20+10)$？
>
> **答案：** 不能。交易 EAD 先按 CRE54.7–54.8 适用 2% 风险权重；抵押品按 CRE54.18–54.23 的持有和破产隔离条件处理；违约基金按 CRE54.24–54.39 的独立公式处理。三类金额不能共用一个 2%。

## 一手来源与复核状态

- Basel Committee, [Basel Framework CRE54, current chapter](https://www.bis.org/basel_framework/chapter/CRE/54.htm)：CRE54.6–54.8 定位 QCCP 范围、清算成员自营交易敞口的 2% 风险权重与 CCR EAD 输入；CRE54.12–54.17 定位客户清算边界。
- Basel Committee, [Basel Framework CRE54, collateral and default-fund treatment](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/54/inforce/2023-01-01/published/2020-03-27)：CRE54.18–54.23 定位提交抵押品和 bankruptcy-remote 条件；CRE54.24–54.40 定位 QCCP 违约基金独立公式与 cap。
- Basel Committee, [Basel Framework CRE54.41–54.42](https://www.bis.org/basel_framework/chapter/CRE/54.htm?inforce=20230101&published=20200327)：定位非 QCCP 交易敞口的标准法处理和违约基金出资的 1250% 风险权重。

> [!warning] 复核状态
> 本轮已通过独立的来源、定义、公式/算例与边界复核，因此状态为 `source-checked`；用户掌握度仍为 `unassessed`。用于实际机构资本计量时，仍须按当前报告日确认 QCCP 资格、本地实施、适用 CCR 方法、客户保护与 porting 条件、抵押品的 bankruptcy-remoteness，以及 QCCP 违约基金公式的全量输入；不得把 2% 作为通用 CCP 系数，也不得把 `source-checked` 当成监管或模型批准。
