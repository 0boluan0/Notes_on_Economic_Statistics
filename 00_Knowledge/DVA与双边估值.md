---
aliases:
  - "DVA 是本方自身不履约风险对负敞口的公允价值调整；双边估值虽常写成 clean value 减 CVA 加 DVA，却必须按同一 close-out 与 first-to-default 规则联合计算"
  - "DVA and first-to-default bilateral valuation"
student_os: knowledge-atom
atom_id: RM-CCR-004
atom_set: counterparty-credit-risk-and-valuation
atom_type: valuation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[对手方敞口指标]]"
  - "[[单边CVA]]"
related:
  - "[[净额与抵押品]]"
  - "[[三类 CVA 口径]]"
part_of:
  - "[[对手方信用风险与估值调整.canvas|对手方信用风险与估值调整]]"
---

# DVA 是本方自身不履约风险对负敞口的公允价值调整；双边估值虽常写成 clean value 减 CVA 加 DVA，却必须按同一 close-out 与 first-to-default 规则联合计算
<!-- bilingual-en:start -->
*DVA is the fair-value adjustment for the valuing party's own non-performance risk on negative exposure; although bilateral value is often written as clean value minus CVA plus DVA, both terms must be computed under one close-out and first-to-default rule*
<!-- bilingual-en:end -->

> [!summary] “减 CVA、加 DVA”先固定是谁的价值，再固定谁先违约
> 从银行价值视角，未来正敞口是对手欠银行的钱，对手先违约会造成 CVA；未来负敞口是银行欠对手的钱，银行先违约时未能全额履约形成 DVA。双边估值不能把两份各自假设另一方永不违约的单边调整直接拼接，因为第一方违约并 close-out 后，原交易不再等待第二次违约。只有 CVA 与 DVA 使用同一净额集合、抵押品、close-out 和 first-to-default 事件时，$V^{bil}=V^0-CVA+DVA$ 才是内部一致的写法。

## 正负敞口必须从同一个价值视角定义

令 $V_0^0$ 表示估值日、从银行 $B$ 视角观察且双方均被视为履约时的 clean value。另令 $X_t$ 表示若在未来时点 $t$ close-out，从银行视角按同一依法可执行净额集合、抵押品与 close-out 规则算出的**有符号未覆盖金额**。定义

$$
X_t^+=\max(X_t,0),
\qquad
X_t^-=\max(-X_t,0).
$$

- $X_t^+>0$：close-out 后对手方 $C$ 仍欠银行；若 $C$ 违约，银行面对损失。
- $X_t^->0$：close-out 后银行仍欠对手方；若银行违约，对手面对损失，而该未履约部分从银行价值视角体现为 DVA。

因此，DVA 不是把 CVA 公式原封不动换成本方违约概率后仍乘正敞口。它必须落在**银行负敞口**上。即使今天 $V_0^0>0$，未来路径仍可能出现 $X_t<0$，所以一项当前资产也可能同时具有未来 CVA 与 DVA。$V_0^0$ 决定估值起点；$X_t$ 才承载违约时的净额、抵押品与 close-out 口径，两者不能混成同一个符号。

## 双边公式使用 first-to-default 事件

令 $\tau_C$ 为对手方违约时刻，$\tau_B$ 为银行违约时刻。在 risk-free close-out、确定损失率、同一净额与抵押品口径，并暂时假定同时违约概率为零时，双边调整可写为

$$
CVA_0^{FTD}
=\mathbb E^{\mathbb Q}\!\left[
D(0,\tau_C)LGD_C X_{\tau_C}^{+}
\mathbf 1_{\{\tau_C<\tau_B,\,\tau_C\le T\}}
\right],
$$

$$
DVA_0^{FTD}
=\mathbb E^{\mathbb Q}\!\left[
D(0,\tau_B)LGD_B X_{\tau_B}^{-}
\mathbf 1_{\{\tau_B<\tau_C,\,\tau_B\le T\}}
\right].
$$

在这些条件下，银行的双边价值为

$$
\boxed{
V_0^{bil}=V_0^0-CVA_0^{FTD}+DVA_0^{FTD}}.
$$

两个指标函数互斥：若对手先违约，交易按该时点 close-out，不再在随后银行违约时对同一交易另计 DVA；银行先违约同理。若模型允许共同违约具有正概率，必须再加入合同指定的共同违约顺位、回收和结算规则，不能任意把等号情形塞入任一方。

## close-out 不是可以事后补上的脚注

上式中的 $X_t$ 若以双方均履约的 replacement value 为起点，就是 risk-free close-out 口径，但仍须随后落实净额、抵押品返还和保证金风险期。若合同或模型采用 replacement close-out，违约时金额还可能包含存续方自身信用、融资与实际替换成本；此时必须按该规则重新定义 $X_t$，CVA、DVA 也会随之改变。

同一双边模型至少要统一：

- 哪些交易属于同一依法可执行的净额集合；
- variation margin、initial margin、haircut、隔离与再使用怎样进入 close-out；
- margin period of risk 内的价值变化和抵押品返还请求怎样处理；
- 两方违约依赖、first-to-default 和共同违约怎样分配；
- clean value、CVA 与 DVA 是否使用一致的贴现、币种和估值时点。

分别运行“银行永不违约的 UCVA”和“对手永不违约的 UDVA”再相减，会把一些已经由第一场违约终止的路径计入两次。它可以作为特定条件下的工程近似，但不能冒充一般双边恒等式。

## 一个可复算的 first-to-default 算例

设 clean value 为 $V_0^0=1.20$，单期贴现因子为 0.98。已知以下概率是**先违约事件概率**，而不是忽略另一方后的单独累计 PD：

- 对手先违约的概率为 2%，该事件上的银行未覆盖正敞口 $X^+$ 为 8，$LGD_C=60\%$；
- 银行先违约的概率为 1.5%，该事件上的银行未覆盖负敞口 $X^-$ 为 5，$LGD_B=50\%$。

于是

$$
CVA^{FTD}=0.98\times0.02\times8\times0.60=0.09408,
$$

$$
DVA^{FTD}=0.98\times0.015\times5\times0.50=0.03675,
$$

$$
V_0^{bil}=1.20-0.09408+0.03675=1.14267.
$$

这个例子同时说明：今天的 clean value 为正，不代表所有未来 close-out 路径的 $X_t$ 都为正；银行先违约时只有未覆盖负敞口路径进入 DVA。若题目给的是两方各自的 standalone PD，而不是 first-to-default 概率，就还需指定两方违约依赖，不能直接代入本例。

> [!question]- 最小自检
> 某头寸在所有未来状态下对银行都满足 $X_t\ge0$。银行自身信用利差上升，按本卡公式该头寸的 DVA 是否自动增加？
>
> **答案：** 不会。因为 $X_t^-=0$，银行没有该头寸上的未覆盖负敞口可因自身不履约而减免。自身违约概率上升只有与负敞口相遇才产生该项 DVA。

## 会计 own-credit 与 Basel 资本不是同一层

IFRS 13 要求负债公允价值反映 non-performance risk，并明确考虑发行人的自身信用。因此，银行信用恶化可能降低其衍生品负债的公允价值，从银行视角表现为 DVA 增加或 own-credit gain。这个会计结果不是收到现金，也不表示银行经济状况改善；若信用随后改善，估值收益还可能转回。

Basel 监管 CVA 则明确排除银行自身违约；它不是“会计 CVA 减去 DVA”。资本层面，Basel CAP30.15 要求在 CET1 中剔除衍生品负债因自身信用产生的会计估值调整，且不允许用对手方信用调整抵销。会计公允价值、监管 CVA、CVA 风险资本和 CCR 违约资本的分工见 [[三类 CVA 口径]]。

## 边界

- $V^0-CVA+DVA$ 是在符号、close-out 与 first-to-default 统一后的条件式；加入融资、税、资本或非线性抵押反馈后，全部 XVA 不一定仍可线性相加。
- DVA 表示负债不完全履约的公允价值影响，不是可无摩擦兑现的交易利润，也不是对银行自身违约损失的保险。
- 对手方与银行的违约相关、[[错向风险|市场—信用依赖]]及 close-out contagion 都可改变两个 first-to-default 期望；只给两个边际 PD 不能唯一确定双边价值。
- recovery 应作用于相应 close-out 债权：债权人实际收回 $R\times exposure$，未收回的 $LGD\times exposure$ 才是 CVA 损失或 DVA 减免。不能把 LGD 写成违约方实际支付比例。
- 本卡给出估值结构，不替代具体 ISDA 主协议、CSA、适用破产法或会计政策的法律与会计判断。

## 来源与核验

- Brigo, Buescu & Morini, [*Counterparty Risk Pricing: Impact of Closeout and First-to-Default Times*](https://arxiv.org/abs/1106.3496)：一手论文，定位两个 unilateral adjustments 简单相减会忽略 first-to-default、重复计量路径，且 close-out 与违约依赖会改变双边价值。
- Brigo & Morini, [*Dangers of Bilateral Counterparty Risk: The Fundamental Impact of Closeout Conventions*](https://arxiv.org/abs/1011.3355)：定位 risk-free 与 replacement close-out 的选择会改变双边调整，close-out 不能与违约模型分开处理。
- IFRS Foundation, [IFRS 13，paragraphs 42–44](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2022/issued/part-a/ifrs-13-fair-value-measurement.pdf?bypass=on)：定位负债公允价值包含 non-performance risk 与自身信用风险。
- Basel Committee on Banking Supervision, [MAR50.3、50.32](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/50/inforce/2023-01-01/published/2020-07-08) 与 [CAP30.15](https://www.bis.org/committees/bcbs/basel-framework/standard/cap/30/inforce/2019-12-15/published/2019-12-15)：分别定位监管 CVA 排除银行自身违约，以及 CET1 对 derivative own-credit valuation adjustments 的剔除和禁止抵销。
- 本轮已通过独立的来源、定义、公式/算例与边界复核，因此状态为 `source-checked`；用户掌握度仍为 `unassessed`。用于实际机构估值时，仍须按具体 ISDA/CSA、close-out 约定、两方违约依赖、适用法律与会计政策重新核验；`source-checked` 不代表模型批准或法律意见。
