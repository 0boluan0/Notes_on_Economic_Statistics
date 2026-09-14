---
aliases:
  - "CDS 的平价利差由保护腿的风险中性现值除以每单位利差的保费腿现值得到；固定票息合约还需用前端支付平衡两腿价值"
  - "CDS par spread and upfront pricing"
student_os: knowledge-atom
atom_id: RM-CDS-001
atom_set: cds-pricing-and-basis
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[CDS信用事件与结算]]"
  - "[[违约强度模型]]"
related:
  - "[[违约损失口径]]"
  - "[[DVA与双边估值]]"
part_of:
  - "[[CDS定价与基差.canvas|CDS定价与基差]]"
---

# CDS 的平价利差由保护腿的风险中性现值除以每单位利差的保费腿现值得到；固定票息合约还需用前端支付平衡两腿价值
<!-- bilingual-en:start -->
*A CDS par spread equals the risk-neutral present value of the protection leg divided by the premium-leg present value per unit spread; a fixed-coupon contract also needs an upfront payment to balance the two legs*
<!-- bilingual-en:end -->

> [!summary] 先算两条腿，再分清“平价利差”与“固定票息加 upfront”
> 保护买方在参考实体尚未发生合同覆盖的信用事件时支付定期保费，并在信用事件发生时结清从上次付息日至事件日的应计保费；保护卖方则按合同约定的结算损失赔付。平价利差是让两腿风险中性现值相等的年化费率。市场若把票息固定在一个标准水平，票息通常不再等于平价利差，价值差由一次性前端支付（upfront）补齐。

## 两条腿的风险中性现值

设名义本金为 $N$，到期日为 $T$，保费支付日为 $t_i$，该期年化计息因子为 $\alpha_i$。令 $\tau_C$ 为合同覆盖的信用事件时刻，$D(0,t)$ 为从 $t$ 贴现到估值日的贴现因子，$a(\tau_C)$ 为从上一个保费日至信用事件日的应计年化比例。每单位年化利差、每单位名义本金的风险年金（risky annuity，也称 RPV01）可写成

$$
\mathcal A_0
=\sum_i \alpha_i\,\mathbb E^{\mathbb Q}
\!\left[D(0,t_i)\mathbf 1_{\{\tau_C>t_i\}}\right]
+\mathbb E^{\mathbb Q}
\!\left[D(0,\tau_C)a(\tau_C)\mathbf 1_{\{0<\tau_C\le T\}}\right].
$$

第一项是仅在参考实体存续时支付的定期保费；第二项是信用事件日应计保费。若合同票息为 $c$，保费腿现值为

$$
PV_{\mathrm{prem}}=Nc\mathcal A_0.
$$

令 $\ell_C(\tau_C)$ 表示每单位名义本金按合同结算规则确定的损失率，例如拍卖结算下由 Auction Final Price 决定，而不是先验等同于某个会计或 Basel LGD。保护腿为

$$
\mathcal P_0
=\mathbb E^{\mathbb Q}
\!\left[D(0,\tau_C)\ell_C(\tau_C)
\mathbf 1_{\{0<\tau_C\le T\}}\right],
\qquad
PV_{\mathrm{prot}}=N\mathcal P_0.
$$

这里把付款日简写为信用事件日以突出两腿结构；实务模型还要按 [[CDS信用事件与结算|合同]]处理事件认定、拍卖或交割的实际付款日、保护起止日、营业日调整与应计细则。两条腿都在风险中性测度 $\mathbb Q$ 下估值并贴现；不能让保护腿使用风险中性概率、保费腿却使用未经转换的历史存活率。

## 平价利差与固定票息加 upfront

若新合约可以把持续费率直接设成使初始价值为零的水平，平价利差满足

$$
Ns_{\mathrm{par}}\mathcal A_0=N\mathcal P_0,
\qquad
\boxed{s_{\mathrm{par}}=\frac{\mathcal P_0}{\mathcal A_0}}.
$$

若合约票息被固定为 $c$，定义 $u_0$ 为每单位名义本金、估值日现值口径的 upfront，且 $u_0>0$ 表示保护买方向保护卖方支付，则公平交易要求

$$
Nu_0+Nc\mathcal A_0=N\mathcal P_0,
$$

所以

$$
\boxed{u_0=\mathcal P_0-c\mathcal A_0
=(s_{\mathrm{par}}-c)\mathcal A_0}.
$$

因此 $c<s_{\mathrm{par}}$ 时买方付 upfront；$c>s_{\mathrm{par}}$ 时 $u_0<0$，即卖方向买方支付。市场的 points upfront 通常按现金结算日和 clean/dirty 约定报价，应计保费可能另行结算；把市场报价代入上式前，必须先统一付款日、应计与正负号口径。

## 一个真正区分两种报价的算例

某五年 CDS 的风险年金为 $\mathcal A_0=4.20$，保护腿现值为每单位名义本金 $\mathcal P_0=0.084$。于是

$$
s_{\mathrm{par}}=0.084/4.20=0.02=200\text{ bp}.
$$

- 若可直接签一份 200 bp 的平价合约，初始 upfront 为零。
- 若标准票息固定为 100 bp，票息腿现值仅为 $0.01\times4.20=0.042$，买方需支付 $u_0=0.084-0.042=4.2\%$ 的估值日现值。
- 若标准票息固定为 500 bp，票息腿现值为 $0.05\times4.20=0.210$，则 $u_0=-12.6\%$，由卖方向买方支付。

本例故意忽略现金结算日前的贴现和另行应计，只用来区分年化 par spread、持续 fixed coupon 与一次性 upfront。把“100 bp + 4.2 points upfront”直接读成“CDS 利差是 4.2%”混合了两个不同量。

> [!question]- 最小自检
> 同一信用曲线下，固定票息从 100 bp 提高到 500 bp，保护腿不变。保护买方应支付的 upfront 会怎样变化？
>
> **答案：** 每单位利差的保费腿现值 $\mathcal A_0$ 不变，$u_0=(s_{\mathrm{par}}-c)\mathcal A_0$ 随 $c$ 提高而下降；它可能由买方支付的正数变成卖方支付的负数。平价利差本身不会因为换了标准票息而改变。

## 强度近似只给量级，不单独识别参数

在连续保费、近似常数的风险中性强度 $\lambda^{\mathbb Q}$ 与条件期望合同回收率、统一贴现体系，并忽略双边违约、流动性和融资等非信用楔子时，简化模型给出 $s=\lambda^{\mathbb Q}(1-R)$。此时两条腿含有同一个贴现—存活积分，贴现曲线本身无需平坦；离散付息、应计近似以及强度或回收率随期限和状态变化时，这一关系才退化为加权量级。一个利差只约束“风险中性到达率 × 合同损失率”的组合，不能分别识别 $\lambda^{\mathbb Q}$ 与 $R$，更不能直接识别真实世界 PD；完整条件与测度边界见 [[违约强度模型]]。

## 边界

- $\mathcal A_0$ 不是无风险年金：定期保费以存续为条件，还包含信用事件日应计；漏掉第二项会系统性改变 par spread 与 upfront。
- $\ell_C$ 是合同结算损失。Auction Final Price、固定回收条款、可交割义务和最便宜可交割选择都可能改变它；不能直接替换为 [[违约损失口径|Basel 经济 LGD]]。
- 若利率与信用事件相关或贴现因子随机，$\mathbb E^{\mathbb Q}[D\mathbf 1_{\{\tau>t\}}]$ 一般不能无条件拆成“独立的贴现因子 × 存活概率”。
- 上式是未加入双边违约、抵押品和 close-out 的基础定价恒等式；交易对手风险及 CVA/DVA 另见 [[DVA与双边估值]]。
- CDS 通常不要求保护买方在买入保护时实际持有参考实体债务，也不要求证明遭受了与赔付同额的实际损失。物理结算时买方必须在结算时交付合格义务，但可以在信用事件后购入；拍卖或现金结算则按合同价格公式赔付。

## 来源与核验

- Federal Reserve Board, [*Credit Default Swaps*（FEDS 2022-023），§6.4、§§7.1–7.2，第 17–21 页](https://www.federalreserve.gov/econres/feds/files/2022023pap.pdf)：定位核验风险中性存活概率、保费腿/保护腿、par spread，以及标准票息与 upfront 的方向。
- ISDA, [*ISDA Standard CDS Contract Converter Specification*，第 2–3 页](https://www.cdsmodel.com/assets/cds-model/docs/ISDA%20Standard%20CDS%20Contract%20Converter%20Specification%20-%20Sept%204,%202009.pdf)：定位核验标准合约按保护腿与保费腿现值计算 MTM、固定票息与 upfront/应计的报价口径及付款方向。
- ISDA, [*Guidelines for Smart Contracts: Credit Derivatives*，Credit Events 与 Settlement Methods](https://www.isda.org/a/ur4TE/Guidelines-for-Smart-Contracts-CDS.pdf)：定位核验赔付由合同覆盖的事件、可交割义务和物理/拍卖结算条款决定，而不是一般化的实际损失补偿。
- 作者逐项核验日：2026-08-30；两腿公式、正负号和算例已复算，独立模型复核通过。
