---
aliases:
  - "损失分位点、分位点减 EL 的中心化值和损失标准差是三个不同量；使用 Credit VaR、UL 或经济资本名称时必须显式声明所指口径"
  - "Loss quantile, centered Credit VaR, and loss volatility"
student_os: knowledge-atom
atom_id: RM-CP-004
atom_set: credit-portfolio-risk-and-credit-var
atom_type: measurement-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[风险度量口径]]"
related:
  - "[[组合损失聚合]]"
  - "[[相同EL不同尾部]]"
  - "[[VaR非次可加]]"
part_of:
  - "[[信用组合风险与 Credit VaR.canvas|信用组合风险与 Credit VaR]]"
---

# 损失分位点、分位点减 EL 的中心化值和损失标准差是三个不同量；使用 Credit VaR、UL 或经济资本名称时必须显式声明所指口径
<!-- bilingual-en:start -->
*A loss quantile, a mean-centered quantile, and loss standard deviation are distinct; Credit VaR, UL, and economic capital require explicit conventions*
<!-- bilingual-en:end -->

> [!summary] 名称不能替代公式
> 本库把 VaR 固定为损失左分位 $q_\alpha(L)$。有些信用风险资料把 $q_\alpha(L)-E[L]$ 称为 Credit VaR、UL 或资本代理量，另一些资料用 UL 指损失标准差，监管框架又有自己的 UL 资本计算。三者不是同一个统计量；任何名称都必须随公式、期限、置信水平和损失口径一起给出。

## 三个量分别回答什么

设组合损失 $L$ 可积，其均值为

$$
\mu_L=E[L].
$$

本库的 VaR 沿用 [[VaR定义]]：

$$
\operatorname{VaR}_\alpha(L)=q_\alpha(L).
$$

它回答“损失分布在置信水平 $\alpha$ 的左分位点是多少”。若另定义中心化分位风险

$$
C_\alpha(L)=q_\alpha(L)-E[L],
$$

它回答“该分位点高于平均损失多少”。只有在文件明确如此定义时，才可以把 $C_\alpha$ 称作 mean-centered Credit VaR（均值中心化的信用 VaR）或某种经济资本代理量。

若进一步有 $E[L^2]<\infty$，损失标准差则是

$$
\sigma_L
=\sqrt{E[(L-\mu_L)^2]}.
$$

它衡量围绕均值的二阶离散程度，对上行和下行偏离对称计权；在偏斜、离散的信用损失分布中，不能仅凭 $\sigma_L$ 恢复某个尾部分位点。

## 一个同时得到三个不同数字的分布

设

$$
P(L=0)=0.98,
\qquad P(L=10)=0.01,
\qquad P(L=100)=0.01.
$$

则

$$
E[L]=0.01\times10+0.01\times100=1.1.
$$

因为 $F_L(0)=0.98$、$F_L(10)=0.99$，

$$
q_{0.99}(L)=10,
\qquad C_{0.99}(L)=10-1.1=8.9.
$$

同时

$$
\sigma_L
=\sqrt{E[L^2]-E[L]^2}
=\sqrt{101-1.21}
=\sqrt{99.79}
\approx9.99.
$$

同一分布、同一期限得到 10、8.9 和约 9.99 三个不同数字。把其中任意一个不加说明地写成“信用 VaR”都会丢失可复核的定义。

## UL 与资本为什么还要单独声明

- **描述性 UL。** 有些教材用 UL 表示损失标准差 $\sigma_L$；这是围绕均值的波动量。
- **分位中心化 UL。** 有些内部经济资本框架用 $q_\alpha(L)-EL$ 表示目标置信水平下由资本覆盖的意外部分。
- **Basel IRB UL。** CRE30–31 的风险权重函数产生监管 UL 资本要求，EL 在 CRE35 中另行处理。Basel 解释文件用特定 ASRF 模型、监管置信水平、相关函数、期限调整和 downturn LGD 构造“VaR 与 EL 的距离”；它不是把任意内部损失分布的标准差换一个名称。
- **经济资本与监管资本。** 经济资本是机构为内部偿付目标选择的资本量，可能以中心化分位、ES、压力情景和附加项为依据；监管资本是规则体系的计算结果。两者可能校准到相似目标，却不因名称相近而相等。

因此，合格的表述应像“在一年实际概率损失分布下，99.9% 左分位减统计 EL”或“Basel CRE31 风险权重函数产生的 UL 资本要求”，而不是只写“UL 为 12”。

> [!question]- 最小自检
> 某报告只写“Credit VaR = 8.9”。至少还缺哪些信息？
>
> **答案：** 至少缺损失变量与正负号、风险期限、概率口径、置信水平、分位定义，以及 8.9 究竟是 $q_\alpha(L)$、$q_\alpha(L)-EL$、$\sigma_L$，还是某个监管或内部资本结果。

## 边界

- 只有在额外分布假设下，分位点才可由均值和标准差映射；信用损失通常离散且偏斜，不能默认正态关系 $q_\alpha=\mu+z_\alpha\sigma$。
- 从分位点减 EL 前，要确认 EL 与分位损失来自同一组合、期限、估值基准和概率测度；否则“中心化”只是两个不可比数字相减。
- 若模型已经输出 $q_\alpha-EL$，资本计算中再次扣除 EL 会重复中心化。
- 资本还可能受模型风险、集中附加、压力测试、监管底线和可用资本定义影响；统计量不自动等于可持有的资本工具金额。
- VaR 是否满足次可加是另一问题，见 [[VaR非次可加|VaR 的次可加边界]]；不能用“中心化”消除 VaR 的离散和聚合边界。

## 来源与核验

- RiskMetrics Group, [*CreditMetrics Technical Document*，第 1.4 节，第 15–17 页](https://www.msci.com/documents/10199/93396227-d449-4229-9143-24a94dab122f)：定位核验标准差与分位水平是两个不同信用风险量，且偏斜信用分布不能仅由标准差恢复分位点。
- Basel Committee on Banking Supervision, [*An Explanatory Note on the Basel II IRB Risk Weight Functions*，第 2、4.4 与 5.1 节，第 2–12 页](https://www.bis.org/bcbs/irbriskweight.pdf)：定位核验该监管模型下 EL、VaR、UL 距离、99.9% 置信水平和资本要求之间的特定关系。
- Basel Committee on Banking Supervision, [CRE30.2：IRB 的 UL/EL 分工](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/30/inforce/2023-01-01/published/2020-03-27)、[CRE31.1：UL risk-weight functions](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/31/inforce/2023-01-01/published/2020-03-27) 与 [CRE35.1–35.3：EL 的单独处理](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/35/inforce/2023-01-01/published/2020-03-27)：定位区分监管 UL 资本要求与 EL amount。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
