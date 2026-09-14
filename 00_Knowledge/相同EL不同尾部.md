---
aliases:
  - "相同预期损失的信用组合仍可因敞口集中和违约依赖不同而具有不同的高分位与尾部损失"
  - "Equal expected loss can hide different credit tails"
student_os: knowledge-atom
atom_id: RM-CP-003
atom_set: credit-portfolio-risk-and-credit-var
atom_type: counterexample
status: source-checked
mastery_state: unassessed
requires:
  - "[[组合损失聚合]]"
  - "[[共同违约概率]]"
  - "[[VaR定义]]"
related:
  - "[[VaR非次可加]]"
leads_to:
  - "[[EL与信用VaR]]"
part_of:
  - "[[信用组合风险与 Credit VaR.canvas|信用组合风险与 Credit VaR]]"
---

# 相同预期损失的信用组合仍可因敞口集中和违约依赖不同而具有不同的高分位与尾部损失
<!-- bilingual-en:start -->
*Credit portfolios with equal expected loss can have different quantiles and tails because of exposure concentration and default dependence*
<!-- bilingual-en:end -->

> [!summary] 均值只固定分布的重心，不固定坏状态怎样出现
> EL 相同，只表示损失的概率加权平均相同。大额单名敞口会把损失集中在少数跳跃状态，共同系统因子会让多个违约同时发生；两者都能在不改变边际 EL 的情况下改变损失分位点和更深尾部。

## 用同一左分位定义比较

本库沿用 [[VaR定义]] 的损失左分位

$$
q_\alpha(L)=\inf\{\ell:F_L(\ell)\ge\alpha\}.
$$

比较两个组合，期限、损失单位和 LGD 均相同：

- **组合 A：** 100 笔各 1 单位的贷款，每笔 PD 为 1%、LGD 为 100%，并为本算例假设相互独立。因此 $L_A\sim\operatorname{Binomial}(100,0.01)$。
- **组合 B：** 一笔 100 单位的贷款，PD 为 1%、LGD 为 100%。因此 $L_B=0$ 的概率为 99%，$L_B=100$ 的概率为 1%。

两者的预期损失都为

$$
E[L_A]=100\times0.01\times1=1,
\qquad
E[L_B]=0.01\times100=1.
$$

但在 99% 左分位，组合 A 满足

$$
P(L_A\le3)=0.98162596<0.99,
\qquad
P(L_A\le4)=0.99656768\ge0.99,
$$

所以

$$
q_{0.99}(L_A)=4.
$$

组合 B 在 0 处的累计概率恰为 99%，故

$$
q_{0.99}(L_B)=0.
$$

这个结果不是说集中组合更安全，而是暴露了离散分位点的断点：按左分位定义，累计概率一旦在 0 处达到置信水平，分位点就停在 0。若把置信水平提高到 99.5%，则

$$
q_{0.995}(L_A)=4,
\qquad
q_{0.995}(L_B)=100,
$$

因为 $P(L_B\le0)=0.99<0.995$，下一处质量点直接跳到 100。此时集中组合的分位损失显著更高。置信水平从 99% 稍微跨过 99% 就触发从 0 到 100 的跳跃，正是离散信用损失中必须说明分位约定和断点的原因。

依赖也能造成同类变化。若组合 A 的 100 个债务人不再独立，而是使用同一个 1% 违约指示变量，则 EL 仍为 1，损失却同样变成 0 或 100。边际 PD 和 EL 没变，尾部被共同违约重新塑形。

> [!question]- 最小自检
> 上例中，为什么不能用“99% VaR：A 为 4、B 为 0”得出 A 比 B 风险更高？
>
> **答案：** 因为 B 的 99% 概率质量恰好停在零损失，左分位在该断点取 0；置信水平一旦提高到 99.5%，B 直接跳到 100，而 A 仍为 4。单个离散分位点不能给出完整尾部排序。

## 边界

- A 的二项分布需要本算例的独立假设；它只用于算出具体分位点。$E[L_A]=1$ 本身不需要独立。
- 比较 VaR 必须使用相同的左分位定义、置信水平、风险期限、损失基准和币种。把 99% 与 99.5% 混在一列没有可比性。
- 相同 EL 不决定 VaR、ES、损失标准差或压力损失；反过来，相同某一 VaR 也不决定更深尾部。
- 分散化不是“贷款笔数多”自动产生的。单名大小、行业集中、担保人重叠和违约依赖都会削弱分散。
- 本例只隔离集中度与依赖边界，不主张现实贷款具有相同 PD、固定 LGD 或严格独立。

## 来源与核验

- Basel Committee on Banking Supervision, [*An Explanatory Note on the Basel II IRB Risk Weight Functions*，第 4.5 节与图 3，第 9–10 页](https://www.bis.org/bcbs/irbriskweight.pdf)：定位核验相同 EL 的组合可因共同系统因子和相关程度不同而具有不同损失波动与 UL。
- RiskMetrics Group, [*CreditMetrics Technical Document*，第 1.1–1.4 节，第 5–17 页；第 3 章，第 35–40 页](https://www.msci.com/documents/10199/93396227-d449-4229-9143-24a94dab122f)：定位核验信用损失的偏斜尾部、集中风险、联合状态以及分位点与标准差的不同信息。
- [[VaR定义]]：承载离散分布左分位及概率质量点边界；[[组合损失聚合]] 与 [[共同违约概率]] 分别承载均值相加和依赖项。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
