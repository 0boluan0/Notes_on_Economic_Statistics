---
aliases:
  - "CreditRisk+ 以条件泊松违约计数和随机行业违约率构造违约损失分布，相关性通过共同因子而非无条件独立进入"
  - "CreditRisk+ Poisson-Gamma count model"
  - "CreditRisk+ default-count model"
student_os: knowledge-atom
atom_id: RM-CP-009
atom_set: credit-portfolio-risk-and-credit-var
atom_type: model
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件独立]]"
  - "[[信用损失三参数]]"
related:
  - "[[随机计数的指标和]]"
  - "[[条件独立不等于边际独立]]"
  - "[[边际分布不定联合分布]]"
  - "[[普通生成函数]]"
  - "[[CreditMetrics估值]]"
  - "[[ASRF颗粒性]]"
part_of:
  - "[[信用组合风险与 Credit VaR.canvas|信用组合风险与 Credit VaR]]"
---

# CreditRisk+ 以条件泊松违约计数和随机行业违约率构造违约损失分布，相关性通过共同因子而非无条件独立进入
<!-- bilingual-en:start -->
*CreditRisk+ builds a default-loss distribution from conditionally Poisson default counts and random sector default rates; dependence enters through common factors, not unconditional independence*
<!-- bilingual-en:end -->

> [!summary] Poisson 提供可计算的条件计数，Gamma 行业因子让坏年份的违约率共同上升
> CreditRisk+ 是 default mode（只看违约损失）的精算型组合模型。给定行业因子后，各债务人的违约计数用低 PD 的 Poisson 近似并条件独立；行业违约率本身是随机的，常用 Gamma 因子表示。共享同一随机行业率的债务人在边际上不独立，因此会出现成簇违约。净敞口或损失严重度被离散成损失单位后，概率生成函数与递推可计算完整离散损失分布。

## 条件 Poisson 与随机行业违约率
<!-- bilingual-en:start -->
*Conditional Poisson counts and random sector default rates*
<!-- bilingual-en:end -->

设债务人 $i$ 的平均一年 PD 为 $p_i$。令 $S_k$ 是第 $k$ 个行业或系统因子，并采用
$$
S_k\sim\operatorname{Gamma}\!\left(\alpha_k,\text{scale}=\frac1{\alpha_k}\right),
\qquad E[S_k]=1,\qquad\operatorname{Var}(S_k)=\frac1{\alpha_k}.
$$
若 $w_{i0}+\sum_{k=1}^Kw_{ik}=1$，则可把债务人 $i$ 的条件计数率写成
$$
\lambda_i(\mathbf S)
=p_i\left(w_{i0}+\sum_{k=1}^Kw_{ik}S_k\right),
\qquad
N_i\mid\mathbf S\sim\operatorname{Poisson}(\lambda_i(\mathbf S)).
$$
给定 $\mathbf S$ 后假设各 $N_i$ 条件独立。因 $E[\lambda_i(\mathbf S)]=p_i$，平均计数仍与输入 PD 对齐；但共同因子变化会同时移动许多 $\lambda_i$。

对两个不同债务人，条件协方差为零，而无条件协方差为
$$
\operatorname{Cov}(N_i,N_j)
=\operatorname{Cov}(\lambda_i(\mathbf S),\lambda_j(\mathbf S))
=p_ip_j\sum_{k=1}^Kw_{ik}w_{jk}\operatorname{Var}(S_k),
$$
其中假定不同 $S_k$ 相互独立。只要两家对同一随机因子的载荷为正，边际协方差便为正。这正是 [[条件独立不等于边际独立|条件独立不推出边际独立]] 的信用组合实例。

## 从计数到损失严重度
<!-- bilingual-en:start -->
*From counts to loss severities*
<!-- bilingual-en:end -->

基础模型先选择损失单位 $L_0$，再把债务人违约净损失 $EAD_i\times LGD_i$ 映射成整数单位
$$
\nu_i\approx\frac{EAD_i\times LGD_i}{L_0}.
$$
组合损失写成
$$
L=L_0\sum_i\nu_iN_i.
$$
因此，次数与严重度承担不同角色：$N_i$ 决定违约事件数，$\nu_iL_0$ 决定一次事件损失多大。把所有 $\nu_i$ 设成 1 只能得到违约数分布，不能得到金额损失分布；把回收率、EAD 或担保误差藏进同一个平均 PD 也会破坏对象口径。

## PGF 与递推的计算边界
<!-- bilingual-en:start -->
*Computational boundary of the PGF and recursion*
<!-- bilingual-en:end -->

条件于 $\mathbf S$，整数损失单位数 $M=L/L_0$ 的[[普通生成函数|概率生成函数]]为
$$
G(z\mid\mathbf S)
=\exp\!\left[\sum_i\lambda_i(\mathbf S)(z^{\nu_i}-1)\right].
$$
定义
$$
Q_0(z)=\sum_ip_iw_{i0}(z^{\nu_i}-1),
\qquad
Q_k(z)=\sum_ip_iw_{ik}(z^{\nu_i}-1).
$$
对独立 Gamma 因子积分后得到
$$
G(z)=\exp(Q_0(z))
\prod_{k=1}^K
\left(1-\frac{Q_k(z)}{\alpha_k}\right)^{-\alpha_k}.
$$
展开 $G(z)=\sum_{m\ge0}g_mz^m$ 后，$g_m=\Pr(L=mL_0)$。若先把
$$
\log G(z)=a_0+\sum_{j\ge1}a_jz^j,
$$
写成幂级数，则系数可用
$$
g_0=e^{a_0},
\qquad
g_m=\frac1m\sum_{j=1}^m j a_jg_{m-j}
$$
递推。PGF 是模型分布的编码，递推只是提取离散概率的数值方法；损失单位选择、四舍五入、截断长度和数值稳定性都要单独验证。所谓“解析可算”是在 Poisson 近似、Gamma 因子和离散严重度设定内成立，不代表对原始 Bernoulli 组合没有近似误差。

## 两个真正区分边界的算例
<!-- bilingual-en:start -->
*Two examples that locate the real boundaries*
<!-- bilingual-en:end -->

**共同因子不等于无条件独立。** 两家债务人的 $p_1=p_2=2\%$，都只受同一个 $S$ 驱动，且 $E[S]=1$、$\operatorname{Var}(S)=0.5$。令 $\lambda_i=0.02S$。给定 $S$ 后两计数独立，但
$$
\operatorname{Cov}(N_1,N_2)
=0.02^2\times0.5=0.0002>0.
$$
因此，共享随机违约率已经产生无条件依赖；把联合分布写成两个无条件 Poisson 边际的乘积会漏掉成簇违约。

**Poisson 只是低 PD 的单名近似。** 若一个债务人的一年 PD 高达 20%，却仍用 $N\sim\operatorname{Poisson}(0.2)$，模型给出
$$
\Pr(N\ge2)=1-e^{-0.2}(1+0.2)\approx1.75\%.
$$
现实中同一债务人在同一固定期限内只能首次违约一次，所以 Bernoulli 指示变量不可能取 2。低 PD 时这部分概率很小，近似便利；高 PD、大额单名或需要精确单名事件时，应保留 Bernoulli/精确卷积或采用经验证的替代实现。

> [!question]- 最小自检
> 报告写道：“CreditRisk+ 假设每个债务人违约都相互独立，因此无法产生违约相关。”这句话错在哪里？
>
> **答案：** 基础计算假设的是给定行业因子后的条件独立。行业违约率是共同随机变量；把它积分掉后，共同载荷债务人的计数通常有正协方差。只有固定所有违约率且无共同随机因子时，才会得到相应的无条件独立版本。

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- CreditRisk+ 的基础对象是违约事件和违约损失，不含未违约评级升级/降级的市值变化；后者见 [[CreditMetrics估值]]。
- 本卡在固定风险期限使用 Poisson 分布近似计数；这不自动假定跨期限计数构成完整的 [[齐次泊松过程|齐次 Poisson 过程]]。固定期限的边际分布不能替代跨期限联合结构。
- Gamma 行业率是为正值、过度离散和解析可处理性选择的分布假设，不是由“行业”二字自动证明；独立行业因子、载荷和违约率波动必须校准与压力测试。
- Poisson 允许同一索引出现多个事件，因而是小单名 PD 下对 Bernoulli 的近似。组合解析便利不能取消高 PD 或大额集中造成的近似风险。
- 固定/离散严重度便于 PGF 和递推；随机 LGD、连续严重度、相关回收或展望期 EAD 需要扩展、数值卷积、FFT 或模拟，并要保持 [[信用损失三参数|PD、LGD、EAD]] 口径一致。
- 递推算出一串系数不等于模型已验证；至少应检查概率非负、总和接近 1、均值与输入 EL 对齐，并对损失单位和尾部截断做敏感性分析。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Credit Suisse First Boston (1997), [*CreditRisk+: A Credit Risk Management Framework*](https://globalriskguard.com/resources/credit/creditrisk.pdf)，§§2.3–3.5：原始技术文件镜像，核验违约率波动、Poisson 违约频数、行业共同因子、损失严重度与集中分析。
- 同一技术文件 Appendix A：核验损失 PGF、Gamma–Poisson 混合及由幂级数/递推提取损失概率的实现结构。
- 本卡把原文件的计算假设显式拆成“给定因子后条件独立”与“积分因子后无条件依赖”，并以协方差恒等式复核；Poisson 高 PD 算例按其概率质量函数重算。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
