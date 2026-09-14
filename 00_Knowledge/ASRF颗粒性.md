---
aliases:
  - "给定系统因子后的条件独立与渐近细分使特有风险消失；ASRF 还依赖单一系统因子来得到组合不变的单笔资本结论"
  - "ASRF single-factor and granularity conditions"
  - "Asymptotic single risk factor model"
student_os: knowledge-atom
atom_id: RM-CP-007
atom_set: credit-portfolio-risk-and-credit-var
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Vasicek违约因子]]"
  - "[[条件独立]]"
related:
  - "[[条件独立不等于边际独立]]"
  - "[[均方收敛]]"
  - "[[CreditMetrics估值]]"
  - "[[VaR非次可加]]"
part_of:
  - "[[信用组合风险与 Credit VaR.canvas|信用组合风险与 Credit VaR]]"
---

# 给定系统因子后的条件独立与渐近细分使特有风险消失；ASRF 还依赖单一系统因子来得到组合不变的单笔资本结论
<!-- bilingual-en:start -->
*Conditional independence and asymptotic granularity eliminate idiosyncratic risk; ASRF additionally relies on a single systematic factor for portfolio-invariant asset-level capital contributions*
<!-- bilingual-en:end -->

> [!summary] ASRF 是一组极限条件，不是任意单因子公式的别名
> 令 $w_i$ 是单名敞口占组合总敞口的比例，$U_i\in[0,1]$ 是单位敞口损失，组合损失率为 $L_n=\sum_iw_iU_i$。若给定系统因子向量 $X$ 后各 $U_i$ 条件独立，并且随着组合扩大有 $\max_iw_i\to0$，那么条件特有风险消失，$L_n$ 收敛到条件期望 $E[L_n\mid X]$；这一步不要求 $X$ 只有一维。只有再把系统风险限制为单一因子，并加入适当的连续、单调和分位稳定条件，组合损失分位才可映射到同一个因子分位，从而产生组合不变的单笔资本贡献。

## 渐近细分怎样消去特有风险
<!-- bilingual-en:start -->
*How asymptotic granularity removes idiosyncratic risk*
<!-- bilingual-en:end -->

设 $\sum_iw_i=1$，并令
$$
L_n=\sum_{i=1}^n w_iU_i,
\qquad
m_n(X)=E[L_n\mid X]=\sum_{i=1}^nw_iE[U_i\mid X].
$$
给定 $X$ 后若 $U_i$ 相互独立，则
$$
\operatorname{Var}(L_n\mid X)
=\sum_{i=1}^nw_i^2\operatorname{Var}(U_i\mid X)
\le\frac14\sum_{i=1}^nw_i^2.
$$
又因为
$$
\sum_iw_i^2\le(\max_iw_i)\sum_iw_i=\max_iw_i,
$$
因此
$$
E\!\left[(L_n-m_n(X))^2\right]
=E\!\left[\operatorname{Var}(L_n\mid X)\right]
\le\frac14\max_iw_i\longrightarrow0.
$$
也就是说，$L_n-m_n(X)$ 在 [[均方收敛|$L^2$ 中]]、因而在概率上收敛到 0。这才是“特有风险被分散”的数学内容：每一笔相对于总组合都可忽略，而不是只要债务人数量看起来很多。这里没有声称几乎处处收敛；若要得到更强的收敛方式，还需额外的速率或可求和条件。

若某一名字始终占 20%，则 $\max_iw_i\ge0.20$；无论把剩余 80% 拆成多少小笔，该名字的条件 Bernoulli 跳跃仍保留在组合损失中。债务人数趋于无穷并不自动等于渐近细分。

## 单一系统因子为什么关系到组合不变性
<!-- bilingual-en:start -->
*Why a single systematic factor is required for portfolio invariance*
<!-- bilingual-en:end -->

在渐近细分后，损失随机性来自 $m_n(X)$。若 $X$ 是一个标量，且沿本库 [[Vasicek违约因子]] 的符号约定“$X$ 越低、损失越高”，$m_n(x)$ 连续且严格递减，则
$$
\operatorname{VaR}_q(m_n(X))=m_n(x_q),
\qquad x_q=F_X^{-1}(1-q).
$$
在额外满足分位稳定等正则条件时，颗粒性收敛进一步给出
$$
\left|\operatorname{VaR}_q(L_n)-m_n(x_q)\right|\longrightarrow0.
$$
若采用“因子越高、损失越高”的相反符号约定，对应因子分位改为 $F_X^{-1}(q)$。每笔资产都在同一个尾部因子状态下计算条件期望损失，因此单笔资本率可以只依赖该资产自己的 PD、LGD、期限和因子敏感度，而不再依赖它被放进哪一个合格组合；这就是 **portfolio invariance（组合不变性）**。

若系统状态是多维向量 $(X_1,X_2)$，不存在一个对所有组合都相同的标量坏状态分位。偏重美国周期的组合与偏重欧洲周期的组合，即使都很细分，其尾部方向也不同；一笔贷款的边际资本便会依赖其余组合的因子构成。多因子模型仍可计算组合风险，但不能直接复用 ASRF 的组合不变单笔资本结论。

## 数值边界：名字多不等于细分
<!-- bilingual-en:start -->
*Numerical boundary: many names do not necessarily imply granularity*
<!-- bilingual-en:end -->

固定某个系统状态，假设每笔单位损失的条件违约概率都是 $q=2\%$、LGD 为 100%。

- **10,000 笔等权贷款：** $\sum_iw_i^2=1/10{,}000$，条件损失率标准差为
  $$
  \sqrt{q(1-q)\sum_iw_i^2}=\sqrt{0.02\times0.98/10{,}000}=0.14\text{ 个百分点}.
  $$
- **一笔占 20%，其余敞口任意细分：** 仅这一个名字就使条件标准差至少为
  $$
  \sqrt{0.02\times0.98\times0.20^2}=2.8\text{ 个百分点}.
  $$

第二个组合可以拥有上万名债务人，却仍保留显著单名跳跃。把第一种组合的 ASRF 分位公式直接套给第二种，会漏掉颗粒性或集中度调整。

> [!question]- 最小自检
> 一个组合满足给定系统因子后条件独立，而且有 50,000 名债务人；最大一笔占总敞口 15%。可以只凭“名字很多”宣布特有风险已经分散吗？
>
> **答案：** 不可以。渐近细分检查的是最大单名份额能否趋于零，而不是债务人数。15% 的单名条件损失跳跃不会被其余小额资产平均掉。

## Vasicek、ASRF 与 Basel IRB 不能互换
<!-- bilingual-en:start -->
*Vasicek, ASRF, and Basel IRB are not interchangeable*
<!-- bilingual-en:end -->

- [[Vasicek违约因子|Vasicek 信用单因子模型]]给出一种具体的高斯潜在变量、阈值违约与条件 PD 机制；使用它不自动满足颗粒性。
- ASRF 给出在一类风险因子模型中获得极限损失与组合不变性的结构条件；其核心结论不要求把所有名字设成同一 PD，也不等于一条完整监管资本公式。
- Basel IRB 风险权重函数是在监管口径下使用 PD、LGD、EAD、监管资产相关函数、期限调整、预期损失处理和 RWA 换算的实施规则。其公式受具体资产类别与监管参数约束，不能把 Vasicek 条件 PD、ASRF 极限定理或 CRE31 的完整 $K$ 公式当成同一个对象互相替换。

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 条件独立必须给定足以吸收共同变化的系统因子；遗漏行业或地区共同因子会把残余相关误称为特有风险。
- 渐近结论不说明有限组合误差一定很小。单名、行业或地区集中应做颗粒性调整、压力测试或完整组合模拟。
- 组合不变性是一种在严格条件下的资本分配性质，不是 VaR 在任意拆分、合并或集中组合中都表现良好的保证。
- 单因子假设可用于可实施的监管近似，却不证明现实经济只有一个信用周期。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Gordy (2003), [*A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules*](https://www.federalreserve.gov/pubs/feds/2002/200255/200255pap.pdf)，摘要、§§1–2 与结论：原始组合不变性论文，核验单一系统因子、条件独立、渐近细分、有限颗粒性和多因子边界。
- Basel Committee on Banking Supervision (2005), [*An Explanatory Note on the Basel II IRB Risk Weight Functions*](https://www.bis.org/publications/explanatory-note-basel-ii-irb-risk-weight-functions)，尤其 §§3–4：核验 ASRF 经济基础与 IRB 风险权重函数之间的定位关系。
- Basel Committee on Banking Supervision, [CRE31: IRB approach—risk weight functions](https://www.bis.org/basel_framework/chapter/CRE/31.htm)：核验现行监管公式另含资产类别、相关、期限、PD/LGD/EAD、预期损失与 RWA 口径，不能缩写成单独的 Vasicek 条件 PD。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
