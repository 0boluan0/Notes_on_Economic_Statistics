---
aliases:
  - "Vasicek 信用单因子模型使债务人在给定共同高斯因子后条件独立，并由该因子产生无条件违约依赖"
  - "Vasicek one-factor default model"
  - "Vasicek latent-factor model"
student_os: knowledge-atom
atom_id: RM-CP-006
atom_set: credit-portfolio-risk-and-credit-var
atom_type: model
status: source-checked
mastery_state: unassessed
requires:
  - "[[违约概率口径]]"
  - "[[条件独立]]"
related:
  - "[[条件独立不等于边际独立]]"
  - "[[Merton违约模型]]"
  - "[[ASRF颗粒性]]"
  - "[[事件指示变量]]"
part_of:
  - "[[信用组合风险与 Credit VaR.canvas|信用组合风险与 Credit VaR]]"
---

# Vasicek 信用单因子模型使债务人在给定共同高斯因子后条件独立，并由该因子产生无条件违约依赖
<!-- bilingual-en:start -->
*The Vasicek one-factor credit model makes obligors conditionally independent given a common Gaussian factor, which induces unconditional default dependence*
<!-- bilingual-en:end -->

> [!summary] 共同因子先改变每个债务人的条件 PD，再把违约聚在同一坏状态
> 对债务人 $i$，把标准化潜在资产收益写成共同冲击与特有冲击之和：
> $$
> A_i=\sqrt{\rho}\,Y+\sqrt{1-\rho}\,\varepsilon_i,
> \qquad Y,\varepsilon_i\overset{\text{ind}}\sim N(0,1).
> $$
> 若无条件一年 PD 为 $p_i$，阈值为 $c_i=\Phi^{-1}(p_i)$，则 $A_i\le c_i$ 时违约。给定 $Y$ 后只剩相互独立的 $\varepsilon_i$，所以违约条件独立；忽略 $Y$ 后，所有债务人又共享同一随机来源，因而通常无条件依赖。

## 潜在变量、阈值与条件 PD
<!-- bilingual-en:start -->
*Latent variable, threshold, and conditional probability of default*
<!-- bilingual-en:end -->

令违约指示变量为
$$
D_i=\mathbf 1\{A_i\le c_i\},
\qquad c_i=\Phi^{-1}(p_i).
$$
因为 $A_i\sim N(0,1)$，无条件概率恰为
$$
\Pr(D_i=1)=\Phi(c_i)=p_i.
$$
在共同因子实现为 $Y=y$ 时，条件 PD 是
$$
p_i(y)
=\Pr(D_i=1\mid Y=y)
=\Phi\!\left(
\frac{\Phi^{-1}(p_i)-\sqrt\rho\,y}{\sqrt{1-\rho}}
\right),
\qquad 0\le\rho<1.
$$
这里采用“$Y$ 越低，经济状态越坏”的符号约定；若另一本教材把坏状态因子定义为 $-Y$，公式中的符号会反转，但模型没有改变。对所有可能的因子状态取平均可恢复无条件 PD：$E[p_i(Y)]=p_i$。

给定 $Y=y$ 后，$D_i$ 只取决于各自的 $\varepsilon_i$，所以
$$
\Pr(D_1=d_1,\ldots,D_n=d_n\mid Y=y)
=\prod_{i=1}^n\Pr(D_i=d_i\mid Y=y).
$$
这是一条**条件独立**陈述，不是说真实世界中的违约无条件独立。

## $\rho$ 是潜在资产收益相关，不是违约指示相关
<!-- bilingual-en:start -->
*Rho is latent asset-return correlation, not default-indicator correlation*
<!-- bilingual-en:end -->

在同质载荷写法下，两个不同债务人的潜在收益满足
$$
\operatorname{Corr}(A_i,A_j)=\rho.
$$
但违约指示变量是把连续潜在收益切在不同阈值后的离散结果。其联合违约概率为
$$
\Pr(D_i=1,D_j=1)=\Phi_2(c_i,c_j;\rho),
$$
所以
$$
\operatorname{Corr}(D_i,D_j)
=\frac{\Phi_2(c_i,c_j;\rho)-p_ip_j}
{\sqrt{p_i(1-p_i)p_j(1-p_j)}}.
$$
它依赖 $p_i,p_j$ 和 $\rho$，一般不等于 $\rho$。因此，把“资产相关 20%”直接写成“违约相关 20%”会把潜在连续变量的参数错贴到离散事件上；市场股价相关也不是未经映射就等于这里的潜在资产收益相关。

## 数值锚点：坏因子怎样抬高条件 PD
<!-- bilingual-en:start -->
*Numerical anchor: how a bad factor raises conditional PD*
<!-- bilingual-en:end -->

设 $p_i=1\%$、$\rho=20\%$。当共同因子落到 $y=-2$ 时，
$$
p_i(-2)
=\Phi\!\left(
\frac{\Phi^{-1}(0.01)-\sqrt{0.2}(-2)}{\sqrt{0.8}}
\right)
\approx5.47\%.
$$
单名无条件 PD 仍是 1%，但在同一个坏因子状态中，每个名字的条件 PD 同时上升。这正是“给定因子后独立、边际上共同违约”的机制；不能把 5.47% 当成新的无条件 PD，也不能据此说债务人之间存在直接因果传染。

> [!question]- 最小自检
> 模型员给所有债务人设 $\rho=0.20$，随后把任意两家的违约指示相关也填成 0.20。缺了哪一步？
>
> **答案：** $\rho$ 参数化的是潜在标准化资产收益的相关。必须结合两家的违约阈值或 PD，通过二元正态联合概率 $\Phi_2(c_i,c_j;\rho)$ 再计算违约指示相关；阈值化后的相关一般不是 0.20。

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 单因子、高斯潜在收益、独立特有冲击和阈值违约都是模型假设；行业、地区、期限或非高斯尾部可能需要多因子或其他依赖结构。
- $p_i$ 属于哪个概率测度必须另行说明。用真实世界 PD 构造经济资本分布，不等于用风险中性概率给交易定价；共同因子公式本身不会完成 $\mathbb P$ 与 $\mathbb Q$ 的转换。
- Vasicek 因子机制只解释联合违约怎样生成。把特有风险在组合层面分散掉还需要 [[ASRF颗粒性|渐近细分]]；少数大额敞口不会因为使用了单因子公式就消失。
- 本卡的信用单因子模型不是同名的均值回复短利率模型。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Vasicek (1987), [*Probability of Loss on Loan Portfolio*](https://mx.nthu.edu.tw/~jtyang/Teaching/Risk_management/Papers/Models/Probability%20of%20Loss%20on%20Loan%20Portfolio.pdf)，第 1–3 页：原始技术札记，核验相关潜在资产过程、共同因子分解、违约阈值和条件违约概率。
- Vasicek (2002), [*The Distribution of Loan Portfolio Value*](https://www.maths.univ-evry.fr/pages_perso/crepey/Finance/Vasicek.pdf)，尤其 “The limiting distribution of portfolio losses”：核验条件独立下的大组合损失分布及其与 1987/1991 技术札记的承接关系。
- Gordy (2003), [*A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules*](https://www.federalreserve.gov/pubs/feds/2002/200255/200255pap.pdf)，§1：交叉核验无条件 PD、条件 PD、潜在收益相关与给定系统因子后条件独立的区分。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
