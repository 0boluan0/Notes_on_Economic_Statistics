---
aliases:
  - "CreditMetrics 以迁移阈值和相关资产收益生成联合评级状态，并用评级对应的远期曲线重估展望期价值"
  - "CreditMetrics migration and revaluation"
  - "CreditMetrics mark-to-market model"
student_os: knowledge-atom
atom_id: RM-CP-008
atom_set: credit-portfolio-risk-and-credit-var
atom_type: model
status: source-checked
mastery_state: unassessed
requires:
  - "[[评级迁移矩阵]]"
  - "[[违约概率口径]]"
  - "[[期限匹配贴现因子]]"
  - "[[信用利差与违约概率]]"
related:
  - "[[Vasicek违约因子]]"
  - "[[CreditRisk+计数]]"
  - "[[无风险利率口径]]"
part_of:
  - "[[信用组合风险与 Credit VaR.canvas|信用组合风险与 Credit VaR]]"
---

# CreditMetrics 以迁移阈值和相关资产收益生成联合评级状态，并用评级对应的远期曲线重估展望期价值
<!-- bilingual-en:start -->
*CreditMetrics generates joint rating states from migration thresholds and correlated asset returns, then revalues horizon values with rating-specific forward curves*
<!-- bilingual-en:end -->

> [!summary] 先生成“到展望期是什么评级”，再问“该评级下值多少钱”
> CreditMetrics 是迁移模式的市值风险框架。当前评级的一行迁移概率被转换为潜在资产收益阈值；相关的潜在资产收益同时落入各自区间，从而给出全组合的联合升级、降级与违约状态。每个非违约状态再用该评级在展望期适用的远期零息曲线重估剩余现金流，违约状态用回收价值。最终对象是组合展望期价值或相对基准价值的损失分布，不只是违约人数。

## 从迁移概率得到评级阈值
<!-- bilingual-en:start -->
*From migration probabilities to rating thresholds*
<!-- bilingual-en:end -->

固定当前评级 $r$，把一年后状态按从差到好排列为
$$
s_1=D, s_2=CCC,\ldots,s_m=AAA,
$$
并从 [[评级迁移矩阵]] 取出完整一行 $p_{r,s_k}$。令标准化潜在资产收益 $Z_i\sim N(0,1)$，定义累计阈值
$$
b_{r,k}=\Phi^{-1}\!\left(\sum_{j=1}^kp_{r,s_j}\right),
\qquad b_{r,0}=-\infty,\qquad b_{r,m}=+\infty.
$$
若
$$
b_{r,k-1}<Z_i\le b_{r,k},
$$
则债务人 $i$ 在展望期进入状态 $s_k$。每个区间的正态概率恰好等于对应迁移概率；违约只是最差的一个状态，升级与非违约降级同样进入损失分布。

对组合而言，令
$$
\mathbf Z=(Z_1,\ldots,Z_n)^\top\sim N(0,\Sigma_A),
$$
其中 $\Sigma_A$ 是潜在资产收益相关矩阵。一次联合抽样给出整组评级状态 $\mathbf S=(S_1,\ldots,S_n)$。若把 $\Sigma_A$ 的非对角元全部设为零，联合状态概率才退化为边际迁移概率的乘积；正相关会提高共同降级与共同违约的概率。迁移矩阵只固定各债务人的边际状态概率，不能单独确定组合联合分布。

## 用评级对应的远期曲线重估
<!-- bilingual-en:start -->
*Revaluation with rating-specific forward curves*
<!-- bilingual-en:end -->

设展望期为 $H$。债券在非违约评级 $s$ 下的展望期价值为
$$
V_i(s;H)=\sum_{u>H}CF_i(u)D_s(H,u),
$$
其中 $D_s(H,u)$ 是从 $H$ 到现金流日 $u$、与评级 $s$ 对应的远期贴现因子。评级变差通常对应更高的信用远期收益率和更低的价值；评级改善则可能产生收益。违约状态不再沿原合同曲线机械贴现，而是使用与债务优先级、回收口径和展望期一致的回收价值，例如
$$
V_i(D;H)=R_i\times EAD_i(H).
$$
把每个联合状态下的单笔价值相加，得到
$$
V_P(\mathbf S;H)=\sum_iV_i(S_i;H),
\qquad
L(\mathbf S)=V_{\mathrm{base}}(H)-V_P(\mathbf S;H).
$$
联合状态的概率和状态价值共同决定损失分布：相关矩阵决定坏状态是否一起出现，远期曲线与回收决定每个状态损失多大。

## 数值锚点：非违约降级也会产生损失
<!-- bilingual-en:start -->
*Numerical anchor: a non-default downgrade can also create loss*
<!-- bilingual-en:end -->

考虑一个极简债权：在展望期后一年只支付 100。以 BBB 远期利率 5% 得到基准展望期价值 $100/1.05=95.24$。

- 若升级到 AA，AA 远期利率为 3%，价值为 $100/1.03=97.09$，相对基准收益约 1.85；
- 若降级到 BB，BB 远期利率为 8%，价值为 $100/1.08=92.59$，虽未违约仍损失约 2.65；
- 若违约并按 40 回收，损失约为 $95.24-40=55.24$。

因此，把 CreditMetrics 简化为“违约时损失、未违约时零损失”会删掉其迁移模式最核心的市值重估。真实债券还要逐笔加入息票、期限、回收随机性和工具特有现金流。

## $\mathbb P$、$\mathbb Q$ 与风险溢价边界
<!-- bilingual-en:start -->
*The boundary among physical probabilities, risk-neutral probabilities, and risk premia*
<!-- bilingual-en:end -->

由历史评级样本估计的迁移矩阵通常描述真实世界测度 $\mathbb P$ 下的未来状态频率。评级远期曲线则来自市场价格，信用利差同时含预期违约损失、系统风险溢价、流动性及其他成分。用历史迁移概率给状态加权、再用市场曲线给状态定价，可以构造真实世界意义下的未来市值分布；这不表示迁移概率已经自动变成风险中性概率。

若目标是经济资本或风险限额，通常需要 $\mathbb P$ 下的联合状态分布和市场一致的状态价值。若目标是无套利定价，则需要与市场价格一致的 $\mathbb Q$ 下迁移/违约动态或明确的风险溢价转换。只因重估用了市场远期曲线，就把历史迁移矩阵标成 $\mathbb Q$，会把“状态概率”和“状态内价值”两个层次混在一起；反过来，从信用利差直接反推 $\mathbb Q$ 迁移也必须说明回收、流动性和风险溢价假设。

> [!question]- 最小自检
> 两位分析者使用同一迁移矩阵和同一组评级远期曲线；甲令债务人的潜在资产收益独立，乙使用正相关矩阵。两者的单笔状态价值相同，组合 Credit VaR 是否必然相同？
>
> **答案：** 不必然相同。迁移矩阵固定边际状态概率，远期曲线固定每个状态的单笔价值；相关矩阵改变共同降级、共同违约等联合状态概率，因此改变组合尾部。

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 迁移阈值要求状态互斥且穷尽；`NR/withdrawn`、到期和主体退出若被删掉，必须重写所估计概率的条件总体。
- 资产收益相关是联合评级模型的输入或映射结果，不等于评级变化指示相关，也不应把短样本股价相关未经验证地直接当成信用相关。
- 评级状态把连续信用质量离散化，同一等级内仍可有利差、期限和工具结构差异；评级曲线不能替代逐笔现金流与优先级。
- 多期迁移不能无条件重复乘同一年度矩阵；需要另行满足一阶 Markov、时间同质或更完整的迁移动态假设。
- CreditMetrics 是迁移/市值模式；[[CreditRisk+计数|CreditRisk+]] 的基础对象是违约计数与违约损失，两者的损失定义不同，VaR 数字不能不说明口径就横向比较。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Gupton, Finger & Bhatia, J.P. Morgan (1997), [*CreditMetrics—Technical Document*](https://www.phy.pmf.unizg.hr/~bp/CMTD1.pdf)，§§2.2–2.4：核验迁移状态、违约回收、评级对应远期零息曲线和展望期重估。
- 同一技术文件 §§6.2–6.3 与 §8.4：核验迁移矩阵到资产收益阈值的映射，以及相关资产收益生成升级、降级与违约联合状态的方法。
- 同一技术文件序言与第 6 章：核验模型以历史迁移资料构造信用质量变化分布；本卡据此将真实世界状态概率与含市场风险溢价的状态估值明确分层，而不把两者误称为同一测度。
- 作者逐项核验日：2026-08-30；独立模型复核通过。
