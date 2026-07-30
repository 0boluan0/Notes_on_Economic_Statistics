---
aliases:
  - "Credit Portfolio Risk and Credit VaR"
  - "Credit VaR"
  - "信用组合风险"
status: source-checked
---

# 信用组合风险与 Credit VaR
<!-- bilingual-en:start -->
*Credit Portfolio Risk and Credit VaR*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 把许多借款人的违约、评级迁移和共同系统因子组合成损失分布，从平均预期损失中分离尾部意外损失与集中风险。
> **具体锚点：** 两笔贷款各有 1% PD；若独立，同时违约约为 0.01%，若都由同一房地产冲击驱动，同时违约概率可远高于独立乘积。
> **核心难点：** 单体 PD/LGD 不够，尾部由违约依赖、敞口集中、迁移、市值重估和参数不确定性共同决定。
> **为什么重要：** 资本、组合限额、行业集中与信用风险转移关注的是坏状态中的共同损失，而不是逐笔平均相加。
> **继续：** 单体参数见 [[信用风险：PD、LGD、EAD 与评级迁移]]；风险度量定义见 [[VaR、ES 与回测]]；依赖建模见 [[相关性、Copula 与尾部依赖]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** It combines default, rating migration, and common systematic factors across many borrowers into a loss distribution and separates tail unexpected loss and concentration from average expected loss.
> **Concrete anchor:** Two loans each have 1% PD. If independent, joint default is about 0.01%; if both are driven by the same property shock, joint default probability can be far above the independence product.
> **Central difficulty:** Individual PD and LGD are insufficient. Tail risk jointly depends on default dependence, exposure concentration, migration, mark-to-market change, and parameter uncertainty.
> **Why it matters:** Capital, portfolio limits, industry concentration, and credit-risk transfer concern common losses in bad states rather than a sum of stand-alone averages.
> **Continue:** For single-name parameters, see [[信用风险：PD、LGD、EAD 与评级迁移|Credit Risk: PD, LGD, EAD, and Rating Migration]]. For risk-measure definitions, see [[VaR、ES 与回测|VaR, Expected Shortfall, and Backtesting]]. For dependence modeling, see [[相关性、Copula 与尾部依赖|Correlation, Copulas, and Tail Dependence]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
<!-- bilingual-en:start -->
> [!source] Basis for this section
> - The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
> - The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
> - Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
<!-- bilingual-en:end -->

## 预期损失与意外损失
<!-- bilingual-en:start -->
*Expected and unexpected loss*
<!-- bilingual-en:end -->

预期损失是损失分布均值，通常由定价、利差与拨备吸收；意外损失是围绕均值尤其尾部的波动，资本用于提高坏状态下的生存能力。Credit VaR 常表示给定置信水平损失分位与期望损失之差，但机构必须明确口径。
<!-- bilingual-en:start -->
Expected loss is the mean of the loss distribution and is normally absorbed through pricing, spread, and provisions. Unexpected loss is variation around the mean, especially in the tail, and capital supports survival in bad states. Credit VaR often means the difference between a chosen loss quantile and expected loss, but institutions must state the convention.
<!-- bilingual-en:end -->

## 信用组合与相关
<!-- bilingual-en:start -->
*Credit portfolios and dependence*
<!-- bilingual-en:end -->

CreditMetrics 用评级迁移和市值变化，CreditRisk+ 用违约计数/行业因子，Vasicek/单因子模型用共同系统因子产生违约相关。相关在尾部和集中组合中主导信用 VaR。
<!-- bilingual-en:start -->
CreditMetrics uses rating migration and mark-to-market changes, CreditRisk+ uses default counts and sector factors, and Vasicek or one-factor models generate default dependence through a common systematic factor. Dependence dominates credit VaR in tails and concentrated portfolios.
<!-- bilingual-en:end -->

模型对象不同：只计违约损失的 default mode 适合持有到期与资本问题；迁移模式在未违约降级时也重估价值，更适合交易或市值组合。不能把两种损失分布的 VaR 数字直接比较而不说明损益定义。
<!-- bilingual-en:start -->
The modeled object differs. Default mode counts only default loss and suits hold-to-maturity and capital questions. Migration mode revalues exposure after non-default downgrades and better suits trading or mark-to-market portfolios. Their VaR numbers cannot be compared without stating the P&L definition.
<!-- bilingual-en:end -->

## 信用 VaR 与集中
<!-- bilingual-en:start -->
*Credit VaR and concentration*
<!-- bilingual-en:end -->

信用损失分布离散、偏斜、厚尾，VaR/ES 需模拟或近似。单一借款人、行业、地域和期限集中应与模型指标并列管理；相关估计的不确定性必须压力测试。
<!-- bilingual-en:start -->
Credit-loss distributions are discrete, skewed, and heavy-tailed, so VaR and ES require simulation or approximation. Single-name, industry, geographic, and maturity concentration should be managed alongside model measures, and uncertainty in dependence estimates must be stressed.
<!-- bilingual-en:end -->

### 数值锚点：相同 EL、不同尾部
<!-- bilingual-en:start -->
*Numerical anchor: equal expected loss, different tails*
<!-- bilingual-en:end -->

组合 A 有 100 笔各 1 单位贷款，每笔 PD 1%、LGD 100%，近似独立，EL 为 1。组合 B 只有一笔 100 单位贷款，PD 1%、LGD 100%，EL 也为 1。99% 左分位的边界取决于定义，但 B 的损失高度集中为 0 或 100；A 的损失围绕少数违约分散。相同 EL 完全不代表相同资本或集中风险。
<!-- bilingual-en:start -->
Portfolio A has 100 loans of one unit each, each with PD 1% and LGD 100%, approximately independent, so EL is 1. Portfolio B has one loan of 100 units with PD 1% and LGD 100%, also EL 1. The exact 99% left-quantile boundary depends on convention, but B's loss is concentrated at either zero or 100, while A spreads loss across a small number of defaults. Equal EL says nothing about equal capital or concentration risk.
<!-- bilingual-en:end -->

## 模型实施流程
<!-- bilingual-en:start -->
*Model implementation workflow*
<!-- bilingual-en:end -->

1. 固定损失口径、期限、组合边界与违约定义。
2. 校准单体 PD、LGD、EAD 或迁移矩阵。
3. 选择共同因子、资产相关或 copula，并映射行业与地区暴露。
4. 模拟系统状态、条件违约/迁移与回收，聚合损失。
5. 计算 EL、分位、ES 与贡献，并对集中和相关做压力。
6. 用实际迁移、违约、回收与组合损失验证不同层级。
<!-- bilingual-en:start -->
1. Fix loss convention, horizon, portfolio boundary, and default definition.
2. Calibrate individual PD, LGD, EAD, or migration matrices.
3. Choose common factors, asset correlation, or copula and map industry and geographic exposure.
4. Simulate systematic states, conditional default or migration, and recovery, then aggregate loss.
5. Calculate EL, quantiles, ES, and contributions and stress concentration plus dependence.
6. Validate at multiple levels using realized migration, default, recovery, and portfolio loss.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnosis*
<!-- bilingual-en:end -->

- 将每笔 EL 相加后称为信用 VaR，完全没有损失分布。
- 用正常期线性相关建危机共同违约，却不做尾部或行业压力。
- 组合高度集中时依赖大数分散近似，低估单名跳跃。
- 模型校准与验证都使用同一短期低违约样本，参数看似稳定但检验无力。
<!-- bilingual-en:start -->
- Adding stand-alone EL and calling the sum credit VaR produces no loss distribution.
- Modeling crisis joint default with normal-period linear correlation without tail or industry stress.
- Applying large-pool diversification approximations to a concentrated portfolio understates single-name jumps.
- Calibrating and validating on the same short low-default sample creates apparently stable parameters but weak tests.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 预期损失和信用 VaR/资本各处理什么？
<!-- bilingual-en:start -->
*What do expected loss and credit VaR or capital address respectively?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 预期损失是平均可预见损失，通常由定价/拨备覆盖；信用 VaR/资本关注尾部意外损失。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Expected loss is the average foreseeable loss normally covered by pricing and provisions; credit VaR and capital focus on tail unexpected loss.
<!-- bilingual-en:end -->

### 为什么信用组合相关很重要？
<!-- bilingual-en:start -->
*Why is credit-portfolio dependence important?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 违约共同发生会使损失无法靠分散平均，尾部集中度和资本需求显著上升。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Joint defaults prevent losses from averaging away through diversification and materially raise tail concentration and capital needs.
<!-- bilingual-en:end -->

### 用自己的话解释：为什么 100 笔小贷款与一笔同额大贷款即使 EL 相同也不等价？
<!-- bilingual-en:start -->
*Explain in your own words: why are 100 small loans and one equally large loan not equivalent even with equal EL?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 小贷款若依赖较低，个别违约可被大量存续贷款平均；单笔大贷款一旦违约便一次损失全部集中，尾部分布与风险贡献完全不同。
<!-- bilingual-en:start -->
> [!answer]- Answer
> If small loans have limited dependence, individual defaults are averaged against many survivors. One large loan produces a concentrated full loss upon default, giving a very different tail distribution and risk contribution.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
- 已逐项核验 default/migration mode、CreditMetrics、CreditRisk+ 与单因子依赖的对象差异；集中算例按离散损失分布复核。
<!-- bilingual-en:start -->
- The vault's Financial Institutions and Risk Management course notes support course scope, classroom examples, and notation.
- The [Basel Framework](https://www.bis.org/basel_framework/) verifies regulatory conventions for capital, market risk, credit risk, and CVA.
- Hull, *Risk Management and Financial Institutions*, cross-checks VaR, ES, Greeks, interest-rate risk, credit risk, and simulation methods.
- Differences among default and migration modes, CreditMetrics, CreditRisk+, and one-factor dependence were checked item by item; the concentration example was verified from its discrete loss distribution.
<!-- bilingual-en:end -->
