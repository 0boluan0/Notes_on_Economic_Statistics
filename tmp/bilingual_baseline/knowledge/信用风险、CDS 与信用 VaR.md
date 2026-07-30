---
aliases:
  - "Credit Risk, CDS and Credit VaR"
  - "CreditMetrics"
  - "信用风险"
status: source-checked
---

# 信用风险、CDS 与信用 VaR

> [!summary] 快速恢复
> **它解决什么：** 把借款人不履约的可能性、违约时损失和组合共同违约转成定价、限额和资本指标。
> **具体锚点：** 两笔预期损失相同的贷款，若其中一笔违约与经济衰退高度相关，它的意外损失和资本需求可能更高。
> **核心难点：** 预期损失 $PD\times LGD\times EAD$ 与尾部意外损失不同；评级、市场利差和真实违约概率也不是同一口径。
> **为什么重要：** 信贷定价、拨备、组合集中、CDS 和银行资本都依赖这些区分。
> **继续：** 先分解 PD/LGD/EAD，再看迁移、相关和组合模型；对手方随市场变化的敞口另见 [[对手方信用风险、CVA 与 DVA]]。

> [!source] 本节依据
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。

## PD、LGD、EAD 与预期损失

在指定期限和口径下，PD 是违约概率，LGD 是违约时损失比例，EAD 是违约时敞口。$EL=PD\times LGD\times EAD$ 是期望；实际定价还含资金、运营、风险溢价和资本成本。三项在压力中可能相关。

## 评级与迁移

评级是对信用质量的离散摘要，迁移矩阵给一段时间内等级变化。through-the-cycle 与 point-in-time 评级对周期反应不同。历史迁移率受样本、定义和 regime 影响，不能当固定自然常数。

## 信用利差与 CDS

CDS protection buyer 支付 spread，违约/信用事件时获得补偿。简化下 spread 与风险中性 PD、LGD 相关，但还含流动性、对手风险和技术供需。cash bond spread 与 CDS spread 的 basis 可偏离零。

## 结构与强度模型

Merton 型结构模型把公司资产低于债务阈值视为违约，连接股权期权性；reduced-form/强度模型直接建违约到达率。前者机制强但资产不可观测，后者校准灵活但经济结构较弱。

## 信用组合与相关

CreditMetrics 用评级迁移和市值变化，CreditRisk+ 用违约计数/行业因子，Vasicek/单因子模型用共同系统因子产生违约相关。相关在尾部和集中组合中主导信用 VaR。

## 信用 VaR 与集中

信用损失分布离散、偏斜、厚尾，VaR/ES 需模拟或近似。单一借款人、行业、地域和期限集中应与模型指标并列管理；相关估计的不确定性必须压力测试。

## 最小自检

### 预期损失和信用 VaR/资本各处理什么？

> [!answer]- 答案
> 预期损失是平均可预见损失，通常由定价/拨备覆盖；信用 VaR/资本关注尾部意外损失。
### CDS spread 能否直接除以 LGD 得真实违约概率？

> [!answer]- 答案
> 只能在很强简化下近似风险中性概率；现实还含期限结构、流动性、风险溢价和对手因素。
### 为什么信用组合相关很重要？

> [!answer]- 答案
> 违约共同发生会使损失无法靠分散平均，尾部集中度和资本需求显著上升。

## 来源与核验

- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
