---
title: Algorithmic fairness in credit scoring
authors:
  - Teresa Bono
  - Karen Croxson
  - Adam Giles
year: 2021
doi: 10.1093/oxrep/grab020
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/07_Bono et al. (2021) - Algorithmic fairness in credit scoring.pdf]]"
---

# Bono et al. (2021)：复杂模型不必然更不公平

> [!cite] Reference
> Bono, T., Croxson, K., & Giles, A. (2021). Algorithmic fairness in credit scoring. *Oxford Review of Economic Policy, 37*(3), 585-617. https://doi.org/10.1093/oxrep/grab020

> [!summary] 一句话梗概
> 在 80 万名英国借款人的信用档案上，ensemble models 比 penalised logit 更准确，却没有系统性恶化已有的群体公平差异；不过它们也没有消除这些问题，且没有任何模型完全满足所有 fairness criteria。

## 成文结构：把“算法会歧视吗”改写成可检验问题

> [!abstract] 作者先锁定的中心
> **现象**：公众担心机器学习会复制或放大历史偏见，但传统信用评分本身也可能已有群体差异。<br>
> **张力**：模型类型不能直接告诉我们公平性；不同 fairness criteria 也可能给出不同判断。<br>
> **中心判断**：从 logit 切换到 ensemble models 提高了总体准确率，却既没有系统性恶化、也没有消除既有公平问题；公平必须针对具体数据、群体、指标和决策层单独检验。

### Section 路线图

| 原文部分 | 这一部分在全文中的任务 | 它为下一部分打开什么问题 |
|---|---|---|
| I. Introduction | 把宽泛担忧压缩成两个问题：预测质量怎样变化；群体统计公平怎样变化。 | 现有公平研究用什么标准回答这些问题？ |
| II. Related literature | 梳理 performance parity、separation、sufficiency 及其冲突，避免把 fairness 当作单一指标。 | 用什么数据和群体才能实际检验？ |
| III. Data | 分开介绍个人信用档案与地区 census 数据，并说明隐私保护和样本范围。 | 三类模型在同一预测任务上谁更准？ |
| IV. Comparing models on accuracy | 定义违约预测、训练/测试/部署样本，比较 penalised logit、extra trees 与 XGBoost。 | 要检查群体公平，敏感身份如何获得？ |
| V. Developing proxies | 构造 gender proxy，并按 race、health、deprivation 聚类地区，明确代理变量的不完美。 | 在这些群体上，各模型满足哪些公平标准？ |
| VI. Statistical fairness | 依次检验 performance parity、separation、sufficiency，比较模型切换前后差异。 | 差异是否来自敏感属性中额外的信息？ |
| VII. Relationship with scoring data | 检查直接加入敏感属性会否改善预测，以及普通信用特征能否反推出敏感属性。 | 全文应得出“公平”还是“有条件的中性”结论？ |
| VIII. Conclusion | 给出双重结论：ensemble 更准且未普遍恶化公平，但所有模型仍有群体差异。 | — |

### 关键段落组怎样推进

1. **去口号段**：把“AI bias”拆成可观察的 accuracy 和 error-rate 问题。
2. **标准段**：在碰数据前定义三类 fairness criteria，并承认它们不能总被同时满足。
3. **基线段**：先测传统 logit 已有的群体差异，避免把所有问题归因于新模型。
4. **增量准确率段**：确认 ensemble 的样本外优势，建立采用新模型的正面理由。
5. **代理构造段**：详细交代身份代理如何生成，使公平结论的测量误差可见。
6. **逐指标比较段**：同一模型依次接受三类公平测试，不用单一数字给“公平认证”。
7. **机制检查段**：直接加入和反推敏感属性，判断普通信用数据已经编码多少相关信息。
8. **双重结论段**：拒绝两个极端——机器学习不必然更不公平，但更准也不会自动解决公平。

> [!tip] 可迁移到你的 essay
> 把它和 [[05 Fuster et al. (2022) - Predictably unequal#成文结构：从平均预测增益到分配后果|Fuster et al.]] 对读：Bono 停在**预测公平指标**，Fuster 继续走到**利率与准入后果**。你的段落可据此写出层级：模型层没有恶化，不代表市场结果层没有分配问题。

## 研究问题

作者检验从传统逻辑回归切换到 ensemble machine learning 后，整体和各群体预测准确率如何变化，以及 gender、race、health status 与 deprivation 相关群体的 performance parity、separation 和 sufficiency 是否改善或恶化。

## 方法与证据

- 使用 80 万名英国成年人，即约 2% 英国成年人口的信用档案。
- 构造 444 个信用特征；用 2015 年数据训练和测试，再用更晚一期数据模拟部署评价。
- 比较 L2-penalised logistic regression、extremely randomised trees 与 XGBoost。
- 通过称谓推断 gender，并用人口普查地区数据构造 race/health/deprivation clusters。

## 主要发现

- 两个 ensemble models 在总体和主要群体上均比逻辑回归有更高样本外 AUROC。
- 传统模型本身已存在群体间准确率和错误结构差异。
- 切换到机器学习会改善部分指标，但总体上既没有消除、也没有明显扩大检测到的公平问题。
- 直接加入敏感属性并未显著提高违约预测，说明相关信息多已编码在普通信用档案中。

## 关键局限

gender 和 demographic cluster 是代理变量，不是准确的个体身份。研究排除了约 32% 的 thin files，而这些人可能恰是公平风险较高的群体。分析是 observational，只评价分数的统计公平性，不研究银行如何把分数转化为定价、准入和最终福利；不同 fairness criteria 也无法在一般情况下同时完全满足。

## 对 LN905 essay 的用途

它可用来反驳“模型越复杂必然越不公平”的过度推断，并与 [[05 Fuster et al. (2022) - Predictably unequal]] 形成直接对话。两篇合起来支持更精确的结论：公平性不能从模型类型推断，必须在具体数据、指标与最终决策层面单独审计。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/07_Bono et al. (2021) - Algorithmic fairness in credit scoring.pdf|Local PDF]]
- [Published article](https://doi.org/10.1093/oxrep/grab020)
