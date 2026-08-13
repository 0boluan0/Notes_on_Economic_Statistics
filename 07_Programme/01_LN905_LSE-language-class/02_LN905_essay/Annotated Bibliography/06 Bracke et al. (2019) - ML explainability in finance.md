---
title: Machine learning explainability in finance - An application to default risk analysis
authors:
  - Philippe Bracke
  - Anupam Datta
  - Carsten Jung
  - Shayak Sen
year: 2019
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/06_Bracke et al. (2019) - ML explainability in finance.pdf]]"
---

# Bracke et al. (2019)：金融机器学习解释与剩余不确定性

> [!cite] Reference
> Bracke, P., Datta, A., Jung, C., & Sen, S. (2019). *Machine learning explainability in finance: An application to default risk analysis* (Staff Working Paper No. 816). Bank of England.

> [!summary] 一句话梗概
> 作者用基于 Shapley value 的 Quantitative Input Influence 方法解释英国抵押贷款违约模型，证明事后解释能识别总体和个体驱动因素，但模型进入新输入区域时解释会明显变化，不能消除剩余不确定性。

## 成文结构：从单笔解释逐层走向模型风险

> [!abstract] 作者先锁定的中心
> **现象**：金融机构可能采用预测更强的复杂模型，但开发者、管理层、验证人员和监管者询问的“解释”并不相同。<br>
> **张力**：事后工具可以说明已有预测，却未必能证明模型在新状态下仍可靠。<br>
> **中心判断**：解释应从个体贡献逐层扩展到总体行为、相对线性基准、局部模式和压力情景；完成这些步骤后仍会保留模型不确定性。

### Section 路线图

| 原文部分 | 这一部分在全文中的任务 | 它为下一部分打开什么问题 |
|---|---|---|
| 1. Introduction | 先按 stakeholder 列出五类解释问题，并说明结果部分将从 particular 走向 general。 | 什么方法能在不读取模型内部结构时回答这些问题？ |
| 2.1 Existing approaches | 定位 global surrogate、feature importance、local/instance explanation、PDP 等工具的能力边界。 | 本文为什么选择 QII？ |
| 2.2 QII | 用输入干预与 Shapley value 定义单个变量对预测的边际影响。 | 大量个体解释怎样总结为模型层理解？ |
| 2.3 Global cluster explanations | 引入聚类，把相似的局部解释组合成输入空间中的行为区域。 | 需要什么真实应用来检验这套框架？ |
| 3. Data | 建立英国抵押贷款样本、违约结果，以及 logit 与 gradient tree boosting 基准。 | 两类模型的解释在不同层级上怎样表现？ |
| 4.1-4.2 Type 1-2 | 先解释个体贷款，再汇总哪些变量总体驱动预测。 | 复杂模型究竟增加了什么非线性信息？ |
| 4.3-4.4 Type 3-4 | 把 GTB 与 logit 并排比较，再用解释聚类展示不同贷款区域的模型逻辑。 | 已知数据上的解释能否外推到新状态？ |
| 4.5 Type 5 | 用模拟压力情景测试训练和测试分布之外的行为，暴露解释排序变化。 | 最终还能声称模型已经“可解释”吗？ |
| 5. Conclusion | 肯定解释工具的诊断价值，同时明确剩余不确定性不能被解释图消除。 | — |

### 关键段落组怎样推进

1. **问题分类段**：先问不同角色需要知道什么，再选择解释方法；不是工具先行。
2. **方法定位段**：把 QII 放进已有解释工具谱系，说明它回答的是 input influence 而非因果。
3. **个案段**：用两笔贷款展示同一变量在不同个体上作用不同，让非线性变得可见。
4. **总体段**：汇总绝对影响，回答模型总体依赖什么，同时警告平均值会抹平异质性。
5. **基准比较段**：把复杂模型和 logit 并排，具体指出复杂度带来了哪些不同关系。
6. **聚类段**：把成千上万个局部解释压缩为几类贷款，连接 local 与 global。
7. **压力测试段**：主动把模型推到未见状态，观察解释是否改变；这是全文真正的风险检查。
8. **结论段**：把“工具能照亮模型”与“工具不能证明所有状态可靠”同时保留。

> [!tip] 可迁移到你的 essay
> 当你评价 post-hoc explanation 时，可以沿 **individual → aggregate → comparison with logit → clusters/interactions → unseen-state stress test** 推进。这样能精确说明解释工具解决了哪一层问题，又在哪一层停止。

## 研究问题

论文研究在不检查复杂模型内部结构的情况下，能否通过系统改变输入、观察输出并计算边际贡献，为开发者、管理层、模型验证人员和监管者提供不同层次的解释。

## 方法与证据

- 使用英国 FCA 监管抵押贷款数据，清理后约 583 万笔贷款，并跟踪 2015-2017 年表现。
- 比较逻辑回归与 gradient tree boosting 的违约预测。
- 使用 QII/Shapley 计算个体和总体变量影响，再聚类相似的局部解释。
- 设计模拟压力场景，检查模型在训练和测试数据之外的行为。

## 主要发现

- Gradient tree boosting 的 ROC-AUC 约为 0.81，高于逻辑回归的 0.78；precision-recall AUC 约为 0.19 对 0.16。
- current loan-to-value ratio 与 current interest rate 是重要驱动因素，与抵押贷款研究中的 double-trigger 解释相符。
- 不同贷款的解释差异很大，聚类比单一 global importance 更能反映非线性模型。
- 压力情景下，复杂模型的变量影响排序显著变化；在普通测试集上的解释不足以证明模型在新状态下仍按预期工作。

## 关键局限

违约率约 2.5%，即使 ROC-AUC 较高，precision 仍有限。训练和测试是随机切分而非真正的 out-of-time 设计；解释只描述预测关系，不建立因果。压力测试数据是模拟的，QII 的结果也依赖输入干预方式。论文研究的是模型质量保证，不是最终贷款公平性或消费者福利。

## 对 LN905 essay 的用途

它能精确界定 post-hoc explanation 的能力边界：解释工具可以支持审计、异常诊断和 stakeholder communication，但不能替代样本外验证、压力测试与领域知识。因此复杂模型的预测增益只有在这些额外验证都能完成时才更有正当性。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/06_Bracke et al. (2019) - ML explainability in finance.pdf|Local PDF]]
- [Bank of England working paper](https://www.bankofengland.co.uk/working-paper/2019/machine-learning-explainability-in-finance-an-application-to-default-risk-analysis)
