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

