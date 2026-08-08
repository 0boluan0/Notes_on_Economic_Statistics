---
title: Comprehensible credit scoring models using rule extraction from support vector machines
authors:
  - David Martens
  - Bart Baesens
  - Tony Van Gestel
  - Jan Vanthienen
year: 2007
doi: 10.1016/j.ejor.2006.04.051
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/09_Martens et al. (2007) - Comprehensible credit scoring via SVM rule extraction.pdf]]"
---

# Martens et al. (2007)：从 SVM 提取可理解规则

> [!cite] Reference
> Martens, D., Baesens, B., Van Gestel, T., & Vanthienen, J. (2007). Comprehensible credit scoring models using rule extraction from support vector machines. *European Journal of Operational Research, 183*(3), 1466-1476. https://doi.org/10.1016/j.ejor.2006.04.051

> [!summary] 一句话梗概
> 论文比较多种从 SVM 黑箱提取 symbolic rules 的方法，发现可读规则通常能保留大部分预测能力；但 surrogate rules 与原模型并不完全一致，且在部分信用数据上的性能损失并不小。

## 研究问题

作者研究能否在保留 SVM 非线性预测优势的同时，提取足以供人理解、审核和说明拒贷原因的规则，并比较 Trepan、G-REX、C4.5 等提取方法。

## 方法与证据

- 讨论 decompositional 与 pedagogical rule extraction 两类路线。
- 在 Ripley、Iris、Wisconsin Breast Cancer、Australian credit 和 Bene-C corporate credit 五个数据集上测试。
- 比较 SVM、logit、C4.5 与多种提取规则的样本外 accuracy、对原黑箱的 fidelity 和规则数量。

## 主要发现

- SVM 在五个数据集上均达到最高或并列最高准确率。
- Trepan 整体上是表现最好的 pedagogical rule extractor；G-REX 能把规则压得更紧凑。
- 在 Australian credit 数据上，SVM 与 logit 均为 85.7%，Trepan 为 85.1%，显示该数据并不存在明显的复杂模型优势。
- 在 Bene-C 数据上，SVM 为 96.5%，但 Trepan 只有 82.0%、fidelity 为 84.3%，说明“只损失少量性能”并非对每个数据集都成立。

## 关键局限

数据集规模很小且混合了合成、医疗、消费者和企业信用任务；评估设计和显著性方法也早于现代 benchmark 标准。规则可读性主要用规则数近似，没有真实用户或监管者验证。提取规则是黑箱的近似，fidelity 低于 100% 时不能被当作模型实际计算的完整解释。

## 对 LN905 essay 的用途

它提供一个早期中间方案：选择不一定只有逻辑回归或完全不透明的复杂模型。但它也能支持 Rudin 的批评，因为 surrogate rule 的准确率和 fidelity 会下降。最合适的结论是 rule extraction 可辅助沟通，却不能自动消除黑箱治理问题。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/09_Martens et al. (2007) - Comprehensible credit scoring via SVM rule extraction.pdf|Local PDF]]
- [Published article](https://doi.org/10.1016/j.ejor.2006.04.051)

