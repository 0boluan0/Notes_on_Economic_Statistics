---
title: Transparency auditability and explainability of machine learning models in credit scoring
authors:
  - Michael Bücker
  - Gero Szepannek
  - Alicja Gosiewska
  - Przemyslaw Biecek
year: 2022
doi: 10.1080/01605682.2021.1922098
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/04_Buecker et al. (2022) - Transparency auditability explainability (author preprint).pdf]]"
---

# Bücker et al. (2022)：TAX4CS 信用评分解释框架

> [!cite] Reference
> Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. *Journal of the Operational Research Society, 73*(1), 70-90. https://doi.org/10.1080/01605682.2021.1922098

> [!summary] 一句话梗概
> 论文提出 TAX4CS，把复杂信用评分模型的解释问题拆成 stakeholder、模型生命周期、解释需求和 XAI 工具四部分；案例说明复杂模型可以被系统审计，但也发现精心构造的传统 scorecard 在简单表格数据上几乎不输复杂模型。

## 研究问题

作者试图回答：如果信用评分模型必须同时满足预测、监管、审计和个体解释要求，应如何把零散的 explainable AI 方法组织成一个可执行的治理流程。

## 方法与证据

- 提出 Transparency, Auditability and eXplainability for Credit Scoring（TAX4CS）框架。
- 使用 FICO HELOC 公共数据：10,459 个样本、23 个原始变量，按 75%/25% 随机分为训练和测试集。
- 比较传统 WOE scorecard、逻辑回归、spline logistic regression、随机森林、SVM、GBM、XGBoost 和 AutoML。
- 从整体性能逐层下钻到 permutation importance、变量响应、局部 attribution、what-if profile 与解释稳定性。

## 主要发现

- 在该数据上，最好的测试表现来自带 spline 的逻辑回归；复杂机器学习对精心处理的 scorecard 只带来很小优势。
- 复杂模型可以用模型级和个体级方法获得与传统 scorecard 可比较的解释表面。
- 解释必须匹配不同 stakeholder 与模型生命周期，单一图表或一次性的 SHAP 输出不足以构成治理。
- 模型行为若违背业务知识，或只在极端样本区域出现，应被视为继续审计而非立即接受的信号。

## 关键局限

实证只使用一个公共数据集和随机切分，没有 out-of-time 测试。TAX4CS 展示的是解释与审计工具链，不等于证明所有 post-hoc explanation 都忠实，也不能把相关性解释为因果关系。框架在真实银行内的维护成本和监管接受度没有被直接测量。

## 对 LN905 essay 的用途

它提供比“透明逻辑回归或不透明黑箱”更细的中间立场：复杂模型可以被采用，但前提是形成完整、可重复的审计流程。与此同时，HELOC 案例支持你的保守基准——当复杂模型只比 scorecard 略好时，复杂度本身缺乏充分理由。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/04_Buecker et al. (2022) - Transparency auditability explainability (author preprint).pdf|Author preprint]]
- [Published article](https://doi.org/10.1080/01605682.2021.1922098)

