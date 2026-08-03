---
aliases:
  - LN905 writing assignment
  - LN905 写作作业
student_os: learning-plan
title: LN905 Essay
track: school
status: active
kind: multi-stage-project
---

# LN905 Essay

> [!summary] 当前项目
> 在 2026-08-26 前完成一篇与未来学位课程相关的 2,000 字论文（具体截止时间待核验）。题目与理由已于 2026-08-01 提交 Moodle，并核验为 Submitted for grading；当前阶段是 2026-08-09 前完成 annotated bibliography。
> 六篇核心来源已于 2026-08-03 选定、下载并核验；下一步是精读来源 1–3。
>
> <!-- bilingual-en:start -->
> **Update, 2 August 2026:** The question and rationale were submitted to Moodle on 1 August and verified as “Submitted for grading”. The current stage is the annotated bibliography due on 9 August.
>
> **Update, 3 August 2026:** Six core sources have been selected, downloaded and checked. The next step is close reading of sources 1–3.
> <!-- bilingual-en:end -->

## 执行清单

- [x] LN905 Essay｜收集 3 个候选问题 #student-os/task ✅ 2026-07-30
- [x] LN905 Essay｜比较候选问题并选定 1 个、起草理由 #student-os/task ✅ 2026-08-01
- [x] LN905 Essay｜在 Moodle 提交题目与选题理由 #student-os/task ✅ 2026-08-01
- [x] LN905 Essay｜检索候选学术文献并选出至少 6 篇与论证直接相关的来源 #student-os/task ✅ 2026-08-03
- [ ] LN905 Essay｜精读来源 1–3，提取核心 claim、证据、用途与局限 #student-os/task ⏳ 2026-08-04
- [ ] LN905 Essay｜精读来源 4–6，提取核心 claim、证据、用途与局限 #student-os/task ⏳ 2026-08-05
- [ ] LN905 Essay｜写完并统一格式化 annotated bibliography #student-os/task ⏳ 2026-08-06
- [ ] LN905 Essay｜上传 annotated bibliography 并检查 Moodle submission status #student-os/task ⏳ 2026-08-08
- [ ] LN905 Essay｜把文献关系整理成支持 thesis 的 argument map #student-os/task ⏳ 2026-08-09
- [ ] LN905 Essay｜搭出 detailed essay plan：thesis、段落 claims 与顺序 #student-os/task ⏳ 2026-08-10
- [ ] LN905 Essay｜为每段补上证据、引用与可能的反方观点 #student-os/task ⏳ 2026-08-11
- [ ] LN905 Essay｜按作业要求修订并完成 detailed essay plan 内容 #student-os/task ⏳ 2026-08-12
- [ ] LN905 Essay｜上传 detailed essay plan 并检查 Moodle submission status #student-os/task ⏳ 2026-08-13

## 里程碑

![[deadlines#2026-08]]

## 选题工作区

- 状态：已于 2026-08-01 提交 Moodle，并核验为 Submitted for grading。
- 中文工作译名：复杂机器学习模型相对逻辑回归的预测提升，在多大程度上足以抵消消费者信用评分中可解释性的下降？

<!-- bilingual-en:start -->
- **Status:** Submitted on Moodle on 1 August 2026 and verified as “Submitted for grading”.
- **Final question:** To what extent do the predictive gains of complex machine-learning models over logistic regression justify reduced interpretability in consumer credit scoring?
<!-- bilingual-en:end -->

### 选题理由

我选择这个问题，是因为它把 ST447 和 ST443 中的统计建模与机器学习内容，同 ST445 的数据管理和分类器评估技能连接起来。信用评分是一个具体的高风险应用场景，需要同时考虑预测性能、可解释性和模型治理。这个题目也建立在我已有的计量经济学与金融风险管理基础上，并可为以后涉及金融或其他受监管领域的 Capstone Project 提供概念基础。

<!-- bilingual-en:start -->
I have chosen this question because it connects the statistical modelling and machine-learning content of ST447 and ST443 with the data-management and classifier-evaluation skills developed in ST445. Credit scoring provides a concrete, high-stakes application in which predictive performance, interpretability and model governance must be considered together. The topic also builds on my previous study of econometrics and financial risk management and could provide a useful conceptual foundation for a later Capstone Project involving finance or another regulated domain.
<!-- bilingual-en:end -->

### 暂定论点

本文将主张：复杂模型只有在样本外预测提升显著且稳定，并且能够满足模型验证、公平性审查和监管问责要求时才值得采用；否则，逻辑回归仍是更合理的基准模型。

<!-- bilingual-en:start -->
This essay will argue that complex models are justified only when their out-of-sample predictive gains are material and stable and when they remain sufficiently transparent for model validation, fairness review and regulatory accountability; otherwise, logistic regression remains the more appropriate baseline.
<!-- bilingual-en:end -->

## 六篇核心来源

<!-- bilingual-en:start -->
*Six core sources*
<!-- bilingual-en:end -->

> [!info] 选择逻辑
> 这组来源共同覆盖论证所需的四个环节：预测性能、经济收益、可解释性与审计，以及公平和分配后果。其中两份本地 PDF 是作者预印本；项目页同时保留正式发表版本的 DOI。
>
> <!-- bilingual-en:start -->
> Together, these sources cover the four parts of the argument: predictive performance, economic value, interpretability and auditability, and fairness and distributional consequences. Two local PDFs are author preprints; the DOI of the published version is retained here in each case.
> <!-- bilingual-en:end -->

**1.** [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/01_Lessmann et al. (2015) - Benchmarking credit-scoring algorithms.pdf|Lessmann et al. (2015)]] · [DOI](https://doi.org/10.1016/j.ejor.2015.05.030)：大规模比较 8 个真实信用评分数据集上的 41 个分类器，用来判断复杂模型相对逻辑回归的提升是否稳定、是否随评估指标和误分类成本而变化。

<!-- bilingual-en:start -->
**Argument use:** A large benchmark of 41 classifiers across eight real credit-scoring datasets. It tests whether gains over logistic regression are stable and whether they change with the performance measure and misclassification costs.
<!-- bilingual-en:end -->

**2.** [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/02_Khandani et al. (2010) - Consumer credit-risk ML models.pdf|Khandani, Kim and Lo (2010)]] · [DOI](https://doi.org/10.1016/j.jbankfin.2010.06.001)：使用大型商业银行的消费者交易与征信数据，并把预测改进转化为约占总损失 6%–25% 的潜在节约，为“提升是否足够重要”提供经济尺度。

<!-- bilingual-en:start -->
**Argument use:** Uses consumer transaction and credit-bureau data from a major bank and translates predictive gains into estimated savings of 6–25% of total losses, giving an economic scale for what counts as a material gain.
<!-- bilingual-en:end -->

**3.** [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/03_Rudin (2019) - Stop explaining black-box models (author preprint).pdf|Rudin (2019) · author preprint]] · [published DOI](https://doi.org/10.1038/s42256-019-0048-x)：提出高风险决策不应依赖事后解释黑箱，而应优先使用内生可解释模型，是反对“用少量准确率换透明度”的核心理论来源。

<!-- bilingual-en:start -->
**Argument use:** Argues that high-stakes decisions should use inherently interpretable models rather than post-hoc explanations of black boxes. This is the central theoretical challenge to trading transparency for a small accuracy gain.
<!-- bilingual-en:end -->

**4.** [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/04_Buecker et al. (2022) - Transparency auditability explainability (author preprint).pdf|Bücker et al. (2022) · author preprint]] · [published DOI](https://doi.org/10.1080/01605682.2021.1922098)：直接针对信用评分提出 TAX4CS 框架，并比较逻辑回归、记分卡与复杂模型，说明复杂模型能否被采用取决于完整的透明、审计和局部解释流程，而非默认的性能优势。

<!-- bilingual-en:start -->
**Argument use:** Proposes the TAX4CS framework specifically for credit scoring and compares logistic regression and scorecards with complex models. It shows that adoption depends on a complete transparency, audit and local-explanation process rather than an assumed performance advantage.
<!-- bilingual-en:end -->

**5.** [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/05_Fuster et al. (2022) - Predictably unequal.pdf|Fuster et al. (2022)]] · [DOI](https://doi.org/10.1111/jofi.13090)：在近千万笔美国抵押贷款上比较逻辑回归与机器学习，发现整体预测更准并不意味着收益均匀分配，为公平审查和治理条件提供顶级金融期刊证据。

<!-- bilingual-en:start -->
**Argument use:** Compares logistic regression with machine learning on nearly ten million US mortgages. It shows that higher aggregate accuracy does not imply evenly distributed benefits, providing top-journal evidence for fairness review and governance conditions.
<!-- bilingual-en:end -->

**6.** [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/06_Bracke et al. (2019) - ML explainability in finance.pdf|Bracke et al. (2019)]] · [Bank of England](https://www.bankofengland.co.uk/working-paper/2019/machine-learning-explainability-in-finance-an-application-to-default-risk-analysis)：在真实抵押贷款违约模型上应用基于 Shapley value 的解释框架，同时指出不同贷款的解释差异与剩余模型不确定性，用来界定事后解释能解决什么、不能解决什么。

<!-- bilingual-en:start -->
**Argument use:** Applies a Shapley-value-based explanation framework to a real mortgage-default model while documenting variation across loans and residual model uncertainty. It helps define what post-hoc explanation can and cannot solve.
<!-- bilingual-en:end -->

## 作业要求

- 题目与未来学位课程相关。
- 正文约 2,000 字。
- 至少使用 6 篇学术文献。
- 包含带有明确 thesis statement 的引言、结构清楚的正文段落、结论和独立参考文献页。
- 选题阶段需在 Moodle 提交 essay question，以及该问题如何为未来学位学习做准备的简短说明。

## 选题方法

1. 从未来课程的 Moodle、往年试卷或课程说明中收集三个候选问题。
2. 检查每个问题是否能在 2,000 字内回答，以及是否容易找到至少 6 篇学术文献。
3. 选择最能为未来课程学习做准备的问题；必要时缩小范围或改写问题。
4. 若时间允许，在提交前把题目草稿给老师确认。

## 来源

- [[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-1/LN905 Week one materials 2026.pdf#page=3|Week 1 workbook · assignment and deadlines]]
- [[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-1/LN905 Week one materials 2026.pdf#page=19|Week 1 workbook · choosing an essay question]]
- [[07_Programme/01_LN905_LSE-language-class/PDF/02_Academic-Writing/Week-1/LN905 Week one materials 2026.pdf#page=20|Week 1 workbook · essay question proposal]]
