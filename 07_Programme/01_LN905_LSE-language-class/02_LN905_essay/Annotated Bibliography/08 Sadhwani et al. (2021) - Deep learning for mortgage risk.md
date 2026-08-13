---
title: Deep learning for mortgage risk
authors:
  - Apaar Sadhwani
  - Kay Giesecke
  - Justin Sirignano
year: 2021
doi: 10.1093/jjfinec/nbaa025
status: summary-draft
source_pdf: "[[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/08_Sadhwani et al. (2021) - Deep learning for mortgage risk (author preprint).pdf]]"
---

# Sadhwani et al. (2021)：超大样本下的抵押贷款深度学习

> [!cite] Reference
> Sadhwani, A., Giesecke, K., & Sirignano, J. (2021). Deep learning for mortgage risk. *Journal of Financial Econometrics, 19*(2), 313-368. https://doi.org/10.1093/jjfinec/nbaa025

> [!summary] 一句话梗概
> 在超过 1.2 亿笔美国抵押贷款和 35 亿条 loan-month observations 上，深度神经网络显著改善多期贷款状态预测，尤其能捕捉 prepayment 与宏观变量之间的强非线性；这是复杂度可能产生实质收益的最强正面案例之一。

## 成文结构：先证明线性误设，再证明复杂度有价值

> [!abstract] 作者先锁定的中心
> **现象**：抵押贷款研究常用线性 logit/Cox 模型，但借款人行为可能包含阈值、高阶交互和随经济状态变化的敏感度。<br>
> **张力**：深度模型能自动学习非线性，却必须证明这不只是更好的样本内拟合，而是贷款、贷款池和投资层面的真实增益。<br>
> **中心判断**：在超大规模、长时间跨度且关系高度非线性的抵押贷款数据中，深度模型显著降低线性误设，并产生多层次样本外价值。

### Section 路线图

| 原文部分 | 这一部分在全文中的任务 | 它为下一部分打开什么问题 |
|---|---|---|
| 1. Introduction | 对照传统线性文献，提出非线性、交互和多状态转移三个缺口，并预告经济含义。 | 原始数据是否真的呈现这些复杂关系？ |
| 2. The Data | 从贷款特征、地方/全国经济变量、状态转移到描述性非线性图，先用现象证明模型需求。 | 什么模型能够统一表示这些动态关系？ |
| 3. Deep Learning Model | 构建多期、多状态的非线性 transition model，把 logit 表述为零隐藏层基准。 | 如此巨大的模型怎样估计并控制过拟合？ |
| 4. Likelihood Estimation | 说明极大似然、GPU 计算、regularisation、dropout、ensemble，以及按时间划分训练/验证/测试。 | 拟合后发现哪些经济关系？ |
| 5. Empirical Results | 先排序变量解释力与经济显著性，再展示 prepayment/delinquency 的形状和变量交互。 | 这些复杂关系能否改善真正未见数据的预测？ |
| 6.1-6.2 Out-of-sample fit | 比较不同网络深度与零层 logit 的 likelihood、AUC 和各状态转移预测。 | 预测提升能否改变实际组合决策？ |
| 6.3 Investment portfolios | 用模型选择贷款组合，比较损失和投资表现。 | 个体优势能否聚合到贷款池？ |
| 6.4 Pool-level accuracy | 比较组合层 prepayment 分布和预测误差，连接 MBS 定价与对冲。 | 结论适用于多大范围？ |
| 7. Conclusion | 汇总非线性发现与多层样本外价值，同时把结论限定在该超大抵押贷款环境。 | — |

### 关键段落组怎样推进

1. **线性缺口段**：先指出既有模型必须手工指定变换与交互，变量很多时不可行。
2. **现象证据段**：在引入神经网络前展示 prepayment 与利率、FICO、loan age、LTV 的非线性图。
3. **模型桥接段**：把深度网络写成 logit 的非线性扩展，使新方法与熟悉基准可比较。
4. **可信估计段**：详细交代时间切分、dropout、regularisation 和 depth selection，预先回应过拟合质疑。
5. **解释结果段**：先回答变量如何影响行为，再回答模型是否更准；不是只报 leaderboard。
6. **样本外段**：从 likelihood 到 transition-specific AUC，逐层确认提升不是单一指标现象。
7. **决策价值段**：把预测放进投资组合和贷款池，证明误差下降具有业务后果。
8. **边界段**：超大样本既是证据强项，也是外推限制；普通信用评分未必复制该收益。

> [!tip] 可迁移到你的 essay
> 这是复杂模型一方的“最强案例”结构：**先证明线性模型具体错在哪里 → 证明复杂模型捕捉了什么 → 做真正时间外检验 → 把增益传到决策价值 → 限定适用环境**。只有完成这条链，复杂度才接近被 justify。

## 研究问题

论文研究线性 logistic formulation 是否会误设抵押贷款表现中复杂的非线性与交互关系，以及深度学习能否改善个体贷款、贷款池和投资组合层面的样本外预测。

## 方法与证据

- 数据覆盖 1995-2014 年超过 1.2 亿笔美国 prime 与 subprime mortgages，约占同期发放量的 70%。
- 清理后约 35 亿条月度观察、272 个贷款及宏观变量、7 种贷款状态。
- 训练集截至 2012 年 4 月，validation 为 2012 年 5-10 月，test 为 2012 年 11 月至 2014 年 5 月。
- 比较 0-layer logistic model 与多层 neural networks，并使用 regularisation、dropout 和 ensemble 控制过拟合。

## 主要发现

- 五层网络及其 ensemble 在样本外 likelihood 和多种状态转换 AUC 上优于线性逻辑模型。
- 非线性对 prepayment 尤其重要；失业率与 FICO、LTV、利率和房价存在明显交互。
- state unemployment 在所考察变量中具有最高解释力，说明住房金融与宏观经济联系比许多线性研究呈现得更强。
- 在作者设定的损失假设下，由五层网络选择的 20,000 笔贷款组合一年损失比线性模型组合低 46%；贷款池层面的 prepayment 预测误差也明显更小。

## 关键局限

研究对象是抵押贷款状态与 MBS 风险，不是典型的消费者贷款申请 scorecard。经济收益来自投资组合实验和假设损失率，并非真实部署后的利润。模型可解释性、公平性和监管成本没有直接测量；超大专有数据环境也难以代表一般银行。

## 对 LN905 essay 的用途

它说明当样本极大、关系高度非线性且线性误设会直接影响风险和投资决策时，复杂模型的收益可能足以达到 material threshold。使用时必须限定范围：这是一种“复杂度可能被证明合理”的边界案例，不足以推出普通信用评分也应默认使用深度学习。

## 原文

- [[07_Programme/01_LN905_LSE-language-class/PDF/10_Essay-Credit-Scoring/08_Sadhwani et al. (2021) - Deep learning for mortgage risk (author preprint).pdf|Author preprint]]
- [Published article](https://doi.org/10.1093/jjfinec/nbaa025)
