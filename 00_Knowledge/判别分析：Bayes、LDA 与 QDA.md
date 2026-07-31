---
aliases:
  - "Discriminant Analysis"
  - "LDA and QDA"
  - "判别分析"
status: source-checked
---

# 判别分析：Bayes、LDA 与 QDA
<!-- bilingual-en:start -->
*Discriminant analysis: Bayes, LDA, and QDA*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 判别分析在已有标签时预测类别；聚类在没有标签时寻找依指定距离和算法形成的组。
> **具体锚点：** 已知历史客户是否违约时做分类；只有客户特征、想探索群体结构时做聚类。
> **核心难点：** 聚类结果不是数据里唯一“真实”的类别，强烈依赖尺度、距离、链接和簇数；分类评估必须在未参与训练的数据上进行。
> **为什么重要：** 先区分监督与无监督，能避免把探索性分组当作已验证预测。
> **继续：** 分类先读 Bayes/LDA/QDA；聚类先做标准化与距离选择，再比较层次法和 K-means。
> <!-- bilingual-en:start -->
> **What it solves:** Discriminant analysis predicts classes when labels are available; clustering searches for groups defined by a chosen distance and algorithm when labels are absent.
> **Concrete anchor:** With historical default labels, classify new customers. With only customer features and a desire to explore population structure, use clustering.
> **Central difficulty:** Clusters are not unique “true” classes hidden in the data; they depend strongly on scaling, distance, linkage, and the number of clusters. Classification must be evaluated on data not used for training.
> **Why it matters:** Distinguishing supervised from unsupervised learning prevents exploratory groupings from being presented as validated predictions.
> **Continue with:** For classification, begin with Bayes, LDA, and QDA. For unlabelled grouping, see [[聚类：层次聚类与 K-means|hierarchical clustering and K-means]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
> <!-- bilingual-en:start -->
> - [Penn State STAT 505](https://online.stat.psu.edu/stat505/) was used to verify definitions and conditions for multivariate normal models, mean inference, MANOVA, PCA, factor analysis, discriminant analysis, and clustering.
> - Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to verify matrix formulas and sampling distributions.
> <!-- bilingual-en:end -->

## Bayes 分类与判别规则
<!-- bilingual-en:start -->
*Bayes classification and decision rules*
<!-- bilingual-en:end -->

给定类别先验和类条件密度，Bayes 规则选择后验概率最大或期望损失最小的类别。错误成本不对称时，阈值应由损失矩阵而非固定 0.5 决定。
<!-- bilingual-en:start -->
Given class priors and class-conditional densities, the Bayes rule chooses either the class with the highest posterior probability or the action with the smallest expected loss. When errors have asymmetric costs, the threshold should come from the loss matrix rather than a fixed value of 0.5.
<!-- bilingual-en:end -->

对类别 k，后验概率正比于 $\pi_k f_k(x)$。比较对数判别得分可避免数值下溢，并把先验、中心距离和协方差惩罚清晰分开。修改先验或损失会改变最优分类，即使类条件模型不变。
<!-- bilingual-en:start -->
For class $k$, posterior probability is proportional to $\pi_k f_k(x)$. Comparing log-discriminant scores avoids numerical underflow and separates prior probability, distance from the class centre, and covariance penalties. Changing priors or losses changes the optimal classifier even when class-conditional models remain unchanged.
<!-- bilingual-en:end -->

## LDA、QDA 与 Fisher 判别
<!-- bilingual-en:start -->
*LDA, QDA, and Fisher discrimination*
<!-- bilingual-en:end -->

多元正态且各类共享协方差时得到线性判别边界 LDA；允许各类协方差不同得到二次边界 QDA，但参数更多。Fisher 判别寻找类间变异相对类内变异最大的投影，与两类 LDA 有密切联系但出发点不同。
<!-- bilingual-en:start -->
Multivariate normal class distributions with a shared covariance matrix yield linear decision boundaries in LDA. Allowing class-specific covariances gives quadratic boundaries in QDA but requires many more parameters. Fisher discrimination seeks a projection that maximises between-class variation relative to within-class variation; it is closely related to two-class LDA but begins from a different criterion.
<!-- bilingual-en:end -->

LDA 在每类样本有限时通过共享协方差降低方差；QDA 用更多自由度捕捉不同形状。若 $p$ 接近或超过每类样本量，样本协方差可能奇异，需正则化、降维或更简单模型，而不是强行求逆。
<!-- bilingual-en:start -->
LDA reduces estimation variance by pooling covariance across classes when class-specific samples are limited. QDA uses more degrees of freedom to capture different shapes. If $p$ approaches or exceeds the sample size within a class, sample covariance may be singular, requiring regularisation, dimension reduction, or a simpler model rather than forced inversion.
<!-- bilingual-en:end -->

## Worked example：两类同方差正态
<!-- bilingual-en:start -->
*Worked example: two normal classes with equal variance*
<!-- bilingual-en:end -->

一维两类满足 $X\mid G=0\sim N(0,1)$、$X\mid G=1\sim N(2,1)$，先验相等、误判成本相同。比较两个正态密度得边界 x=1：x>1 判为类别 1。若类别 1 的先验更低，边界会向 2 移动，需要更强证据才判为稀有类别。
<!-- bilingual-en:start -->
Suppose $X\mid G=0\sim N(0,1)$ and $X\mid G=1\sim N(2,1)$, with equal priors and equal misclassification costs. Comparing the two normal densities gives boundary $x=1$, so values above one are assigned to class 1. If class 1 has a smaller prior probability, the boundary moves toward 2, requiring stronger evidence before assigning the rare class.
<!-- bilingual-en:end -->

这个例子说明判别边界不是只由类均值决定。先验、损失与协方差共同进入决策；将训练样本中的类别比例机械当部署先验，会在抽样比例被人为控制时产生错误阈值。
<!-- bilingual-en:start -->
The example shows that class means alone do not determine the boundary. Priors, losses, and covariance all enter the decision. Mechanically treating the training class proportion as the deployment prior gives the wrong threshold when sampling proportions were controlled by design.
<!-- bilingual-en:end -->

## 分类评估
<!-- bilingual-en:start -->
*Evaluating classification*
<!-- bilingual-en:end -->

训练误差偏乐观，应使用独立测试集或交叉验证。混淆矩阵、灵敏度、特异度、precision、recall 和概率校准回答不同问题；类别不平衡时准确率尤其可能误导。
<!-- bilingual-en:start -->
Training error is optimistic, so use an independent test set or cross-validation. A confusion matrix, sensitivity, specificity, precision, recall, and probability calibration answer different questions. Accuracy is especially misleading under class imbalance.
<!-- bilingual-en:end -->

任何标准化、变量选择、缺失值填补和协方差正则化都必须在每个训练折内完成。若先用全数据选变量再交叉验证，测试折信息已经泄漏，误差估计仍偏乐观。
<!-- bilingual-en:start -->
Standardisation, variable selection, missing-value imputation, and covariance regularisation must all be performed inside each training fold. Selecting variables on the full dataset before cross-validation leaks information from the validation folds and leaves the error estimate optimistic.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure diagnostics*
<!-- bilingual-en:end -->

检查标签质量、类先验漂移、异常点、多元正态与协方差假设、概率校准和不同群体上的误差。LDA/QDA 边界解释依赖模型；若目标纯预测，应与合理基准比较，而不因方法经典就默认有效。
<!-- bilingual-en:start -->
Check label quality, prior shift, outliers, multivariate-normal and covariance assumptions, probability calibration, and errors across subgroups. Interpretation of LDA and QDA boundaries depends on the model. For a predictive goal, compare them with sensible baselines rather than assuming validity because the methods are classical.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 为什么有标签时通常不应把聚类当分类替代品？
<!-- bilingual-en:start -->
*Why should clustering generally not replace classification when labels are available?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 聚类优化的是内部距离结构，不利用标签或预测损失，得到的簇未必对应目标类别。
> <!-- bilingual-en:start -->
> Clustering optimises internal distance structure without using labels or predictive loss, so its groups need not correspond to the target classes.
> <!-- bilingual-en:end -->

### LDA 和 QDA 的关键假设差别是什么？
<!-- bilingual-en:start -->
*What is the key assumption distinguishing LDA from QDA?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> LDA 假设各类协方差相同从而边界线性；QDA 允许不同协方差，边界更灵活但估计参数更多。
> <!-- bilingual-en:start -->
> LDA assumes equal covariance matrices across classes and therefore has linear boundaries. QDA allows different covariances, making boundaries more flexible but increasing the number of estimated parameters.
> <!-- bilingual-en:end -->

### 类别 1 很稀有时，为什么 accuracy 可能很高却没有用？
<!-- bilingual-en:start -->
*When class 1 is rare, why can accuracy be high yet useless?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 永远预测多数类就可能有很高准确率，却把所有稀有目标漏掉；应结合 recall、precision、损失和校准评估。
> <!-- bilingual-en:start -->
> Always predicting the majority class can achieve high accuracy while missing every rare target. Recall, precision, decision loss, and calibration must also be assessed.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 505](https://online.stat.psu.edu/stat505/)：核验多元正态、均值推断、MANOVA、PCA、因子分析、判别与聚类的定义和使用条件。
- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验矩阵公式与抽样分布。
<!-- bilingual-en:start -->
- [Penn State STAT 505](https://online.stat.psu.edu/stat505/) was used to verify the definitions and conditions of the multivariate methods in this course.
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was used to verify matrix formulas and sampling distributions.
<!-- bilingual-en:end -->
- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10)：逐项核验 Bayes 分类、LDA/QDA 假设、判别函数与误分类评估。
<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 10](https://online.stat.psu.edu/stat505/Lesson10) was checked section by section for Bayes classification, LDA and QDA assumptions, discriminant functions, and misclassification assessment.
<!-- bilingual-en:end -->
