# 1. 第11章：分类与判别（Discrimination and Classification）
<!-- bilingual-en:start -->
*1. Chapter 11: Discrimination and Classification*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 判别关注“不同组有什么差异”，分类关注“新观测应该分到哪一组”。本章从两总体分类规则、误分类成本、Fisher 判别和分类效果评估展开。
> <!-- bilingual-en:start -->
> Discrimination asks how groups differ, while classification asks which group should receive a new observation. This chapter develops two-population classification rules, misclassification costs, Fisher discrimination, and the evaluation of classification performance.
> <!-- bilingual-en:end -->

![[判别分析.canvas]]

## 1.1. 引言
<!-- bilingual-en:start -->
*1.1. Introduction*
<!-- bilingual-en:end -->

判别和分类是多变量技术，用于区分不同对象集合，并把新对象分配到已定义的组。
<!-- bilingual-en:start -->
Discrimination and classification are multivariate techniques for distinguishing sets of objects and assigning new objects to pre-defined groups.
<!-- bilingual-en:end -->

这里讨论的是“类别已经由标签定义”的监督问题；若没有标签、只是按距离探索分组，应转到聚类。两类任务的完整边界见 [[判别分类与聚类]]。
<!-- bilingual-en:start -->
This chapter concerns supervised problems whose classes are defined by labels. If labels are absent and the aim is to explore groups through distances, the task is clustering; see [[判别分类与聚类|classification versus clustering]] for the full boundary.
<!-- bilingual-en:end -->

主要目标：
<!-- bilingual-en:start -->
The main objectives are to:
<!-- bilingual-en:end -->

1. 用图形或代数方法描述组间差异。
2. 将观测分为两个或多个标记类别。
<!-- bilingual-en:start -->

&nbsp;
**1.** Describe differences among groups graphically or algebraically.<br>
**2.** Classify observations into two or more labelled categories.<br>
<!-- bilingual-en:end -->

术语区分：
<!-- bilingual-en:start -->
The terms differ as follows:
<!-- bilingual-en:end -->

- 判别：描述差异特征。
- 分类：把对象分配到类别中。
<!-- bilingual-en:start -->
- Discrimination describes the characteristics that distinguish groups.
- Classification assigns objects to categories.
<!-- bilingual-en:end -->

## 1.2. 两个总体的区分与分类
<!-- bilingual-en:start -->
*1.2. Distinguishing and Classifying Two Populations*
<!-- bilingual-en:end -->

设两个总体为 $\pi_1$ 和 $\pi_2$，观测向量为 $X$。
<!-- bilingual-en:start -->
Let the two populations be $\pi_1$ and $\pi_2$, and let the observation vector be $X$.
<!-- bilingual-en:end -->

典型例子：
<!-- bilingual-en:start -->
Typical examples include:
<!-- bilingual-en:end -->

| 类别问题 | 测量变量 |
|---|---|
| 偿付能力正常 vs 财务困境保险公司 | 总资产、股票和债券成本、市场价值、保费支出 |
| 新产品购买者 vs 滞后购买者 | 教育水平、收入、家庭规模、品牌切换次数 |
| 成功毕业 vs 未毕业学生 | 入学成绩、高中均分、活动数量 |
| 良好信用 vs 较差信用 | 收入、年龄、信用卡数量、家庭规模 |
<!-- bilingual-en:start -->
| Classification problem | Measured variables |
|---|---|
| Solvent versus financially distressed insurers | Total assets, costs of equities and bonds, market value, and premium expenditure |
| Early buyers versus late buyers of a new product | Education, income, household size, and number of brand switches |
| Students who graduate successfully versus those who do not | Admission score, secondary-school average, and number of activities |
| Good versus poor credit | Income, age, number of credit cards, and household size |
<!-- bilingual-en:end -->

## 1.3. 判别规则设定
<!-- bilingual-en:start -->
*1.3. Specifying a Classification Rule*
<!-- bilingual-en:end -->

这一节的统一原则是 [[Bayes分类规则|最小化后验期望损失]]；[[先验与分类风险]]说明先验、误判成本和评价口径怎样共同改变最优决定。
<!-- bilingual-en:start -->
The unifying principle is to [[Bayes分类规则|minimise posterior expected loss]]. [[先验与分类风险|Priors and classification risk]] explains how priors, error costs, and the evaluation target jointly determine the optimal decision.
<!-- bilingual-en:end -->

### 1.3.1. 三个输入
<!-- bilingual-en:start -->
*1.3.1. Three Inputs*
<!-- bilingual-en:end -->

1. 先验概率：$P_1,P_2$。
2. 误分类成本：$c(2|1),c(1|2)$。
3. 条件密度函数：$f_1(x),f_2(x)$。
<!-- bilingual-en:start -->

&nbsp;
**1.** Prior probabilities: $P_1,P_2$.<br>
**2.** Misclassification costs: $c(2|1),c(1|2)$.<br>
**3.** Conditional density functions: $f_1(x),f_2(x)$.<br>
<!-- bilingual-en:end -->

### 1.3.2. 分类区域
<!-- bilingual-en:start -->
*1.3.2. Classification Regions*
<!-- bilingual-en:end -->

定义：
<!-- bilingual-en:start -->
Define:
<!-- bilingual-en:end -->

- $R_1$：分配到 $\pi_1$ 的区域；
- $R_2$：分配到 $\pi_2$ 的区域；
- $R_2=\Omega-R_1$。
<!-- bilingual-en:start -->
- $R_1$: the region assigned to $\pi_1$;
- $R_2$: the region assigned to $\pi_2$;
- $R_2=\Omega-R_1$.
<!-- bilingual-en:end -->

正确分类概率：
<!-- bilingual-en:start -->
The probabilities of correct classification are:
<!-- bilingual-en:end -->
$$
P(1|1)=\int_{R_1}f_1(x)\,dx,
\qquad
P(2|2)=\int_{R_2}f_2(x)\,dx.
$$

错误分类概率：
<!-- bilingual-en:start -->
The probabilities of misclassification are:
<!-- bilingual-en:end -->
$$
P(2|1)=\int_{R_2}f_1(x)\,dx,
\qquad
P(1|2)=\int_{R_1}f_2(x)\,dx.
$$

### 1.3.3. 期望误分类成本（ECM）
<!-- bilingual-en:start -->
*1.3.3. Expected Cost of Misclassification (ECM)*
<!-- bilingual-en:end -->

两类情形下：
<!-- bilingual-en:start -->
For two classes:
<!-- bilingual-en:end -->
$$
ECM=P_1P(2|1)c(2|1)+P_2P(1|2)c(1|2).
$$

最优分类规则是最小化 ECM。
<!-- bilingual-en:start -->
The optimal classification rule minimises ECM.
<!-- bilingual-en:end -->

把样本分到 $R_1$ 的规则为
<!-- bilingual-en:start -->
Assign an observation to $R_1$ when
<!-- bilingual-en:end -->
$$
\frac{f_1(x)}{f_2(x)}
\geq
\frac{c(1|2)}{c(2|1)}\frac{P_2}{P_1}.
$$

否则分到 $R_2$。
<!-- bilingual-en:start -->
and otherwise assign it to $R_2$.
<!-- bilingual-en:end -->

>[!attention] 误链修正
> 这里的 ECM 是 Expected Cost of Misclassification，不是计量经济学里的 Error Correction Model。
> <!-- bilingual-en:start -->
> Here ECM means Expected Cost of Misclassification, not the Error Correction Model used in econometrics.
> <!-- bilingual-en:end -->

### 1.3.4. 特殊情况
<!-- bilingual-en:start -->
*1.3.4. Special Cases*
<!-- bilingual-en:end -->

| 条件 | 分类规则 |
|---|---|
| $P_1=P_2$ | 比较密度比和成本比 |
| $c(1|2)=c(2|1)$ | 比较密度比和先验概率比 |
| 先验和成本都相等 | 若 $f_1(x)\geq f_2(x)$，分到 $\pi_1$ |
<!-- bilingual-en:start -->
| Condition | Classification rule |
|---|---|
| $P_1=P_2$ | Compare the density ratio with the cost ratio |
| $c(1|2)=c(2|1)$ | Compare the density ratio with the prior-probability ratio |
| Priors and costs are both equal | Assign to $\pi_1$ when $f_1(x)\geq f_2(x)$ |
<!-- bilingual-en:end -->

### 1.3.5. Worked example：两类同方差正态
<!-- bilingual-en:start -->
*1.3.5. Worked example: two normal classes with equal variance*
<!-- bilingual-en:end -->

设 $X\mid G=0\sim N(0,1)$、$X\mid G=1\sim N(2,1)$，两种误判成本相同。若两类先验也相等，比较两个密度可得边界 $x^*=1$：$x>1$ 时判为类别 1。
<!-- bilingual-en:start -->
Suppose $X\mid G=0\sim N(0,1)$ and $X\mid G=1\sim N(2,1)$, with equal costs for the two errors. Equal priors give the boundary $x^*=1$, so an observation above one is assigned to class 1.
<!-- bilingual-en:end -->

若先验分别为 $\pi_0$ 与 $\pi_1$，成本仍相等，则边界变为

$$
x^*=1+\frac12\log\frac{\pi_0}{\pi_1}.
$$

因此类别 1 越稀有，边界越向右移；它不只会“靠近 2”，也可能超过 2。这个例子把 [[Gaussian判别得分]] 中的中心距离与先验项具体化，也说明部署先验不能由人为配平的训练样本比例机械代替。
<!-- bilingual-en:start -->
With priors $\pi_0$ and $\pi_1$ and the same equal costs, the boundary becomes $x^*=1+\tfrac12\log(\pi_0/\pi_1)$. A rarer class 1 therefore moves the boundary to the right; it may pass 2 rather than merely moving “towards” it. This makes the distance and prior terms in the [[Gaussian判别得分|Gaussian discriminant score]] concrete and shows why an artificially balanced training proportion need not be the deployment prior.
<!-- bilingual-en:end -->

若部署变化确实只有类别先验改变，而每一类内部的 $P(X\mid Y)$ 保持不变，源 posterior 才能按先验比重新加权；公式与失败边界见 [[先验漂移后验修正]]。
<!-- bilingual-en:start -->
Only when deployment changes class priors while leaving every $P(X\mid Y)$ unchanged can source posteriors be reweighted by prior ratios; see [[先验漂移后验修正|prior-shift posterior correction]] for the formula and its failure boundary.
<!-- bilingual-en:end -->

## 1.4. Fisher 判别方法
<!-- bilingual-en:start -->
*1.4. Fisher's Discriminant Method*
<!-- bilingual-en:end -->

[[Fisher判别]]从“类间分离相对类内变异最大”的投影准则出发。两类共享协方差时，它为何与 LDA 给出成比例的方向、又为何不自动得到同一个完整分类规则，见 [[Fisher与LDA]]。
<!-- bilingual-en:start -->
[[Fisher判别|Fisher discrimination]] starts from a projection criterion that maximises between-class separation relative to within-class variation. See [[Fisher与LDA|Fisher versus LDA]] for why the two-class shared-covariance directions are proportional without defining the same complete classification rule.
<!-- bilingual-en:end -->

Fisher 判别通过线性变换把多变量 $X$ 转为单变量
<!-- bilingual-en:start -->
Fisher discrimination transforms the multivariate $X$ into the univariate quantity
<!-- bilingual-en:end -->
$$
Y=a'X,
$$
使两类投影均值尽量分开。
<!-- bilingual-en:start -->
so that the projected class means are separated as much as possible.
<!-- bilingual-en:end -->

常用判别向量：
<!-- bilingual-en:start -->
A commonly used discriminant vector is
<!-- bilingual-en:end -->
$$
\hat a=S_{\text{pooled}}^{-1}(\bar x_1-\bar x_2).
$$

在两类 class-conditional Gaussian 且共享同一协方差的 LDA 模型下，若先验和两种误判成本也相等，相应 Bayes 分类阈值是两组 Fisher 投影均值的中点：
<!-- bilingual-en:start -->
Under a two-class LDA model with Gaussian class-conditional distributions and one shared covariance, equal priors and equal error costs make the corresponding Bayes threshold the midpoint of the two Fisher-projected means:
<!-- bilingual-en:end -->
$$
c=\frac12\hat a'(\bar x_1+\bar x_2).
$$

分类规则：
<!-- bilingual-en:start -->
The classification rule is:
<!-- bilingual-en:end -->
$$
\hat a'x_0\geq c
$$
则分到 $\pi_1$，否则分到 $\pi_2$。
<!-- bilingual-en:start -->
assign to $\pi_1$ when the inequality holds, and otherwise to $\pi_2$.
<!-- bilingual-en:end -->

若只采用 Fisher 分离准则而不承担上述 Gaussian 共享协方差模型，中点只能是另加的分类约定，不能由相等先验和成本单独推出。
<!-- bilingual-en:start -->
If only Fisher's separation criterion is adopted without the shared-covariance Gaussian model, the midpoint is an additional classification convention rather than a consequence of equal priors and costs alone.
<!-- bilingual-en:end -->

样本规则中的 $S_{\mathrm{pooled}}$ 不是全体观测围绕总均值的协方差。[[判别协方差估计]]给出 LDA 合并组内估计与 QDA 逐类估计的定义；[[判别协方差秩边界]]则单独回答这些矩阵何时因维数和样本量必然不可逆。
<!-- bilingual-en:start -->
The $S_{\mathrm{pooled}}$ in the sample rule is not the covariance of all observations around the grand mean. [[判别协方差估计|Discriminant covariance estimation]] defines the pooled-within and classwise estimators, while [[判别协方差秩边界|their rank boundary]] states when dimension and sample size force singularity.
<!-- bilingual-en:end -->

>[!note] 做题重点
> 核心不是背公式，而是先求 $\hat a$，再把新样本和两个组均值都投影到同一条线上。
> <!-- bilingual-en:start -->
> The key is not memorising the formula. First find $\hat a$, then project the new observation and both group means onto the same line.
> <!-- bilingual-en:end -->

## 1.5. 分类性能评估
<!-- bilingual-en:start -->
*1.5. Evaluating Classification Performance*
<!-- bilingual-en:end -->

TPM 与 ECM 的区别见 [[先验与分类风险]]；混淆矩阵中 sensitivity、specificity、precision 与 accuracy 的分母和用途见 [[分类评估口径]]。训练内误差为何不能当作泛化证据，见 [[分类验证与泄漏]]。
<!-- bilingual-en:start -->
See [[先验与分类风险|priors and classification risk]] for the distinction between TPM and ECM, [[分类评估口径|classification metrics]] for the denominators and uses of sensitivity, specificity, precision, and accuracy, and [[分类验证与泄漏|validation and leakage]] for why training error is not evidence of generalisation.
<!-- bilingual-en:end -->

若模型输出概率，还要另问这些数值能否解释为目标总体中的事件频率；这属于 [[分类概率校准]]，不是一张混淆矩阵能够回答的问题。
<!-- bilingual-en:start -->
If the model reports probabilities, their interpretation as event frequencies in the target population requires a separate [[分类概率校准|calibration]] check; a confusion matrix cannot answer that question.
<!-- bilingual-en:end -->

### 1.5.1. 误分类总概率（TPM）
<!-- bilingual-en:start -->
*1.5.1. Total Probability of Misclassification (TPM)*
<!-- bilingual-en:end -->

$$
TPM=P_1P(2|1)+P_2P(1|2).
$$

等价积分形式为
<!-- bilingual-en:start -->
The equivalent integral expression is
<!-- bilingual-en:end -->
$$
TPM=P_1\int_{R_2}f_1(x)\,dx+
P_2\int_{R_1}f_2(x)\,dx.
$$

### 1.5.2. 实际误差率（AER）
<!-- bilingual-en:start -->
*1.5.2. Actual Error Rate (AER)*
<!-- bilingual-en:end -->

AER 用样本分类结果估计真实错误率。训练集 AER 可能偏乐观，因此常配合交叉验证。
<!-- bilingual-en:start -->
AER estimates the true error rate from sample classifications. Training-set AER can be optimistic, so it is commonly paired with cross-validation.
<!-- bilingual-en:end -->

## 1.6. 交叉验证方法
<!-- bilingual-en:start -->
*1.6. Cross-Validation Methods*
<!-- bilingual-en:end -->

[[分类验证与泄漏]]把下面的逐一留出算法扩展到完整 pipeline：插补、标准化、变量选择、降维、正则化与调参都必须在每次留下观测后重新拟合。
<!-- bilingual-en:start -->
[[分类验证与泄漏|Validation and leakage]] extends the leave-one-out algorithm below to the whole pipeline: imputation, scaling, feature selection, dimension reduction, regularisation, and tuning must all be refitted after the observation is held out.
<!-- bilingual-en:end -->

### 1.6.1. Jackknife 方法
<!-- bilingual-en:start -->
*1.6.1. Jackknife Method*
<!-- bilingual-en:end -->

1. 每次移除一个观测。
2. 用剩余样本构建分类器。
3. 分类被移除的观测。
4. 对所有观测重复并汇总错分次数。
<!-- bilingual-en:start -->

&nbsp;
**1.** Remove one observation at a time.<br>
**2.** Build the classifier from the remaining sample.<br>
**3.** Classify the omitted observation.<br>
**4.** Repeat for every observation and total the misclassifications.<br>
<!-- bilingual-en:end -->

条件误分类概率可估计为
<!-- bilingual-en:start -->
The conditional misclassification probabilities can be estimated by
<!-- bilingual-en:end -->
$$
\hat P(2|1)=\frac{n_{1m}(H)}{n_1},
\qquad
\hat P(1|2)=\frac{n_{2m}(H)}{n_2}.
$$

平均错误率估计为
<!-- bilingual-en:start -->
The estimated average error rate is
<!-- bilingual-en:end -->
$$
\widehat{AER}=
\frac{n_{1m}(H)+n_{2m}(H)}{n_1+n_2}.
$$

## 1.7. 多个总体的分类
<!-- bilingual-en:start -->
*1.7. Classification with Several Populations*
<!-- bilingual-en:end -->

多类情形仍遵循 [[Bayes分类规则]]：对每个可选行动计算后验期望损失，再选择风险最小者；不能把“选最大 posterior”从 0–1 loss 情形无条件外推。
<!-- bilingual-en:start -->
The multiclass case still follows the [[Bayes分类规则|Bayes decision rule]]: compute posterior expected loss for each available action and choose the smallest. Selecting the largest posterior is the special case for 0–1 loss, not a universal rule.
<!-- bilingual-en:end -->

对 $g$ 个总体，若真实属于 $\pi_i$，条件期望误分类成本为
<!-- bilingual-en:start -->
With $g$ populations, if the true population is $\pi_i$, the conditional expected misclassification cost is
<!-- bilingual-en:end -->
$$
ECM(i)=\sum_{k\neq i}P(k|i)c(k|i).
$$

总体期望误分类成本为
<!-- bilingual-en:start -->
The overall expected misclassification cost is
<!-- bilingual-en:end -->
$$
ECM=\sum_{i=1}^g p_iECM(i).
$$

## 1.8. 知识地图与延伸
<!-- bilingual-en:start -->
*1.8. Knowledge Map and Extensions*
<!-- bilingual-en:end -->

- **任务边界：** [[判别分类与聚类]]
- **决定原则：** [[Bayes分类规则]]、[[先验与分类风险]]
- **Gaussian 判别族：** [[Gaussian判别得分]]、[[LDA共享协方差]]、[[QDA类别协方差]]、[[Fisher判别]]、[[Fisher与LDA]]
- **估计与稳定性：** [[判别协方差估计]]、[[判别协方差秩边界]]、[[正则化判别]]
- **评价与部署：** [[分类评估口径]]、[[分类验证与泄漏]]、[[分类概率校准]]、[[先验漂移后验修正]]、[[判别模型诊断]]
