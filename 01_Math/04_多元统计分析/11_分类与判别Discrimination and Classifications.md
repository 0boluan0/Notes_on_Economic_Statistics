# 1. 第11章：分类与判别（Discrimination and Classification）
<!-- bilingual-en:start -->
*1. Chapter 11: Discrimination and Classification*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 判别关注“不同组有什么差异”，分类关注“新观测应该分到哪一组”。本章从两总体分类规则、误分类成本、Fisher 判别和分类效果评估展开。
> <!-- bilingual-en:start -->
> Discrimination asks how groups differ, while classification asks which group should receive a new observation. This chapter develops two-population classification rules, misclassification costs, Fisher discrimination, and the evaluation of classification performance.
> <!-- bilingual-en:end -->

## 1.1. 引言
<!-- bilingual-en:start -->
*1.1. Introduction*
<!-- bilingual-en:end -->

判别和分类是多变量技术，用于区分不同对象集合，并把新对象分配到已定义的组。
<!-- bilingual-en:start -->
Discrimination and classification are multivariate techniques for distinguishing sets of objects and assigning new objects to pre-defined groups.
<!-- bilingual-en:end -->

主要目标：
<!-- bilingual-en:start -->
The main objectives are to:
<!-- bilingual-en:end -->

1. 用图形或代数方法描述组间差异。
2. 将观测分为两个或多个标记类别。
<!-- bilingual-en:start -->
1. Describe differences among groups graphically or algebraically.
2. Classify observations into two or more labelled categories.
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

### 1.3.1. 三个输入
<!-- bilingual-en:start -->
*1.3.1. Three Inputs*
<!-- bilingual-en:end -->

1. 先验概率：$P_1,P_2$。
2. 误分类成本：$c(2|1),c(1|2)$。
3. 条件密度函数：$f_1(x),f_2(x)$。
<!-- bilingual-en:start -->
1. Prior probabilities: $P_1,P_2$.
2. Misclassification costs: $c(2|1),c(1|2)$.
3. Conditional density functions: $f_1(x),f_2(x)$.
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

## 1.4. Fisher 判别方法
<!-- bilingual-en:start -->
*1.4. Fisher's Discriminant Method*
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

若先验和成本相等，分类阈值为两组投影均值的中点：
<!-- bilingual-en:start -->
When priors and costs are equal, the classification threshold is the midpoint of the two projected means:
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

>[!note] 做题重点
> 核心不是背公式，而是先求 $\hat a$，再把新样本和两个组均值都投影到同一条线上。
> <!-- bilingual-en:start -->
> The key is not memorising the formula. First find $\hat a$, then project the new observation and both group means onto the same line.
> <!-- bilingual-en:end -->

## 1.5. 分类性能评估
<!-- bilingual-en:start -->
*1.5. Evaluating Classification Performance*
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

### 1.6.1. Jackknife 方法
<!-- bilingual-en:start -->
*1.6.1. Jackknife Method*
<!-- bilingual-en:end -->

1. 每次移除一个观测。
2. 用剩余样本构建分类器。
3. 分类被移除的观测。
4. 对所有观测重复并汇总错分次数。
<!-- bilingual-en:start -->
1. Remove one observation at a time.
2. Build the classifier from the remaining sample.
3. Classify the omitted observation.
4. Repeat for every observation and total the misclassifications.
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

## 1.8. 关联卡片
<!-- bilingual-en:start -->
*1.8. Related Cards*
<!-- bilingual-en:end -->

- [[判别分析：Bayes、LDA 与 QDA#Bayes 分类与判别规则|Classification Rule Selection]]
- [[判别分析：Bayes、LDA 与 QDA#Bayes 分类与判别规则|Expected Cost of Misclassification]]
- [[判别分析：Bayes、LDA 与 QDA#Bayes 分类与判别规则|Total Probability of Misclassification]]
- [[判别分析：Bayes、LDA 与 QDA#LDA、QDA 与 Fisher 判别|Fisher Linear Discriminant]]
- [[判别分析：Bayes、LDA 与 QDA#LDA、QDA 与 Fisher 判别|Fisher Discriminant Procedure]]
- [[判别分析：Bayes、LDA 与 QDA#分类评估|Actual Error Rate]]
- [[判别分析：Bayes、LDA 与 QDA#分类评估|Jackknife Classification]]
