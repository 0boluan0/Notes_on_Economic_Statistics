---
aliases:
- Econometrics-hub
- Econometrics
- 计量经济学
- 计量经济学知识地图
tags:
- hub
- econometrics
---
# Econometrics-hub

## 这组卡解决什么

计量经济学这组卡按“建模、估计、推断、诊断、识别”组织，而不是把所有检验堆在一起。复习时先看 OLS 主线，再看假设被破坏时该怎么诊断和修正。

## 学习路线

1. 回归模型入口：[[Linear Regression Model]]、[[Simple Linear Regression]]、[[Multiple Linear Regression]]、[[OLS Basics]]、[[OLS Estimator]]、[[Residual]]。
2. OLS 操作与证明：[[OLS Estimation Steps]]、[[OLS Normal Equations]]、[[OLS unbiasedness]]、[[OLS consistency]]、[[Gauss-Markov theorem]]。
3. 推断工具：[[P-value]]、[[Confidence Interval]]、[[t Test]]、[[F-test]]、[[Wald Test]]、[[Lagrange Multiplier Test]]、[[R-squared]]、[[AIC]]、[[BIC]]。
4. 模型设定：[[Model Misspecification]]、[[Omitted Variable Bias]]、[[Measurement Error]]、[[Ramsey RESET Test]]、[[Davidson-MacKinnon J Test]]。
5. 假设诊断：[[Heteroskedasticity]]、[[Autocorrelation]]、[[Multicollinearity]]、[[Endogeneity]]。
6. 修正方法：[[White Robust Standard Errors]]、[[Newey-West]]、[[Weighted Least Squares]]、[[Generalized Least Squares]]、[[FGLS]]、[[Instrumental Variable]]、[[2SLS]]。
7. 面板与准实验：[[Panel Data Model]]、[[Fixed Effects Model]]、[[Random Effects Model]]、[[LSDV Estimation]]、[[Dynamic Panel Data Model]]、[[DID-hub]]。

## OLS 与推断

- [[OLS Basics]]：OLS 是什么。
- [[OLS Estimation Steps]]：怎样从数据算出回归结果。
- [[OLS unbiasedness]]、[[OLS consistency]]、[[Gauss-Markov theorem]]：为什么在假设下成立。
- [[t Test]]：单个系数检验。
- [[F-test]]：联合限制或整体显著性检验。
- [[Wald Test]]、[[Lagrange Multiplier Test]]、[[Likelihood Ratio Test]]：三类常见约束检验。
- [[R-squared]]：拟合优度，不等于因果可信度。

## 回归诊断

- 异方差：[[Heteroscedasticity Diagnosis]]、[[White Test]]、[[White Robust Standard Errors]]、[[Weighted Least Squares]]。
- 自相关：[[Autocorrelation Diagnosis]]、[[Durbin-Watson Statistic]]、[[Breusch-Godfrey Test]]、[[Newey-West]]、[[Cochrane-Orcutt]]。
- 多重共线性：[[Multicollinearity]]、[[Variance Inflation Factor]]、[[Condition Index]]。
- 异常值和影响点：[[Outlier Detection]]、[[Cook's Distance]]。
- 模型设定：[[Omitted Variable Bias]]、[[Measurement Error]]、[[Ramsey RESET Test]]、[[Chow Test]]、[[Jarque-Bera Test]]。

## 识别与内生性

- [[Endogeneity]]：解释变量与误差项相关。
- [[Endogeneity Diagnosis]]：诊断来源和处理路径。
- [[Instrumental Variable]] 与 [[2SLS]]：工具变量估计。
- [[Hausman Test]]：比较估计量差异。
- [[Hansen J Test]]：GMM/IV 过度识别检验。
- [[Simultaneity Bias]]、[[Simultaneous Equations Model]]、[[Parameter Identification]]：联立方程与识别。

## 离散选择与面板

- [[Linear Probability Model]]
- [[Logit Model]]
- [[Probit Model]]
- [[Discrete Choice Model]]
- [[Panel Data Model]]
- [[Fixed Effects Model]]
- [[Random Effects Model]]
- [[LSDV Estimation]]
- [[Dynamic Panel Data Model]]

## 课程例子入口

- [[Marginal Propensity to Consume]]
- [[Cobb-Douglas Production Function]]
- [[Environmental Kuznets Curve]]

## 来自课程位置

- [[01_导论]]
- [[02_一元线性回归]]
- [[03_多元线性回归]]
- [[04_模型设定]]
- [[05_多元回归模型的矩阵表达]]
- [[06_多重共线性]]
- [[07_异方差]]
- [[08_自相关]]
- [[09_联立方程模型(内生性)]]
- [[10_虚拟变量]]
- [[11_平稳时间序列模型]]
- [[12_非平稳时间序列]]
- [[13_面板数据模型]]
- [[零散知识点]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
