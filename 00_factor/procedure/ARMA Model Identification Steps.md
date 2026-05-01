---
aliases:
- ARMA Model Identification Steps
- ARMA模型识别步骤
- ARMA模型定阶步骤
- ARMA order identification
tags:
- procedure
- 时间序列
---
# ARMA Model Identification Steps

## 这张卡什么时候用

当你有一条看起来已经平稳的单变量时间序列，要决定用 AR、MA 还是 ARMA，以及阶数 $p,q$ 时，用这张卡。

## 输入

- 序列 $y_t$；
- 平稳性检验结果；
- ACF/PACF 图；
- 候选模型的 AIC/BIC；
- 残差诊断结果。

## 输出

- 一个候选 ARMA$(p,q)$ 模型；
- 或者明确结论：序列不平稳，需要先差分/去趋势/做协整分析。

## Step 1. 先确认序列平稳

先看图，再用 [[Unit Root Test]]。

如果序列非平稳：

- 单变量预测：转 [[ARIMA]]；
- 多变量长期关系：转 [[Cointegration]]；
- 不要直接拿水平序列做 ARMA。

## Step 2. 看 ACF/PACF 做初筛

| 图像 | 候选模型 |
| --- | --- |
| ACF 拖尾，PACF 截尾 | AR(p) |
| ACF 截尾，PACF 拖尾 | MA(q) |
| ACF 拖尾，PACF 拖尾 | ARMA(p,q) |

这里的“截尾”是理论图像。样本中要结合显著性界限，不要机械读图。

## Step 3. 估计几个候选模型

不要只估一个模型。

根据 ACF/PACF 提出一组小范围候选，比如：

- AR(1), AR(2)；
- MA(1), MA(2)；
- ARMA(1,1)。

然后分别估计。

## Step 4. 用信息准则比较

比较 [[AIC]] / [[BIC]]。

- AIC 通常更愿意保留复杂模型；
- BIC 惩罚更重，更偏向简洁模型。

如果两个模型表现接近，优先选更容易解释、残差更干净的模型。

## Step 5. 做残差诊断

对残差做：

- [[White Noise Test]]；
- [[Ljung-Box Test]]；
- 残差 ACF；
- 必要时看残差平方的 ARCH 效应。

如果残差仍有自相关，回到 Step 2 重新定阶。

如果残差本身白噪声，但残差平方有结构，均值模型可能够了，方差模型要转 [[GARCH]]。

## 常见错误

- 忘记先检验平稳性。
- 只靠 ACF/PACF 定阶，不做信息准则和残差诊断。
- 过度拟合：为了降低残差而加很多阶。
- 把 ARMA 当成波动率模型；它主要描述条件均值。

## 来自课程位置

- [[03_平稳时间序列模型#0.回忆用|时间序列 03：ARMA 识别回忆]]
- [[03_平稳时间序列模型#3. ACF|时间序列 03：ACF/PACF 与定阶]]

## 关联卡片

- [[ARMA]]
- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[Box-Jenkins Method]]
- [[White Noise Test]]
- [[GARCH Model Estimation Steps]]

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
