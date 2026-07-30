---
aliases:
- GARCH Model Estimation Steps
- GARCH模型估计步骤
- GARCH模型估计
tags:
- procedure
- 时间序列
- 波动建模
type: procedure
---
# GARCH Model Estimation Steps

## 这张卡什么时候用

当均值模型残差没有明显自相关，但残差平方表现出波动聚集或 ARCH 效应时，用这张卡估计 GARCH 模型。

## 输入

- 已拟合均值模型后的残差 $\hat\varepsilon_t$；
- 备选方差模型，如 GARCH(1,1)；
- 参数约束和分布假设。

## 输出

- 条件方差序列 $\hat h_t$；
- 参数估计 $\hat\omega,\hat\alpha,\hat\beta$；
- 标准化残差；
- 波动率预测。

## Step 1. 先确认均值模型残差

先检查残差本身是否还有自相关。

如果残差仍有明显自相关，先回去修 ARMA/均值模型。

如果残差本身较干净，但残差平方有自相关，再进入方差建模。

## Step 2. 做 ARCH 效应诊断

使用：

- [[ARCH LM Test]]；
- [[McLeod-Li Test]]；
- 残差平方 ACF。

若显著，说明条件方差结构值得建模。

## Step 3. 设定 GARCH(1,1)

常用起点：
$$
\hat\varepsilon_t=\nu_t\sqrt{h_t},
$$
$$
h_t=\omega+\alpha\hat\varepsilon_{t-1}^2+\beta h_{t-1}.
$$

约束：
$$
\omega>0,\qquad \alpha\geq0,\qquad \beta\geq0.
$$

若要求长期方差存在：
$$
\alpha+\beta<1.
$$

## Step 4. 用最大似然估计

在正态假设下，对数似然包含：
$$
\ln h_t+\frac{\hat\varepsilon_t^2}{h_t}.
$$

因为 $h_t$ 递推依赖过去，需要按时间顺序计算条件方差。

## Step 5. 检查估计结果

重点看：

- 参数是否满足约束；
- $\alpha+\beta$ 是否接近 1；
- 标准化残差是否仍有自相关；
- 标准化残差平方是否仍有 ARCH 效应。

## Step 6. 输出波动率预测

一步预测：
$$
\hat h_{T+1}
=\hat\omega+\hat\alpha\hat\varepsilon_T^2+\hat\beta\hat h_T.
$$

长期方差：
$$
\frac{\hat\omega}{1-\hat\alpha-\hat\beta}
$$
只在 $\hat\alpha+\hat\beta<1$ 时有意义。

## 常见错误

- 在均值模型没修好时直接估 GARCH。
- 参数违反非负或平稳约束却继续解释。
- 估计完不检查标准化残差。
- 把 GARCH 的方差预测当作收益方向预测。

## 来自课程位置

- [[04_波动建模 Modeling Volatility#2. ARCH,GARCH|时间序列 04：ARCH/GARCH]]
- [[04_波动建模 Modeling Volatility#2.4 侦测ARCH/GARCH效应|时间序列 04：诊断 ARCH 效应]]

## 关联卡片

- [[GARCH]]
- [[ARCH]]
- [[Conditional Heteroskedasticity]]
- [[Volatility Clustering]]
- [[ARCH LM Test]]
- [[McLeod-Li Test]]

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
