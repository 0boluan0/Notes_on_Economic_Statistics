---
aliases:
  - 相关系数
  - Correlation Coefficient
  - Pearson correlation
tags:
  - procedure
  - statistics
type: procedure
---

# Correlation coefficient

## 什么时候用

用于从成对观测 $(x_i,y_i)$ 计算 Pearson 样本相关系数，描述两个变量的线性关联强度与方向。

## 输入

- $n$ 对同时期观测 $(x_i,y_i)$。
- 需要明确是否使用样本协方差（分母 $n-1$）；相关系数中该口径会相互抵消，但标准差和协方差的单独报告必须一致。

## 输出

样本相关系数

\[
r_{xy}=\frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar x)^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar y)^2}}\in[-1,1].
\]

## Step 1：检查数据配对

删除或说明缺失值处理；确保每个 $x_i$ 与同一时点的 $y_i$ 配对，不能分别排序。

## Step 2：计算样本均值

\[
\bar x=\frac1n\sum_i x_i,\qquad \bar y=\frac1n\sum_i y_i.
\]

## Step 3：中心化并计算三项和

\[
S_{xy}=\sum_i(x_i-\bar x)(y_i-\bar y),\quad
S_{xx}=\sum_i(x_i-\bar x)^2,\quad
S_{yy}=\sum_i(y_i-\bar y)^2.
\]

## Step 4：代入公式并解释

计算 $r_{xy}=S_{xy}/\sqrt{S_{xx}S_{yy}}$。正负表示线性方向，绝对值越接近 1 表示线性关联越强；$r=0$ 只表示无线性相关，不等于独立。

## 检查点

- $S_{xx}=0$ 或 $S_{yy}=0$ 时相关系数未定义。
- 散点图应与符号和强弱大致一致；异常值可能显著改变结果。
- 相关不等于因果，非线性关系也可能有 $r\approx0$。

## 关联卡片

- [[Choosing Covariance vs Correlation Matrix]]
- [[Multivariate Normality Check]]

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
