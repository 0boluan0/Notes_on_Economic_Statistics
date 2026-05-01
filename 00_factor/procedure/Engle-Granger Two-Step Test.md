---
aliases:
- Engle-Granger Two-Step Test
- Engle-Granger Cointegration Test
- EG two-step test
- EG两步法
- EG两步检验法
tags:
- procedure
- 时间序列
- 计量经济学
---
# Engle-Granger Two-Step Test

## 这张卡什么时候用

当你怀疑两个或少数几个 $I(1)$ 变量存在长期均衡关系，并想用残差平稳性检验协整时，用 EG 两步法。

## 输入

- 两个或多个同阶单整变量，通常是 $I(1)$；
- 一个有经济含义的长期关系设定；
- 单位根检验工具。

## 输出

- 是否存在协整关系；
- 一个长期关系残差；
- 若有协整，进入 [[Error Correction Model]]。

## Step 1. 先做单位根预检

对每个变量做 [[Unit Root Test]]。

如果变量不是同阶单整，通常不要直接做 EG 协整检验。

最常见场景是所有变量都是 $I(1)$。

## Step 2. 估计长期关系

例如：
$$
y_t=\alpha+\beta x_t+e_t.
$$

用 OLS 得到残差：
$$
\hat e_t=y_t-\hat\alpha-\hat\beta x_t.
$$

这里的残差就是估计出来的长期偏离。

## Step 3. 检验残差是否平稳

对 $\hat e_t$ 做单位根检验。

原假设是：

> 残差有单位根，不存在协整。

如果拒绝原假设，说明残差平稳，变量存在协整关系。

注意：残差是估计出来的，临界值不能直接照搬普通 ADF 表。

## Step 4. 若存在协整，建立 ECM

例如：
$$
\Delta y_t=\alpha+\gamma\hat e_{t-1}+\sum_i\phi_i\Delta y_{t-i}+\sum_j\theta_j\Delta x_{t-j}+u_t.
$$

$\hat e_{t-1}$ 负责长期纠偏，差分项负责短期动态。

## 常见错误

- 没有先确认变量是 $I(1)$。
- 残差 ADF 使用普通临界值。
- 换被解释变量后结果可能不同，却没有解释标准化选择。
- 多变量多个协整关系仍硬用 EG；这时应看 [[Johansen Cointegration Test]]。

## 来自课程位置

- [[07_协整和误差修正模型#3.1 EG两步法|时间序列 07：EG 两步法]]

## 关联卡片

- [[Cointegration]]
- [[Error Correction Model]]
- [[ADF Test Steps]]
- [[Spurious Regression]]
- [[Johansen Cointegration Test]]

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
