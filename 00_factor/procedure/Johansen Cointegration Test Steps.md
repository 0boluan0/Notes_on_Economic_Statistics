---
aliases:
- Johansen Cointegration Test Steps
- Johansen协整检验步骤
tags:
- procedure
- 时间序列
- 计量经济学
---
# Johansen Cointegration Test Steps

## 这张卡什么时候用

当你有多个 $I(1)$ 变量，并且要判断协整关系数量 $r$ 时，用 Johansen 检验。

## 输入

- 多变量时间序列向量 $y_t$；
- 每个变量的单位根检验结果；
- VAR 滞后阶数；
- deterministic terms 设定。

## 输出

- 协整秩 $r$；
- 协整向量 $\beta$；
- 调整系数 $\alpha$；
- 是否建立 VECM。

## Step 1. 做单位根预检

确认变量通常是 $I(1)$。

如果变量本身平稳，Johansen 协整问题就不成立。

如果变量阶数不同，要先重新判断建模框架。

## Step 2. 选择 VAR 滞后阶数

用 AIC/BIC 或残差诊断选择 VAR$(p)$。

Johansen 检验对滞后阶数敏感，所以这里不能省。

## Step 3. 写成 VECM

把 VAR 改写为：
$$
\Delta y_t=\Pi y_{t-1}+\sum_{i=1}^{p-1}\Gamma_i\Delta y_{t-i}+\varepsilon_t.
$$

核心是检验 $\Pi$ 的秩。

## Step 4. 从 $r=0$ 开始检验

使用 Trace test 或 Max-eigen test。

Trace test：
$$
H_0:\operatorname{rank}(\Pi)\le r.
$$

Max-eigen test：
$$
H_0:\operatorname{rank}(\Pi)=r.
$$

如果拒绝，就增加 $r$ 继续检验。

## Step 5. 确定协整秩

找到第一个不能拒绝的位置，结合经济含义确定最终 $r$。

若 $r>0$，继续估计协整向量和调整系数。

## Step 6. 写出 VECM 解释

将
$$
\Pi=\alpha\beta^T
$$
写清楚：

- $\beta^Ty_{t-1}$：长期均衡误差；
- $\alpha$：各变量对偏离的调整速度。

## 常见错误

- 跳过单位根预检。
- 忽略确定性趋势/截距设定。
- Trace 和 Max-eigen 冲突时只机械选一个。
- 找到协整后不解释 $\alpha$ 和 $\beta$ 的经济含义。

## 来自课程位置

- [[07_协整和误差修正模型#3.2 Johansen系统协整检验|时间序列 07：Johansen 系统协整检验]]

## 关联卡片

- [[Johansen Cointegration Test]]
- [[Cointegration]]
- [[Error Correction Model]]
- [[Unit Root Test]]
- [[Matrix Rank]]

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
