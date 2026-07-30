---
aliases:
- Johansen Cointegration Test
- Johansen Test
- Johansen检验
- 约翰森协整检验
tags:
  - concept
  - 时间序列
  - 计量经济学
---
# Johansen Cointegration Test

## 先记一句话

Johansen 检验就是：**在 VAR/VECM 系统里，通过长期影响矩阵的秩判断有几个协整关系**。

它比 EG 两步法更适合多变量系统。

## 它是什么

VECM 写作：
$$
\Delta y_t=\Pi y_{t-1}+\sum_{i=1}^{p-1}\Gamma_i\Delta y_{t-i}+\varepsilon_t.
$$

关键是 $\Pi$ 的秩：

- $r=0$：无协整；
- $0<r<k$：有 $r$ 个协整关系；
- $r=k$：变量本身平稳，不是典型协整问题。

如果
$$
\Pi=\alpha\beta^T,
$$
则 $\beta$ 给出协整向量，$\alpha$ 给出调整速度。

## 它解决什么判断

Johansen 检验回答：

> 多个 $I(1)$ 变量之间到底有几个长期均衡关系？

这正是 EG 两步法不擅长的问题。

## 两个统计量

Trace test：
$$
\lambda_{\text{trace}}(r)
=-T\sum_{i=r+1}^{k}\ln(1-\hat\lambda_i).
$$

Max-eigen test：
$$
\lambda_{\max}(r,r+1)
=-T\ln(1-\hat\lambda_{r+1}).
$$

直觉上，显著非零的特征值越多，协整秩越高。

## 常见误区

- Johansen 检验前仍要做单位根预检，确认变量通常是 $I(1)$。
- $\Pi$ 的秩不是随便一个相关矩阵的秩，它来自 VECM 的长期影响结构。
- Trace 和 Max-eigen 可能给出不同结论，需要结合经济含义和模型设定。

## 来自课程位置

- [[07_协整和误差修正模型#3.2 Johansen系统协整检验|时间序列 07：Johansen 检验]]

## 关联卡片

- [[Johansen Cointegration Test Steps]]
- [[Cointegration]]
- [[Error Correction Model]]
- [[Cointegration theorem]]
- [[VAR Model]]
- [[Matrix Rank]]


## 最小例子

把 **Johansen Cointegration Test** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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
