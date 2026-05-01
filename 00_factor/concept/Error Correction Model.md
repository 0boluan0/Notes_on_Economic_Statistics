---
aliases:
- Error Correction Model
- ECM
- Error Correction Mechanism
- 误差修正模型
- 误差纠正模型
- 误差修正机制
tags:
- concept
- 时间序列
- 计量经济学
---
# Error Correction Model

## 先记一句话

ECM 就是：**短期变化由两部分解释：短期冲击 + 上一期偏离长期均衡的程度**。

它是协整的动态版本。

## 它是什么

若长期关系是
$$
y_t-\beta x_t=e_t,
$$
且 $e_t$ 平稳，那么 ECM 可写成
$$
\Delta y_t=\alpha+\gamma e_{t-1}+\text{short-run terms}+\varepsilon_t.
$$

其中 $e_{t-1}$ 是上一期偏离长期均衡的误差。

## 它解决什么判断

ECM 回答：

> 当变量短期偏离长期关系后，下一期会不会被拉回去？

如果 $\gamma$ 的符号和经济含义一致，并且显著，就说明误差修正机制存在。

## 一个最小直觉

长期均衡是：
$$
y_t=\beta x_t.
$$

如果上一期
$$
y_{t-1}>\beta x_{t-1},
$$
说明 $y$ 高于长期均衡。

ECM 中的误差修正项会让 $\Delta y_t$ 向下调整。

## VECM 版本

多变量系统中：
$$
\Delta x_t=\alpha\beta^Tx_{t-1}+\sum_{i=1}^{p-1}\Gamma_i\Delta x_{t-i}+\varepsilon_t.
$$

其中：

- $\beta$：协整向量；
- $\alpha$：调整系数；
- $\beta^Tx_{t-1}$：长期偏离。

## 常见误区

- ECM 不是只做差分；它保留了长期均衡误差。
- 没有协整关系时，ECM 的长期误差项没有稳定意义。
- 调整系数的符号要结合变量定义解释，不能机械背“必须小于 0”。

## 来自课程位置

- [[07_协整和误差修正模型#2.3 协整与误差修正模型|时间序列 07：ECM 与 VECM]]

## 关联卡片

- [[Cointegration]]
- [[Cointegration theorem]]
- [[Engle-Granger Two-Step Test]]
- [[Johansen Cointegration Test]]
- [[VAR Model]]

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
