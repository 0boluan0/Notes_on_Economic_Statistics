---
aliases:
- Cointegration
- cointegration
- 协整
tags:
- concept
- 时间序列
- 计量经济学
---

# Cointegration

## 先记一句话

协整就是：**几个各自非平稳的变量，存在某个线性组合是平稳的**。

直觉上：

> 它们可以短期乱走，但长期不能无限偏离彼此。

## 它是什么

若 $x_t$ 的各分量都是 $I(1)$，但存在非零向量 $\beta$，使得
$$
\beta^Tx_t
$$
是 $I(0)$，则这些变量存在协整关系。

$\beta$ 是协整向量。

## 一个最小例子

长期利率和短期利率可能各自像随机游走一样非平稳。

但如果某个利差
$$
r_{L,t}-\beta r_{S,t}
$$
是平稳的，就表示两者有长期均衡关系。

短期可以偏离，但偏离不会无限扩大。

## 它解决什么判断

协整用于判断：

- 两个或多个 $I(1)$ 变量能不能做水平关系分析；
- 水平回归是不是 [[Spurious Regression]]；
- 是否应该用 [[Error Correction Model]] 而不是只做差分回归。

## 和差分的分流

如果变量 $I(1)$ 且没有协整：

- 通常差分后建模；
- 水平回归容易伪回归。

如果变量 $I(1)$ 且有协整：

- 不应只保留差分；
- 要把长期均衡误差放进 ECM/VECM。

## 常见误区

- 协整不是普通相关；它要求变量单整，并且某个线性组合平稳。
- 两个变量都上升不等于协整。
- 差分能解决非平稳，但会丢掉长期均衡信息。
- EG 两步法只能方便地检一个关系；多变量多个协整向量要看 [[Johansen Cointegration Test]]。

## 来自课程位置

- [[07_协整和误差修正模型#2.1 协整的定义|时间序列 07：协整定义]]
- [[07_协整和误差修正模型#2.3 协整与误差修正模型|时间序列 07：协整与 ECM]]

## 关联卡片

- [[Error Correction Model]]
- [[Engle-Granger Two-Step Test]]
- [[Johansen Cointegration Test]]
- [[Cointegration theorem]]
- [[Spurious Regression]]
- [[Unit Root Test]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Spurious Regression]]、[[Error Correction Model]]、[[Johansen Cointegration Test]]、[[07_协整和误差修正模型]]、[[Engle-Granger Two-Step Test]]、[[Cointegration theorem]]、[[Unit Root Test]]。

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
