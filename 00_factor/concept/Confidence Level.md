---
aliases:
- Confidence Level
- 置信水平
- VaR confidence level
tags:
- concept
- risk-management
- statistics
---

# Confidence Level

## 先记一句话

置信水平 $1-\alpha$ 是一套区间构造程序在重复抽样中的长期覆盖率。

## 它是什么

若随机区间 $C(X)$ 满足
$$
\Pr_\theta\{\theta\in C(X)\}\ge 1-\alpha,
$$
则它具有至少 $1-\alpha$ 的置信水平。概率来自重复样本 $X$，不是把固定参数 $\theta$ 当作随机变量。

在 VaR 语境里，同一个数值 $\alpha$ 也常表示损失分布所取的分位水平：

VaR 的 $\alpha$ 置信水平对应尾部概率 $1-\alpha$：

$$
P(L>\operatorname{VaR}_\alpha)\approx 1-\alpha
$$

99% VaR 意味着模型允许约 1% 的例外概率。

## 解决什么判断

在统计推断中，它规定区间生成规则的覆盖保证；在 VaR 中，它规定关注的损失分位数。两种语境必须由上下文区分。

## 最小例子

同一组合下，99% VaR 通常高于 95% VaR，因为它取更深的损失尾部。

## 易混点

- 99% VaR 不是“有 99% 概率损失这么多”，而是“约 1% 概率损失超过它”。
- 置信水平越高，尾部样本越少，[[VaR Standard Error]] 往往越大。
- 不同置信水平的 VaR 不能直接比较监管含义。

## 来自课程位置

- [[12_VAR风险]]

## 关联卡片

- [[VaR]]
- [[Holding Period]]
- [[Observation Window]]
- [[Kupiec Test]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[VaR Standard Error]]、[[12_VAR风险]]、[[VaR]]、[[Holding Period]]、[[Observation Window]]、[[Kupiec Test]]。

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
