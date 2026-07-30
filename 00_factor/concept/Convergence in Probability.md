---
aliases:
- Convergence in Probability
- 依概率收敛
tags:
- concept
---

# Convergence in Probability

## 它是什么

「Convergence in Probability」是指随机变量序列以概率收敛到某随机变量。

>[!note] 它是什么
> - 「Convergence in Probability」是指随机变量序列以概率收敛到某随机变量。
>
>[!note] 最小可检索信息
> - 定义：随机变量序列以概率收敛到某随机变量。
> - 符号/公式：$P(|$X_n$-X|>\varepsilon)\to 0$。
> - 最小例子：样本均值以概率收敛到期望。
>
## 最小例子

样本均值以概率收敛到期望。

## 关联卡片
- [[Law of Large Numbers]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Law of Large Numbers]]。

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
