---
aliases:
- 赤池信息准则
- Akaike Information Criterion
- AIC
tags:
- 统计学
- 模型选择
- concept
---

# AIC

AIC（Akaike Information Criterion，赤池信息准则）是衡量统计模型拟合优度与模型复杂度之间权衡的信息准则。

>[!note] 定义
>
> $AIC = -2 \ln(L_{max}) + 2k$
>
> 其中：
> - $L_{max}$：模型的最大化似然值
> - k：模型中估计参数的个数
>
## 构成部分

1. **拟合项**：$-2\ln(L_{max})$，衡量模型残差的不可解释程度
2. **惩罚项**：$2k$，对模型复杂度的惩罚

## 性质

- AIC值越小，模型越好
- 在所有备选模型中选择AIC最小的模型
- 对参数个数的惩罚较温和（相对于BIC）

## 应用

1. **时间序列模型选择**：选择ARMA模型的阶数(p,q)
2. **回归模型选择**：选择最优的变量子集
3. **VAR模型选择**：选择VAR模型的滞后阶数p

## AIC的变体

- **AICc**：$小样本修正的AIC：AICc = AIC + \frac{2k(k+1)}{n-k-1}$
- **HQIC**（Hannan-Quinn Information Criterion）：另一种信息准则

## 与BIC的比较

| 准则 | 惩罚项 | 特点 |
|------|--------|------|
| AIC | 2k | 惩罚较轻，倾向于选择更复杂的模型 |
| BIC | k·ln(n) | 惩罚较重，倾向于选择更简洁的模型 |

在大样本下，BIC比AIC更倾向于选择真实模型（如果真实模型在备选集中）。

## 相关链接

- 比较准则：[[BIC]]
- 应用：[[ARMA]]模型选择

相关链接: [[BIC]], [[ARMA]], [[VAR Model|VAR]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[BIC]]、[[ARMA]]、[[VAR Model]]。

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
