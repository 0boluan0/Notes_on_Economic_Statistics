---
aliases:
- Ljung-Box检验
- Box-Pierce Test
- Ljung
- Ljung-Box Test
tags:
  - 时间序列
  - 统计检验
  - concept
---
# Ljung-Box Test

Ljung-Box检验是用于检验时间序列是否存在自相关的统计方法，常用于检验残差序列是否为白噪声。

## 检验统计量

**Box-Pierce统计量：**

$Q = T \sum_{k=1}^{m} \hat{\rho}_k^2$

**Ljung-Box统计量（小样本修正）：**

$Q_{LB} = T(T+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{T-k}$

其中：
- T：样本量
- m：检验的滞后阶数
- $\hat{\rho}_k$：样本自相关系数

## 假设

**原假设H₀**：序列是白噪声（所有自相关系数为0）
**备择假设H₁**：序列不是白噪声（存在自相关）

## 分布

在H₀下，Q和$Q_LB$近似服从χ²(m)分布。

## 检验步骤

1. **计算样本ACF**：$计算\hat{\rho}_k，k = 1, 2, ..., m$
2. **计算检验统计量**：使用Box-Pierce或Ljung-Box公式
3. **比较临界值**：比较统计量与χ²(m)分布的临界值
4. **判断**：若统计量显著，拒绝白噪声假设

## Box-Pierce与Ljung-Box比较

| 方法 | 公式 | 特点 |
|------|------|------|
$| Box-Pierce | Q = T \sum \hat{\rho}_k^2 | 简单，但在小样本下效果较差 |$
$| Ljung-Box | Q_{LB} = T(T+2) \sum \frac{\hat{\rho}_k^2}{T-k} | 小样本修正，更常用 |$

## 应用

1. **模型诊断**：检验ARMA/GARCH模型的残差是否为白噪声
2. **白噪声检验**：检验序列是否完全随机
3. **模型识别**：帮助识别ARMA模型的阶数

## 扩展：McLeod-Li检验

McLeod-Li检验用于检验平方序列的自相关，检测ARCH效应：

$Q = T(T+2) \sum_{k=1}^{m} \frac{r_k^2}{T-k}$

其中$r_k$是平方序列的自相关系数。

相关链接: [[White Noise Test|白噪声检验]], [[ARCH]], [[Autocorrelation Function|自相关函数]]


## 最小例子

把 **Ljung-Box Test** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
## 概念边界

本卡片只回答“它是什么”，集中在定义、核心关系与最小例子；具体估计步骤、证明和诊断流程应分别放在 procedure、proof 或 system 卡片中。
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
