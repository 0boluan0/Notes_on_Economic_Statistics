---
aliases:
- 因子分析
- FA
tags:
- concept
- multivariate statistics
---
# 因子分析

## 定义

因子分析是用少量不可观测的潜在因子（Factors）来描述多个可观测变量之间协方差结构的统计方法。

## 核心思想

如果变量可按相关性分组：
- **组内变量**：高度相关
- **组间变量**：相关性较低

每组用一个公共因子表示。

## 正交因子模型

### 模型设定

$$ X - \mu = LF + \epsilon $$

其中：
- $X$：p × 1 可观测随机向量
- $\mu$：p × 1 均值向量
- $L$：p × m 因子载荷矩阵（Factor Loading Matrix）
- $F$：m × 1 公因子向量（Common Factors），不可观测
- $\epsilon$：p × 1 特殊因子向量（Unique Factors）

### 假设条件

1. **公因子**：
   - $E(F) = 0$
   - $\text{Cov}(F) = I_m$（单位矩阵，因子之间不相关）

2. **特殊因子**：
   - $E(\epsilon) = 0$
   - $\text{Cov}(\epsilon) = \Psi$（对角矩阵）

3. **独立性**：
   - $\text{Cov}(\epsilon, F) = 0$

## 协方差分解

$$ \Sigma = LL' + \Psi $$

其中：
- $LL'$：公因子贡献的协方差（公共部分）
- $\Psi$：特殊因子的协方差（特殊部分）

## 公共度与特殊方差

### 公共度（Communality）

$$ h_i^2 = \sum_{j=1}^m l_{ij}^2 $$

表示第 i 个变量由公因子解释的方差部分。

### 特殊方差（Specific Variance）

$$ \psi_i = \sigma_{ii} - h_i^2 $$

表示第 i 个变量未被公因子解释的方差（独特部分）。

## 因子载荷的旋转

### 非唯一性
因子载荷矩阵 $L$ 不是唯一的。如果 $T$ 是正交矩阵：

$$ L^* = LT $$

也满足模型，因为 $(LT)(LT)' = LTT'L' = LL'$。

### 旋转目的
- 使因子载荷矩阵更易解释
- 获得简单结构（Simple Structure）：每个变量主要与少数因子相关

### 常见旋转方法
1. **方差最大化旋转（Varimax）**：最大化载荷平方的方差
2. **四次方最大化旋转（Quartimax）**
3. **斜交旋转（Oblimin）**：允许因子之间相关

## 参数估计方法

### 1. 主成分法（Principal Component Method）

**步骤**：
1. 使用样本协方差矩阵 $S$
2. 对 $S$ 进行特征值分解：$S = \sum_{j=1}^p \lambda_j e_j e_j'$
3. 保留最大的 m 个特征值
4. 计算因子载荷：$L = [\sqrt{\lambda_1}e_1, \sqrt{\lambda_2}e_2, \dots, \sqrt{\lambda_m}e_m]$
5. 计算特殊方差：$\psi_i = \sigma_{ii} - \sum_{j=1}^m l_{ij}^2$

### 2. 极大似然法（Maximum Likelihood Method）

假设 $X \sim N_p(\mu, \Sigma)$，在约束 $L'\Psi^{-1}L$ 为对角矩阵下估计 $L$ 和 $\Psi$。

**考试不考**。

## 确定因子数量

### 1. 碎石图法（Scree Plot）
- 类似 PCA，观察特征值变化

### 2. 累积方差贡献率
- 公因子解释的方差比例

### 3. Kaiser 准则
- 保留特征值大于 1 的因子

### 4. 平行分析（Parallel Analysis）
- 与随机数据的特征值比较

## 与主成分分析的比较

| 特征 | 主成分分析 | 因子分析 |
|------|-----------|---------|
| 目标 | 数据压缩 | 结构建模 |
| 模型 | $Y = PX$ | $X = LF + \epsilon$ |
| 方差 | 完全解释 | 区分公共和特殊 |
| 唯一性 | 唯一 | 不唯一（可旋转）|
| 模型假设 | 无 | 有潜在因子假设 |

因子分析可以看作是 PCA 的扩展。

## 应用

1. **心理学**：智力测验、人格量表
2. **金融**：风险因子模型（如 Fama-French 三因子）
3. **社会科学**：问卷数据分析
4. **营销**：消费者行为因子分析

## 相关概念

- [[PCA|主成分分析]]
- [[00_factor/concept/Communality|公共度]]
- [[Factor Loadings|因子载荷]]
