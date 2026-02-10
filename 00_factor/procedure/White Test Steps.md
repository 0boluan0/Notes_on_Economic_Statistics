---
aliases:
- White异方差检验步骤
- White检验步骤
- White Test Steps
tags:
- procedure
- 01_Econometrics
---
# White检验步骤

## 适用场景

检验回归模型是否存在异方差问题，特别适用于检验误差项方差是否随自变量变化而变化。

## 所需数据/条件

- 原始回归模型：$y_i = \beta_0 + \beta_1 x_{1i} + \cdots + \beta_k x_{ki} + \epsilon_i$
- 回归残差 $\hat{\epsilon}_i$
- 自变量数据 $x_{1i}, \dots, x_{ki}$

## 计算步骤

### 步骤 1：运行原始回归

$对原始模型进行OLS估计，得到残差序列 \{\hat{\epsilon}_i\}_{i=1}^n。$

**注意点**：原始模型需要正确设定，遗漏变量可能导致虚假异方差。

### 步骤 2：构造辅助回归（White检验形式）

建立辅助回归，将残差平方对常数项、原自变量及其平方和交叉项回归：

$ \hat{\epsilon}_i^2 = \alpha_0 + \alpha_1 x_{1i} + \cdots + \alpha_k x_{ki} + \alpha_{k+1} x_{1i}^2 + \cdots + \alpha_{2k} x_{ki}^2 + \alpha_{2k+1} x_{1i}x_{2i} + \cdots + \text{error}_i $

**注意点**：当自变量数量多时，此回归包含大量参数，可能损失自由度。

### 步骤 3：使用简化形式（可选）

为减少参数损失，可以使用不含交叉项的简化White检验：

$ \hat{\epsilon}_i^2 = \alpha_0 + \alpha_1 x_{1i} + \cdots + \alpha_k x_{ki} + \alpha_{k+1} x_{1i}^2 + \cdots + \alpha_{2k} x_{ki}^2 + \text{error}_i $

**注意点**：简化形式在大多数情况下检验功效接近完整形式。

### 步骤 4：估计辅助回归

对辅助回归进行OLS估计，计算：
- 辅助回归的判定系数 $R^2_{\text{aux}}$
- 或未修正的 $R^2$（取决于软件实现）

### 步骤 5：构造检验统计量

$ LM = n \times R^2_{\text{aux}} $

其中 $n$ 为样本量。

**注意点**：此统计量在原假设成立时渐近服从 $\chi^2$ 分布。

### 步骤 6：确定自由度

检验的自由度为辅助回归中斜率系数的个数：
- 完整形式：$df = 2k + k(k-1)/2$
- 简化形式：$df = 2k$

**注意点**：自由度不包括常数项。

### 步骤 7：进行假设检验

- 原假设 $H_0$：$同方差性成立，\alpha_1 = \alpha_2 = \cdots = 0$
- 备择假设 $H_1$：存在异方差，至少一个 $\alpha_i \neq 0$

**决策规则**：
- 若 $LM > \chi^2_{\alpha}(df)$，拒绝 $H_0$，认为存在异方差
- 若 $LM \le \chi^2_{\alpha}(df)$，无法拒绝 $H_0$，认为同方差成立

**注意点**：显著性水平 $\alpha$ 常取 5% 或 1%。

## 关键公式

**完整White检验辅助回归**：
$ \hat{\epsilon}_i^2 = \alpha_0 + \sum_{j=1}^{k} \alpha_j x_{ji} + \sum_{j=1}^{k} \alpha_{k+j} x_{ji}^2 + \sum_{1 \le j < l \le k} \alpha_{\cdot} x_{ji}x_{li} + \text{error}_i $

**LM统计量**：
$ LM = n \times R^2_{\text{aux}} \xrightarrow{d} \chi^2(df) $

**临界值**：
$ \text{Critical Value} = \chi^2_{\alpha}(df) $

## 常见问题

1. **自由度损失**：大量自变量导致检验自由度大，功效降低。
2. **模型设定错误**：遗漏变量可能被误判为异方差。
3. **小样本偏差**：小样本下LM统计量的渐近性质不成立。
4. **多重共线性**：辅助回归中的高次项可能高度相关。

## 相关概念
[[Heteroskedasticity|异方差]]
[[Weighted Least Squares Estimation|加权最小二乘估计]]
[[White Robust Standard Errors|稳健标准误]]

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
