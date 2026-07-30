---
aliases:
- 极大似然估计步骤
- MLE估计步骤
- MLE
- MLE Estimation Steps
tags:
- procedure
- 01_Econometrics
type: procedure
---
# MLE估计步骤

## 输入

- 观测样本、参数化概率模型和参数空间约束。
- 似然函数可计算，及用于收敛、标准误和模型比较的诊断规则。

## 输出

极大似然估计 \hat{\theta}、对数似然值、标准误/信息矩阵和收敛诊断。

## 适用场景

当模型非线性（如ARMA、GARCH、Logit等）或无法使用OLS时，使用极大似然估计参数。适用于已知误差项分布的情况。

## 所需数据/条件

- 样本数据 $($y_1$, $x_1$), \dots, ($y_n$, $x_n$)$
- 误差项的概率分布 $f(\epsilon; \theta)$
- 参数空间 $\Theta$
- 似然函数 $L(\theta; \text{data})$

## 计算步骤

### 步骤 1：设定模型和误差项分布

明确模型形式和误差项分布假设：
$ y_i = g(x_i; \theta) + \epsilon_i, \quad \epsilon_i \sim f(\cdot; \theta_0) $

**注意点**：分布假设正确性至关重要，错误假设导致估计有偏。

### 步骤 2：写出似然函数

假设样本独立同分布，联合密度函数为：
$ L(\theta) = \prod_{i=1}^{n} f(y_i - g(x_i; \theta); \theta) $

**注意点**：对于分类数据，使用离散分布；对于连续数据，使用连续密度。

### 步骤 3：取对数似然函数

$ \ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln f(y_i - g(x_i; \theta); \theta) $

**注意点**：对数变换将乘积转换为求和，便于数值计算和求导。

### 步骤 4：构建最优化问题

$ \hat{\theta}_{\text{MLE}} = \arg\max_{\theta \in \Theta} \ell(\theta) $

等价于：
$ \hat{\theta}_{\text{MLE}} = \arg\min_{\theta \in \Theta} [-\ell(\theta)] $

**注意点**：对数似然的负值为损失函数。

### 步骤 5：求解一阶条件（如可解析解）

求导并令为零：
$ \frac{\partial \ell(\theta)}{\partial \theta} = 0 $

**注意点**：大多数模型无解析解，需用数值方法。

### 步骤 6：使用数值优化算法（通常情况）

**常用算法**：
- 牛顿-拉夫森法（Newton-Raphson）
- 拟格朗日乘子法（L-BFGS）
- 共轭梯度法（Conjugate Gradient）
- BFGS（Broyden-Fletcher-Goldfarb-Shanno）

**算法流程**：
1. 初始猜测 \theta_0
2. $计算梯度 g_k = \nabla \ell(\theta_k)$
3. $更新 \theta_{k+1} = \theta_k + \alpha_k d_k（搜索方向）$
4. 收敛检验：$|$g_k$| < \epsilon$ 或 $|\ell(\theta_{k+1}) - \ell(\theta_k)| < \epsilon$

**注意点**：初始值选择影响收敛速度和结果。

### 步骤 7：验证二阶条件（检查是否为最大值）

$计算海森矩阵 H = \frac{\partial^2 \ell(\theta)}{\partial \theta \partial \theta'}$

**注意点**：在最优解处，$H$ 应为负定矩阵（最大化）或正定矩阵（最小化负对数似然）。

### 步骤 8：计算标准误（如需）

$ \text{Var}(\hat{\theta}) = [I(\hat{\theta})]^{-1} $

其中信息矩阵：
$ I(\theta) = -E\left[\frac{\partial^2 \ell(\theta)}{\partial \theta \partial \theta'}\right] $

**注意点**：用海森矩阵的负逆近似信息矩阵逆。

## 关键公式

**对数似然函数**：
$\ell(\theta) = \sum_{i=1}^{n} \ln f(y_i - g(x_i; \theta); \theta)$

**信息矩阵**：
$ I(\theta) = -E[H(\theta)] $

**渐近有效性**：
$ \sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1}) $

## 常见问题

1. **局部最优**：对数似然可能非凸，算法可能落入局部最优。
2. **初值敏感**：不同初始值可能收敛到不同解。
3. **数值不稳定**：某些参数区域对数似然值很大或很小。
4. **收敛困难**：参数过多或数据不足时优化困难。

## 相关概念
[[OLS Estimation Steps|OLS估计步骤]]
[[Likelihood Ratio Test|似然比检验]]
[[ARMA Model Identification Steps|ARMA模型识别]]
[[GARCH Model Estimation Steps|GARCH模型估计]]

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
