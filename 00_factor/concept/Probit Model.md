---
aliases:
- probit模型
- probit
- Probit Model
tags:
- 计量经济学
- 离散选择模型
- 经济
- concept
---

# Probit Model
probit模型（Probit Model）是用于分析二元被解释变量（取值为0或1）的回归模型，使用标准正态分布作为潜在变量的分布。

## 模型形式

潜在效用模型：

$U_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik} + \varepsilon_i$

其中 $\varepsilon_i \sim N(0, 1)$。

实际选择结果：

$$
y_i =
\begin{cases}
1, & \text{if } U_i \geq 0 \\
0, & \text{if } U_i < 0
\end{cases}
$$

## 概率表达式

$P(y_i = 1 | x_i) = P(U_i \geq 0) = P(\varepsilon_i \geq -\beta_0 - \beta_1 x_{i1} - \cdots - \beta_k x_{ik})$

$= 1 - \Phi(-\beta_0 - \beta_1 x_{i1} - \cdots - \beta_k x_{ik}) = \Phi(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik})$

其中Φ(·)是标准正态分布的累积分布函数。

## 优势比（Odds Ratio）

$\text{Odds} = \frac{P(y_i = 1)}{P(y_i = 0)} = \frac{\Phi(X_i\beta)}{1 - \Phi(X_i\beta)}$

与logit模型不同，probit模型的优势比与解释变量呈非线性关系。

## 系数解释

- **截距 $\beta_0$**：当所有解释变量为 0 时，事件发生的概率为 $\Phi(\beta_0)$
- **斜率 $\beta_j$**：解释变量 $x_j$ 对概率的影响通过 $\Phi$ 函数传递
- **边际效应**：$\partial P(y=1)/\partial x_j = \phi(X\beta)\beta_j$，其中 $\phi$ 是标准正态密度函数

## 估计方法

极大似然估计（MLE）

对数似然函数：

$\ell(\beta) = \sum_{i=1}^{n} y_i \ln \Phi(X_i\beta) + (1-y_i) \ln [1-\Phi(X_i\beta)]$

## 模型评估

### 1. 拟合优度

#### 伪R²（Pseudo R²）

McFadden R²：
$R^2_{McFadden} = 1 - \frac{\ell(\hat{\beta})}{\ell(0)}$

#### 拟合优度检验

比较完整模型和仅包含截距的模型。

### 2. 预测准确性

#### 分类表

| 实际/预测 | 预测=1 | 预测=0 |
|-----------|--------|--------|
| 实际=1   | TP     | FN     |
| 实际=0   | FP     | TN     |

#### 正确率

$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}$

#### 灵敏度和特异度

- 灵敏度（Sensitivity）= TP/(TP+FN)
- 特异度（Specificity）= TN/(FP+TN)

### 3. AUC和ROC曲线

- ROC曲线：以(1-特异度, 灵敏度)绘制的曲线
- AUC（Area Under Curve）：ROC曲线下面积，衡量模型区分能力

## 与logit模型比较

| 性质 | probit模型 | logit模型 |
|------|----------|-----------|
| 误差分布 | 标准正态分布 | 逻辑分布 |
| 优势比 | 非线性 | 线性（易于解释） |
| 估计方法 | MLE | MLE |
| 结果差异 | 类似 | 类似 |

## 应用

1. **经济学**：就业决策、消费选择、购买决策
2. **金融学**：违约预测、股票涨跌预测
3. **市场营销**：客户购买行为预测
4. **社会科学**：投票行为、政策支持度

相关链接: [[Logit Model|logit模型]], [[Linear Probability Model|LPM模型]], [[Maximum Likelihood Estimation|极大似然估计]]

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
