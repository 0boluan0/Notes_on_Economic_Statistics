---
aliases:
  - Logit Model
  - Logistic Regression
tags:
  - 计量经济学
  - 离散选择模型
---

logit模型（Logit Model）是用于分析二元被解释变量（取值为0或1）的回归模型，使用逻辑分布作为潜在变量的分布。

## 模型形式

潜在效用模型：

$$U_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik} + \varepsilon_i$$

其中ε_i服从逻辑分布。

实际选择结果：

$$y_i = \begin{cases}
1, & \text{if } U_i \geq 0 \\
0, & \text{if } U_i < 0
\end{cases}$$

## 概率表达式

$$P(y_i = 1 | x_i) = \frac{e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}}} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik})}}$$

或者写作：

$$P(y_i = 1 | x_i) = \Lambda(\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik})$$

其中Λ(z) = 1/(1+e^{-z})是逻辑函数（sigmoid函数）。

## 优势比（Odds Ratio）

$$\textOdds} = \frac{P(y_i = 1)}{P(y_i = 0)} = \frac{P(y_i = 1)}{1 - P(y_i = 1)} = e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}}$$

## 系数解释

- **截距β₀**：当所有解释变量为0时，事件发生的优势比为e^{β₀}
- **斜率β_j**：解释变量x_j增加1个单位，优势比变化e^{β_j}倍
- **边际效应**：∂P(y=1)/∂x_j = Λ(Xβ)[1-Λ(Xβ)]β_j

## 估计方法

极大似然估计（MLE）

对数似然函数：

$$\ell(\beta) = \sum_{i=1}^{n} y_i \ln P(y_i=1) + (1-y_i) \ln P(y_i=0)$$

## 模型评估

### 1. 拟合优度

#### 伪R²（Pseudo R²）

McFadden R²：
$$R^2_{McFadden} = 1 - \frac{\ell(\hat{\beta})}{\ell(0)}$$

#### 拟合优度检验

比较完整模型和仅包含截距的模型。

### 2. 预测准确性

#### 分类表

| 实际/预测 | 预测=1 | 预测=0 |
|-----------|--------|--------|
| 实际=1   | TP     | FN     |
| 实际=0   | FP     | TN     |

#### 正确率

$$\textAccuracy} = \frac{TP + TN}{TP + FP + TN + FN}$$

#### 灵敏度和特异度

- 灵敏度（Sensitivity）= TP/(TP+FN)
- 特异度（Specificity）= TN/(FP+TN)

### 3. AUC和ROC曲线

- ROC曲线：以(1-特异度, 灵敏度)绘制的曲线
- AUC（Area Under Curve）：ROC曲线下面积，衡量模型区分能力

## 与probit模型比较

| 性质 | logit模型 | probit模型 |
|------|----------|-----------|
| 误差分布 | 逻辑分布 | 标准正态分布 |
| 优势比 | 线性（易于解释） | 非线性 |
| 估计方法 | MLE | MLE |
| 结果差异 | 类似 | 类似 |

## 应用

1. **经济学**：就业决策、消费选择、购买决策
2. **金融学**：违约预测、股票涨跌预测
3. **市场营销**：客户购买行为预测
4. **社会科学**：投票行为、政策支持度

相关链接: [[probit模型]], [[LPM模型]], [[极大似然估计]]
