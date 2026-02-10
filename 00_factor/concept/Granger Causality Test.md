---
aliases:
- 格兰杰因果关系检验
- 格兰杰因果检验
- Granger Causality Test
tags:
- 时间序列
- 计量经济学
- concept
---
格兰杰因果检验（Granger Causality Test）用于检验一个时间序列对预测另一个时间序列是否有帮助。

## 定义

序列$x_t$被称为$y_t$的格兰杰原因，如果包含$x_t$的滞后项可以显著改善对$y_t$的预测。

## 基本思想

如果$x_t$的滞后信息能够降低$y_t$的预测误差，则称$x_t$格兰杰导致$y_t$。

## 检验方法

### 1. VAR模型方法

估计受限模型（不包含$x_t$滞后）和非受限模型（包含$x_t$滞后），然后使用F检验比较两者。

受限模型：
$y_t = \sum_{i=1}^{p} \alpha_i y_{t-i} + \varepsilon_t$

非受限模型：
$y_t = \sum_{i=1}^{p} \alpha_i y_{t-i} + \sum_{j=1}^{q} \beta_j x_{t-j} + u_t$

### 2. F检验统计量

$F = \frac{(RSS_R - RSS_U)/q}{RSS_U/(T-p-q)}$

其中：
- $RSS_R$：受限模型的残差平方和
- $RSS_U$：非受限模型的残差平方和
- q：$x_t$滞后阶数
- p：$y_t$滞后阶数

## 判断标准

- 若F统计量显著，拒绝原假设，认为$x_t$是$y_t$的格兰杰原因
- 反之，则认为$x_t$不是$y_t$的格兰杰原因

## 注意事项

1. **方向性**：格兰杰因果具有方向性，$x_t$→$y_t$不一定意味着$y_t$→$x_t$
2. **非因果关系**：格兰杰因果不同于统计相关或物理因果关系
3. **平稳性要求**：序列需要平稳或协整
4. **滞后阶数**：检验结果对滞后阶数选择敏感

## 应用

1. **政策分析**：评估政策变量的因果影响
2. **市场研究**：分析市场之间的信息传导
3. **宏观分析**：研究经济变量的因果关系

相关链接: [[VAR Model|VAR]], [[Impulse Response Function|脉冲响应函数]], [[Cointegration|协整]]

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
