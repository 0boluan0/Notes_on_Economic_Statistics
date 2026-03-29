---
aliases:
- 公共度
- Common Variance
- Communality
tags:
- concept
- multivariate statistics
---
# 公共度

>[!note] 定义
>
> 公共度是指某个可观测变量的方差中，能被所有公共因子共同解释的部分。
>
> 在因子分析中：
>
> $ h_i^2 = \sum_{j=1}^m l_{ij}^2 $
>
> 其中：
> - h_i^2：第 i 个变量的公共度
> - l_{ij}：第 i 个变量在第 j 个因子上的载荷
> - m：公共因子数量
>
## 数学含义

从协方差分解 $\Sigma = LL' + \Psi$ 来看：
$$
\sigma_{ii} = h_i^2 + \psi_i
$$
即：
$$
h_i^2 = \sigma_{ii} - \psi_i
$$

变量的总方差 = 公共度 + 特殊方差

## 直观理解

### 高公共度（接近总方差）
- 该变量与其他变量高度相关
- 能很好地由公共因子代表

### 低公共度
- 该变量与其他变量相关性弱
- 有很大部分是独特的、未被解释的

## 计算示例

假设因子载荷矩阵：

$$
L =
\begin{pmatrix}
0.8 & 0.2 \\
0.6 & 0.7 \\
0.4 & 0.5
\end{pmatrix}
$$

计算公共度：

- 变量 1：$h_1^2 = 0.8^2 + 0.2^2 = 0.64 + 0.04 = 0.68$
- 变量 2：$h_2^2 = 0.6^2 + 0.7^2 = 0.36 + 0.49 = 0.85$
- 变量 3：$h_3^2 = 0.4^2 + 0.5^2 = 0.16 + 0.25 = 0.41$

## 应用

### 1. 评估因子模型质量
- 如果某些变量的公共度很低（如 < 0.5），说明模型对该变量解释不好
- 可能需要增加因子数量或调整模型

### 2. 变量选择
- 公共度高的变量更适合参与因子分析
- 公共度低的变量可能是噪音或测量误差

### 3. 模型比较
- 比较不同因子数量模型的公共度
- 较高的平均公共度表示更好的模型拟合

## 相关概念

- [[Specific Variance|特殊方差]]
- [[Factor Analysis|因子分析]]
- [[Factor Loadings|因子载荷]]

## 性质

1. **范围**：$0 \leq h_i^2 \leq \sigma_{ii}$（或 $0 \leq h_i^2 \leq 1$ 对标准化变量）
2. **非负**：公共度总是非负的
3. **总和**：所有变量公共度之和等于所有因子方差之和

$$
\sum_{i=1}^p h_i^2 = \sum_{j=1}^m \left(\sum_{i=1}^p l_{ij}^2\right)
$$

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
