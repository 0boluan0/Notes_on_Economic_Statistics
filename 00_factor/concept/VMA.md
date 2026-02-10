---
aliases:
- 向量移动平均
- Vector Moving Average
- VMA
tags:
- 时间序列
- 多变量模型
- concept
---
VMA（Vector Moving Average，向量移动平均）是VAR向量自回归模型的可逆表示形式，将当前值表示为当前和过去冲击的函数。

## 模型形式

$\mathbf{y}_t = \sum_{j=0}^{\infty} \mathbf{\Psi}_j \mathbf{\varepsilon}_{tj}$

其中：
- $\mathbf{y}_t$：n×1维的变量向量
- $\mathbf{\Psi}_j$：$n×n维的脉冲响应矩阵，\mathbf{\Psi}_0 = \mathbf{I}$
- $\mathbf{\varepsilon}_t$：n×1维的误差项向量

## 从VAR到VMA的转换

对于VAR(p)模型：

$\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{y}_{ti} + \mathbf{\varepsilon}_t$

通过迭代可以得到VMA(∞)表示：

$\mathbf{y}_t = \mu + \sum_{j=0}^{\infty} \mathbf{\Psi}_j \mathbf{\varepsilon}_{tj}$

## 脉冲响应矩阵

脉冲响应矩阵$\mathbf{\Psi}_j$的计算：

$\mathbf{\Psi}_0 = \mathbf{I}$
$\mathbf{\Psi}_1 = \mathbf{A}_1$
$\mathbf{\Psi}_2 = \mathbf{A}_1\mathbf{\Psi}_1 + \mathbf{A}_2$
$\mathbf{\Psi}_j = \sum_{i=1}^{p} \mathbf{A}_i \mathbf{\Psi}_{ji}$

## 性质

1. **正交化脉冲响应**：通常通过Cholesky分解使误差项正交
2. **收敛条件**：如果VAR模型平稳，则$\mathbf{\Psi}_j \to 0$当j → ∞
3. **冲击识别**：需要额外的结构假设识别冲击

## 应用

1. **[[Impulse Response Function|脉冲响应函数]]**：通过$\mathbf{\Psi}_j$绘制脉冲响应图
2. **冲击分析**：分析一个冲击对系统中所有变量的动态影响
3. **方差分解**：计算各变量冲击对预测误差方差的贡献

相关链接: [[VAR Model|VAR]], [[Impulse Response Function|脉冲响应函数]], [[Variance Decomposition|方差分解]]

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
