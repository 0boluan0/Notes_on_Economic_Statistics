---
aliases:
- Cramer's Rule
- Cramers Rule
- 克拉默法则
tags:
- concept
- 线性代数
---
# Cramer's Rule

## 它是什么
- 「Cramer's Rule」是指用行列式显式写出方阵线性方程组解的公式。

## 最小可检索信息
- 定义：当 $\det(A)\neq 0$ 时，$Ax=b$ 的各分量可由列替换后的行列式比值得到。
- 符号/公式：$x_i=\dfrac{\det(A_i)}{\det(A)}$。
- 最小例子：二维线性方程组可直接用两个 2×2 行列式求解。

## 关键性质
- 只适用于可逆方阵。
- 理论上清晰，但数值计算中通常不如消元稳定。

## 关联卡片
- [[Determinant]]
- [[Matrix Inverse]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```
