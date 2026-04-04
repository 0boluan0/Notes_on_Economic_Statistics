---
aliases:
- Least Squares
- least-squares
- 最小二乘
tags:
- concept
- 线性代数
---
# Least Squares

>[!note] 它是什么
> - 「Least Squares」是指当 $Ax=b$ 无精确解时，通过最小化残差平方和来寻找最佳近似解的方法。
>
>[!note] 最小可检索信息
> - 定义：寻找 $\hat{x}$ 使 $\|A\hat{x}-b\|^2$ 最小。
> - 符号/公式：正规方程 $A^TA\hat{x}=A^Tb$。
> - 最小例子：用一条直线对一组散点做 best fit。
>
## 关键性质
- 最小二乘解来自把 $b$ 正交投影到 $\mathrm{Col}(A)$ 上。
- 若 A 列独立，则 $A^TA$ 可逆，最小二乘解唯一。

## 关联卡片
- [[Orthogonal Projection]]
- [[Projection Matrix]]
- [[Least Squares via Normal Equations]]
- [[Pseudoinverse]]

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
