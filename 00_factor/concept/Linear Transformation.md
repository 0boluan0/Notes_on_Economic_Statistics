---
aliases:
- Linear Transformation
- linear map
- linear operator
- 线性变换
- 线性映射
tags:
- concept
- 线性代数
---
# Linear Transformation

## 先记一句话

线性变换就是：**保持加法和数乘结构的映射**。

它满足
$$
T(u+v)=T(u)+T(v),
\qquad
T(cu)=cT(u).
$$

矩阵只是线性变换在某组基底下的坐标表示。

## 它是什么

常见线性变换包括：

- 旋转；
- 缩放；
- 投影；
- 剪切；
- 反射。

这些操作都不会破坏“线性组合”的结构：
$$
T(c_1v_1+c_2v_2)=c_1T(v_1)+c_2T(v_2).
$$

## 一个最小例子

令
$$
T(x,y)=(2x,3y).
$$

那么
$$
T\left(\begin{bmatrix}x\\y\end{bmatrix}\right)
=
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix}
\begin{bmatrix}x\\y\end{bmatrix}.
$$

这说明同一个线性变换可以用矩阵表示。

## 它在题里负责什么

- 把矩阵从“数字表格”解释成“空间操作”。
- 解释 kernel/image：
  - kernel 对应 [[Null Space]]；
  - image 对应 [[Column Space]]。
- 解释 [[Change of Basis]] 和 [[Similar Matrix]]：矩阵变了，但线性变换本身没变。

## 常见误区

- 不是所有函数都是线性变换。带常数项的平移通常不是线性的。
- 矩阵表示依赖基底；线性变换本身不依赖某一个坐标系。
- 对角化不是改变线性变换本身，而是换到更适合它的基底。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.6 Linear transformations and their matrices|Session 3.6]]：线性变换和矩阵表示。
- [[03_Positive Definite Matrices and Applications#Session 3.7 Change of basis; image compression|Session 3.7]]：换基与表示变化。

## 关联卡片

- [[Change of Basis]]
- [[Similar Matrix]]
- [[Matrix Inverse]]
- [[Column Space]]
- [[Null Space]]

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
