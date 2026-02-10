---
aliases: []
tags:
  - linear-algebra
  - MIT-18.06SC
  - unit3
date: 2026-02-09
科目: Math
---

# Unit3 课堂笔记（MIT 18.06SC）

> 对应资料顺序：`MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf` 到 `Ses3.9sum.pdf`。
> 本单元主线：对称与正定（symmetric & positive definite）→ 复矩阵与 FFT → 相似/Jordan → SVD/换基/伪逆。

## Session 3.1 Symmetric matrices and positive definiteness

### 课程目标
- 理解对称矩阵（symmetric matrix）的谱性质。
- 掌握正定矩阵（positive definite）的判定与意义。

### 核心定义
- 对称矩阵：
  $$A=A^T$$
- 正定：
  $$x^TAx>0,\ \forall x\neq0$$
- 谱分解（对称情形）：
  $$A=Q\Lambda Q^T$$

### 关键推导
- 对称矩阵可正交对角化，特征值均实数。
- 正定等价于所有特征值严格为正。
- 在优化中，正定 Hessian 对应严格凸。

### 例题（含解法）
$$A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$$
- 特征值 $3,1$，均正。
- 因此 `A` 正定。

### 易错点
- 把“主对角线正”误当正定充分条件。
- 忽略矩阵是否对称就讨论正定。

### 与00_factor关联
- [[Symmetric Matrix|对称矩阵]]
- [[Eigenvalues|特征值]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[2,1],[1,2]], dtype=float)
w = np.linalg.eigvals(A)
print("eigenvalues:", w)
print("is positive definite?", np.all(w > 0))
```

## Session 3.2 Complex matrices; fast Fourier transform

### 课程目标
- 理解复内积与共轭转置（conjugate transpose）。
- 理解 FFT 的分治思想。

### 核心定义
- 共轭转置：$A^*=\overline{A}^T$。
- 酉矩阵（unitary）：
  $$U^*U=I$$
- 离散傅里叶变换（DFT）是典型酉变换（适当归一化）。

### 关键推导
- DFT 的偶/奇拆分将复杂度从 $O(n^2)$ 降到 $O(n\log n)$。
- 频域系数可看作在复指数正交基上的投影坐标。

### 例题（含解法）
长度4序列 $x=[1,2,0,0]$：
- 直接 DFT 可得4个频域系数。
- 用 FFT 会得到同样结果，但计算更快。

### 易错点
- 忘记共轭导致内积定义错误。
- 频率索引与物理频率对应关系混乱。

### 与00_factor关联
- [[Power Spectral Density|功率谱密度]]
- [[Spectral Decomposition|谱分解]]

### 代码块（完整保留）
```python
import numpy as np

x = np.array([1,2,0,0], dtype=complex)
X_dft = np.fft.fft(x)
print("FFT coefficients:", X_dft)
print("inverse FFT:", np.fft.ifft(X_dft))
```

## Session 3.3 Positive definite matrices and minima

### 课程目标
- 把线性方程和二次优化联系起来。
- 理解正定矩阵保证唯一极小点。

### 核心定义
- 二次函数：
  $$f(x)=\frac12x^TAx-b^Tx$$
- 梯度条件：
  $$\nabla f(x)=Ax-b=0$$

### 关键推导
- 当 `A` 正定时，驻点唯一且为全局极小：
  $$x^*=A^{-1}b$$
- 这是数值优化中最经典的线性系统来源。

### 例题（含解法）
$$f(x_1,x_2)=x_1^2+x_2^2-2x_1-4x_2$$
- 梯度零点：$(x_1,x_2)=(1,2)$。
- Hessian 为 $2I$，正定，故为唯一极小点。

### 易错点
- 只看一阶条件，不验二阶正定。
- 把半正定场景误判成唯一极小。

### 与00_factor关联
- [[Positive Definite Matrix|正定矩阵]]
- [[Differential Equation|微分方程]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[2,0],[0,2]], dtype=float)
b = np.array([2,4], dtype=float)
x = np.linalg.solve(A, b)
print("argmin x =", x)
```

## Session 3.4 Similar matrices and Jordan form

### 课程目标
- 理解相似变换（similarity transform）含义。
- 理解 Jordan 形用于不可对角化矩阵。

### 核心定义
- 相似：
  $$B=S^{-1}AS$$
- Jordan 块：
  $$J=\lambda I+N,\ N^k=0$$

### 关键推导
- 相似矩阵有相同特征值、迹、行列式。
- 当特征向量不足时，需用广义特征向量形成 Jordan 链。

### 例题（含解法）
$$A=\begin{bmatrix}1&1\\0&1\end{bmatrix}$$
- 只有一个线性无关特征向量，不可对角化。
- Jordan 形就是该矩阵本身（单一 Jordan 块）。

### 易错点
- 误把“特征值相同”当作“矩阵相等”。
- 对角化失败后不知道 Jordan 替代路径。

### 与00_factor关联
- [[Eigenvalues|特征值]]
- [[Eigenvectors|特征向量]]

### 代码块（完整保留）
```python
import sympy as sp

A = sp.Matrix([[1,1],[0,1]])
P, J = A.jordan_form()
print("Jordan J =")
sp.pprint(J)
print("P =")
sp.pprint(P)
```

## Session 3.5 Singular value decomposition

### 课程目标
- 掌握 SVD（奇异值分解）结构。
- 理解奇异值与秩、压缩、降噪关系。

### 核心定义
- SVD：
  $$A=U\Sigma V^T$$
- 奇异值：
  $$\sigma_i=\sqrt{\lambda_i(A^TA)}$$

### 关键推导
- 非零奇异值个数等于 `rank(A)`。
- 截断 SVD 给出最佳低秩近似（Eckart-Young 定理）。

### 例题（含解法）
对
$$A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$$
做 SVD，观察奇异值大小。
- 大奇异值对应主要信息方向。

### 易错点
- 把特征值与奇异值混淆。
- `U,\Sigma,V` 形状写错。

### 与00_factor关联
- [[Spectral Decomposition|谱分解]]
- [[Matrix Rank|矩阵秩]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[3,0],[4,5]], dtype=float)
U, s, VT = np.linalg.svd(A)
print("singular values:", s)
print("reconstruct:\n", U @ np.diag(s) @ VT)
```

## Session 3.6 Linear transformations and their matrices

### 课程目标
- 理解“矩阵是线性变换在某组基下的表示”。
- 会从变换定义写出矩阵。

### 核心定义
- 线性变换：
  $$T(x)=Ax$$
- 基变换后矩阵：
  $$[T]_\mathcal{B}=P^{-1}AP$$

### 关键推导
- 先看 $T(e_i)$（基向量像），再拼成矩阵列。
- 不同基下矩阵不同，但变换本体不变。

### 例题（含解法）
$T(x,y)=(x+y,y)$。
- $T(e_1)=(1,0)$，$T(e_2)=(1,1)$。
- 标准基矩阵：
  $$A=\begin{bmatrix}1&1\\0&1\end{bmatrix}$$

### 易错点
- 把仿射变换误当线性变换（有常数项则不是）。
- 基变换公式左右顺序写错。

### 与00_factor关联
- [[Linear Systems|线性系统]]
- [[Matrix Operations|矩阵运算]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,1],[0,1]], dtype=float)
x = np.array([2,3], dtype=float)
print("T(x)=", A @ x)
```

## Session 3.7 Change of basis; image compression

### 课程目标
- 理解换基（change of basis）带来的表示简化。
- 理解图像压缩中的低秩思想。

### 核心定义
- 坐标变换：
  $$x=Pc,\ c=P^{-1}x$$
- 截断 SVD：
  $$A_k=\sum_{i=1}^k\sigma_i u_i v_i^T$$

### 关键推导
- 换基后，信息可以集中到少数坐标。
- 压缩就是保留主方向、舍弃小奇异值方向。

### 例题（含解法）
对矩阵做 rank-1 近似，比较原矩阵与近似矩阵误差。
- 误差显著下降主要由第一奇异值方向贡献。

### 易错点
- 截断过度导致信息损失严重。
- 忘记比较误差指标（Frobenius norm）。

### 与00_factor关联
- [[PCA|主成分分析]]
- [[Matrix Rank|矩阵秩]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,2,3],[2,4,6],[1,1,1]], dtype=float)
U, s, VT = np.linalg.svd(A)
A1 = np.outer(U[:,0]*s[0], VT[0,:])
print("rank(A)=", np.linalg.matrix_rank(A))
print("rank(A1)=", np.linalg.matrix_rank(A1))
print("||A-A1||_F=", np.linalg.norm(A-A1, 'fro'))
```

## Session 3.8 Left and right inverses; pseudoinverse

### 课程目标
- 区分左逆（left inverse）与右逆（right inverse）。
- 掌握 Moore-Penrose 伪逆（pseudoinverse）用途。

### 核心定义
- 高矩阵（列满秩）可能有左逆。
- 宽矩阵（行满秩）可能有右逆。
- 伪逆：
  $$A^+=V\Sigma^+U^T$$

### 关键推导
- 超定系统：$\hat{x}=A^+b$ 给最小二乘解。
- 欠定系统：$\hat{x}=A^+b$ 给最小范数解。

### 例题（含解法）
$$A=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\ b=\begin{bmatrix}1\\2\\2\end{bmatrix}$$
- `x = pinv(A) b` 是最小二乘意义下最优解。

### 易错点
- 把伪逆当普通逆使用场景。
- 忽略秩亏导致数值不稳定问题。

### 与00_factor关联
- [[Matrix Inverse|逆矩阵]]
- [[OLS Basics|OLS基础]]
- [[Matrix Rank|矩阵秩]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,0],[0,1],[1,1]], dtype=float)
b = np.array([1,2,2], dtype=float)
x = np.linalg.pinv(A) @ b
print("x_hat =", x)
print("residual norm =", np.linalg.norm(A @ x - b))
```

## Session 3.9 Exam 3 review

### 课程目标
- 复盘 Unit3 的核心模型与题型。
- 构建考试答题时的“判别-工具”快速路径。

### 核心定义
- 高频题型：
  - 对称/正定判定
  - 相似与 Jordan
  - SVD 与低秩近似
  - 伪逆与最小二乘

### 关键推导
- 解题顺序建议：
  1. 判定矩阵类型（对称？可对角化？秩亏？）
  2. 选工具（谱分解/Jordan/SVD/pinv）
  3. 给结构结论 + 数值验证

### 例题（含解法）
给矩阵 `A`：要求判断是否正定、是否可对角化、是否需要伪逆。
- 答案模板：
  - 看对称性与特征值号
  - 看特征向量个数
  - 看秩与方程形状

### 易错点
- 看到特征值就直接结论，不检查前提。
- 把“有解”与“最好解（least-squares best）”混淆。

### 与00_factor关联
- [[Symmetric Matrix|对称矩阵]]
- [[Positive Definite Matrix|正定矩阵]]
- [[Spectral Decomposition|谱分解]]
- [[Matrix Inverse|逆矩阵]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[2,1],[1,2]], dtype=float)
print("eigvals:", np.linalg.eigvals(A))
print("is symmetric:", np.allclose(A, A.T))
print("pinv(A):\n", np.linalg.pinv(A))
```
