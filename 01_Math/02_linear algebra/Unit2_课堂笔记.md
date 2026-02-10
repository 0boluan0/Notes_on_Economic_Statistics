---
aliases: []
tags:
  - linear-algebra
  - MIT-18.06SC
  - unit2
date: 2026-02-09
科目: Math
---

# Unit2 课堂笔记（MIT 18.06SC）

> 对应资料顺序：`MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf` 到 `Ses2.12sum.pdf`。
> 本单元主线：正交（orthogonality）→ 投影（projection）→ 最小二乘（least squares）→ 行列式与特征值（determinant & eigenvalues）→ 动态系统（differential equation / Markov）。

## Session 2.1 Orthogonal vectors and subspaces

### 课程目标
- 理解正交向量与正交子空间的定义。
- 理解 `S` 与 `S^\perp` 的分解思想。
- 会用内积判定正交关系。

### 核心定义
- 两向量正交：
  $$u^Tv=0$$
- 子空间正交补：
  $$S^\perp=\{x\mid x^Ts=0,\ \forall s\in S\}$$

### 关键推导
- 若 $S$ 是 $\mathbb{R}^n$ 的子空间，则
  $$\mathbb{R}^n=S\oplus S^\perp$$
- 这意味着任意向量都可唯一写成“在 `S` 中的部分 + 在 `S^\perp` 中的部分”。

### 例题（含解法）
给 $u=(1,2,2)^T$, $v=(2,-1,0)^T$。
- 计算 $u^Tv=1\cdot2+2\cdot(-1)+2\cdot0=0$。
- 所以两向量正交。

### 易错点
- 把二维“垂直”的直观生搬硬套到高维，忘了内积才是定义。
- 忘记在同一内积定义下判定（默认欧氏内积）。

### 与00_factor关联
- [[Vector Space|向量空间]]
- [[Subspace|子空间]]
- [[Linear Combination|线性组合]]

### 代码块（完整保留）
```python
import numpy as np

u = np.array([1, 2, 2], dtype=float)
v = np.array([2, -1, 0], dtype=float)
print("u·v =", u @ v)
```

## Session 2.2 Projections onto subspaces

### 课程目标
- 掌握向量投影到一维子空间与高维子空间的基本公式。
- 理解“最短距离逼近”与“残差正交”的等价性。

### 核心定义
- 投影到方向 $a$：
  $$p=\frac{a^Tb}{a^Ta}a$$
- 残差：
  $$e=b-p,\quad a^Te=0$$

### 关键推导
- 最优化问题
  $$\min_x\|b-ax\|_2^2$$
  的一阶条件给出
  $$a^T(b-ax)=0$$
- 即投影残差与投影子空间正交。

### 例题（含解法）
将 $b=(3,1)^T$ 投影到 $a=(1,1)^T$。
- 系数：$c=\frac{a^Tb}{a^Ta}=\frac{4}{2}=2$。
- 投影：$p=2a=(2,2)^T$。
- 残差：$e=(1,-1)^T$，满足 $a^Te=0$。

### 易错点
- 分母写成 $\|a\|$ 而不是 $a^Ta$。
- 忘记验证残差正交条件。

### 与00_factor关联
- [[Column Space|列空间]]
- [[Matrix Operations|矩阵运算]]

### 代码块（完整保留）
```python
import numpy as np

a = np.array([1, 1], dtype=float)
b = np.array([3, 1], dtype=float)
p = (a @ b) / (a @ a) * a
e = b - p
print("projection p =", p)
print("residual e =", e)
print("a·e =", a @ e)
```

## Session 2.3 Projection matrices and least squares

### 课程目标
- 理解最小二乘是“投影到列空间”问题。
- 会写正规方程与投影矩阵。

### 核心定义
- 正规方程（normal equations）：
  $$A^TA\hat{x}=A^Tb$$
- 投影矩阵：
  $$P=A(A^TA)^{-1}A^T$$

### 关键推导
- 最小化 $\|b-Ax\|_2^2$，对 $x$ 求导得正规方程。
- 若 $A$ 列满秩，$A^TA$ 可逆，唯一最小二乘解存在。
- 投影结果是 $\hat{b}=A\hat{x}=Pb$。

### 例题（含解法）
拟合常数模型 $y\approx c$ 到数据 $(1,2,4)$。
- $A=[1,1,1]^T$，$b=[1,2,4]^T$。
- 正规方程：$3c=7$，得 $c=7/3$。

### 易错点
- 把“最小二乘解”误认为“精确解”。
- $A^TA$ 不可逆时仍硬求逆。

### 与00_factor关联
- [[OLS Basics|OLS基础]]
- [[Matrix Inverse|逆矩阵]]
- [[Column Space|列空间]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1.0], [1.0], [1.0]])
b = np.array([1.0, 2.0, 4.0])
xt = np.linalg.solve(A.T @ A, A.T @ b)
print("least squares c =", xt[0])
print("fitted =", A @ xt)
```

## Session 2.4 Orthogonal matrices and Gram-Schmidt

### 课程目标
- 理解正交矩阵保持长度与角度。
- 掌握 Gram-Schmidt 正交化流程。

### 核心定义
- 正交矩阵：
  $$Q^TQ=I$$
- QR 分解：
  $$A=QR$$

### 关键推导
- Gram-Schmidt 逐步把新向量减去在已构造正交基上的投影。
- 标准化后得到正交标准基（orthonormal basis）。

### 例题（含解法）
$a_1=(1,1,0)^T$, $a_2=(1,0,1)^T$。
- $q_1=a_1/\|a_1\|$。
- $u_2=a_2-(q_1^Ta_2)q_1$。
- $q_2=u_2/\|u_2\|$。

### 易错点
- 只减去最近一个投影，漏掉前面方向。
- 忘记归一化。

### 与00_factor关联
- [[Orthogonality|正交性]]
- [[Vector Space|向量空间]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
Q, R = np.linalg.qr(A)
print("Q^T Q =\n", Q.T @ Q)
print("R =\n", R)
print("Q R =\n", Q @ R)
```

## Session 2.5 Properties of determinants

### 课程目标
- 掌握行列式的核心性质。
- 理解行列式与可逆性、体积缩放的关系。

### 核心定义
- 行列式乘法法则：
  $$\det(AB)=\det(A)\det(B)$$
- 转置不变：
  $$\det(A^T)=\det(A)$$

### 关键推导
- 行交换使行列式变号。
- 某行乘常数 $c$，行列式乘 $c$。
- 某行加上另一行倍数，行列式不变。

### 例题（含解法）
$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$$
- $\det(A)=1\cdot4-2\cdot3=-2$。
- 因此可逆，且线性变换会翻转方向（符号为负）。

### 易错点
- 把“行加倍数”误当改变行列式。
- 把非方阵也讨论行列式。

### 与00_factor关联
- [[Determinant|行列式]]
- [[Matrix Inverse|逆矩阵]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=float)
print("det(A) =", np.linalg.det(A))
```

## Session 2.6 Determinant formulas and cofactors

### 课程目标
- 会用余子式（cofactor）展开计算行列式。
- 理解伴随矩阵与逆矩阵公式联系。

### 核心定义
- cofactor 展开：
  $$\det(A)=\sum_j a_{ij}C_{ij}$$
- 伴随公式：
  $$A^{-1}=\frac{1}{\det(A)}\operatorname{adj}(A)$$

### 关键推导
- $C_{ij}=(-1)^{i+j}M_{ij}$，其中 $M_{ij}$ 是余子式（minor）。
- 稀疏行/列展开时计算量可大幅下降。

### 例题（含解法）
$$A=\begin{bmatrix}2&1&0\\1&3&1\\0&1&2\end{bmatrix}$$
沿第一行展开：
$$\det(A)=2\det\begin{bmatrix}3&1\\1&2\end{bmatrix}-1\det\begin{bmatrix}1&1\\0&2\end{bmatrix}=10-2=8$$

### 易错点
- $(-1)^{i+j}$ 符号位错。
- 把 minor 与 cofactor 混淆。

### 与00_factor关联
- [[Determinant|行列式]]
- [[Matrix Inverse|逆矩阵]]

### 代码块（完整保留）
```python
import sympy as sp

A = sp.Matrix([[2,1,0],[1,3,1],[0,1,2]])
print("det(A)=", A.det())
print("adj(A)=\n", A.adjugate())
```

## Session 2.7 Cramer’s rule, inverse matrix, and volume

### 课程目标
- 理解克拉默法则（Cramer's rule）来源。
- 理解行列式体积比意义。

### 核心定义
- 对可逆方阵 $Ax=b$：
  $$x_i=\frac{\det(A_i(b))}{\det(A)}$$
- 其中 $A_i(b)$ 为把第 `i` 列替换为 `b` 后的矩阵。

### 关键推导
- 每个分子是“替换某方向后的体积”，分母是原体积。
- 体积比给出该方向坐标。

### 例题（含解法）
$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix},\ b=\begin{bmatrix}5\\11\end{bmatrix}$$
- $\det(A)=-2$。
- $\det(A_1)=-2$，$\det(A_2)=-4$。
- 故 $x_1=1,x_2=2$。

### 易错点
- 替换错列。
- 分母接近0时数值不稳定仍强算。

### 与00_factor关联
- [[Determinant|行列式]]
- [[Matrix Inverse|逆矩阵]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,2],[3,4]], dtype=float)
b = np.array([5,11], dtype=float)
A1 = A.copy(); A1[:,0] = b
A2 = A.copy(); A2[:,1] = b
x1 = np.linalg.det(A1)/np.linalg.det(A)
x2 = np.linalg.det(A2)/np.linalg.det(A)
print("x=", x1, x2)
```

## Session 2.8 Eigenvalues and eigenvectors

### 课程目标
- 掌握特征值/特征向量定义与求法。
- 理解特征方向“不变方向”含义。

### 核心定义
- 特征方程：
  $$Av=\lambda v$$
- 非零解存在条件：
  $$\det(A-\lambda I)=0$$

### 关键推导
- 解多项式得到特征值，再解 $(A-\lambda I)v=0$ 得特征向量。
- 若有足够线性无关特征向量，为后续对角化做准备。

### 例题（含解法）
$$A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$$
- 特征值：$\lambda_1=3,\lambda_2=1$。
- 对应特征向量可取 $(1,1)^T$ 与 $(1,-1)^T$。

### 易错点
- 把特征向量允许为零向量（不允许）。
- 解出特征值后不回代验证。

### 与00_factor关联
- [[Eigenvalues|特征值]]
- [[Eigenvectors|特征向量]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[2,1],[1,2]], dtype=float)
w, V = np.linalg.eig(A)
print("eigenvalues:", w)
print("eigenvectors:\n", V)
```

## Session 2.9 Diagonalization and powers of A

### 课程目标
- 理解对角化条件与用途。
- 会利用对角化快速计算 $A^k$。

### 核心定义
- 若 $A$ 有 `n` 个线性无关特征向量，则
  $$A=P\Lambda P^{-1}$$
- 从而
  $$A^k=P\Lambda^kP^{-1}$$

### 关键推导
- 对角矩阵幂只需逐元素幂：计算代价低。
- 迭代系统增长率由主特征值决定。

### 例题（含解法）
对角化矩阵
$$A=\begin{bmatrix}2&0\\0&3\end{bmatrix}$$
- $A^{10}=\operatorname{diag}(2^{10},3^{10})$。
- 若 $A$ 相似于该对角矩阵，也能同样快速求幂。

### 易错点
- 误以为任何矩阵都可对角化。
- $P^{-1}$ 与 $P^T$ 混用（仅正交矩阵时相同）。

### 与00_factor关联
- [[Eigenvalues|特征值]]
- [[Eigenvectors|特征向量]]
- [[Spectral Decomposition|谱分解]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[4,1],[2,3]], dtype=float)
w, V = np.linalg.eig(A)
D = np.diag(w)
A5 = V @ np.linalg.matrix_power(D, 5) @ np.linalg.inv(V)
print("A^5 (diagonalization) =\n", A5)
print("A^5 (direct) =\n", np.linalg.matrix_power(A, 5))
```

## Session 2.10 Differential equations and e^{At}

### 课程目标
- 把线性微分方程系统写成矩阵形式。
- 理解矩阵指数 $e^{At}$ 的求解意义。

### 核心定义
- 系统：
  $$x'(t)=Ax(t),\quad x(0)=x_0$$
- 解：
  $$x(t)=e^{At}x_0$$

### 关键推导
- 若 $A=P\Lambda P^{-1}$，则
  $$e^{At}=Pe^{\Lambda t}P^{-1}$$
- 对角情形下 $e^{\Lambda t}$ 很容易计算。

### 例题（含解法）
$$A=\begin{bmatrix}1&0\\0&-2\end{bmatrix},\ x_0=(1,1)^T$$
- $x(t)=(e^t,e^{-2t})^T$。
- 一个分量增长，一个分量衰减。

### 易错点
- 把 $e^{At}$ 错写成元素逐个指数。
- 初值代入错位。

### 与00_factor关联
- [[Differential Equation|微分方程]]
- [[Eigenvalues|特征值]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,0],[0,-2]], dtype=float)
x0 = np.array([1,1], dtype=float)
w, V = np.linalg.eig(A)
for t in [0, 0.5, 1.0]:
    expAt = V @ np.diag(np.exp(w * t)) @ np.linalg.inv(V)
    xt = expAt @ x0
    print(f"t={t}:", xt)
```

## Session 2.11 Markov matrices; Fourier series

### 课程目标
- 理解马尔可夫矩阵（Markov matrix）迭代。
- 初步理解傅里叶级数（Fourier series）是正交展开。

### 核心定义
- 马尔可夫矩阵列和（或行和）为1，元素非负。
- 状态更新：
  $$x_{k+1}=Px_k$$
- 傅里叶展开：函数在正交基上的系数分解。

### 关键推导
- 马尔可夫长期行为由特征值结构决定，稳态向量满足 $Px=x$。
- 傅里叶系数来自正交投影。

### 例题（含解法）
$$P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix},\ x_0=(1,0)^T$$
迭代后趋向稳态分布。

### 易错点
- 把概率向量归一化漏掉。
- 混淆“列随机”与“行随机”约定。

### 与00_factor关联
- [[Stationary Distribution|平稳分布]]
- [[Markov Chain|马尔可夫链]]
- [[Power Spectral Density|功率谱密度]]

### 代码块（完整保留）
```python
import numpy as np

P = np.array([[0.9, 0.2], [0.1, 0.8]], dtype=float)
x = np.array([1.0, 0.0])
for k in range(10):
    x = P @ x
print("after 10 steps:", x, "sum=", x.sum())
```

## Session 2.12 Exam 2 review

### 课程目标
- 复盘 Unit2 的题型和方法映射。
- 建立“正交-投影-特征分解”一条线的复习框架。

### 核心定义
- 高频题型：
  - 投影 / 最小二乘
  - 行列式性质与 cofactor
  - 特征值特征向量 / 对角化
  - $e^{At}$ 与 Markov 迭代

### 关键推导
- 复习主线：
  1. 几何语言（正交、投影）
  2. 代数语言（矩阵方程）
  3. 动态语言（迭代与微分方程）
- 重点是把同一结构在不同题目中识别出来。

### 例题（含解法）
给定 $A,b$：
- 判断是否做最小二乘、是否可对角化、是否需要行列式判定可逆。
- 按题型选工具，不机械套公式。

### 易错点
- 推导会写、题目不会分类。
- 把最小二乘题硬按精确解处理。
- 对角化条件没检查就直接用。

### 与00_factor关联
- [[Determinant|行列式]]
- [[Eigenvalues|特征值]]
- [[Eigenvectors|特征向量]]
- [[Matrix Inverse|逆矩阵]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[2,1],[1,2]], dtype=float)
b = np.array([1,3], dtype=float)
print("det(A)=", np.linalg.det(A))
print("solve Ax=b:", np.linalg.solve(A,b))
print("eig:", np.linalg.eig(A)[0])
```
