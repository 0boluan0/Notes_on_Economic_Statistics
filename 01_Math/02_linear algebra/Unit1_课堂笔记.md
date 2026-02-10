---
aliases: []
tags:
  - linear-algebra
  - MIT-18.06SC
  - unit1
date: 2026-02-09
科目: Math
---

# Unit1 课堂笔记（MIT 18.06SC）

> 对应资料顺序：`MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf` 到 `Ses1.14sum.pdf`。
> 本笔记以课堂顺序展开，吸收 `01_matrices and Gaussian Elimination.md` 与 `02_vector spaces and subspace.md` 的已有内容。

## Session 1.1 The geometry of linear equations

### 课程目标
- 把“解方程组”统一成 `Ax=b`。
- 同时理解 row picture（方程交点）和 column picture（列向量线性组合）。
- 建立“有解/无解/多解”的几何判据。

### 核心定义
- 线性方程组：每个方程都是未知量的线性组合。
- 矩阵形式：
  $$Ax=b$$
- column picture：`b` 能否由 `A` 的列向量线性组合得到。

### 关键推导
- 若 $A=[a_1,a_2,\dots,a_n]$，则
  $$Ax=x_1a_1+x_2a_2+\cdots+x_na_n$$
- 所以“求解 x”就是“找系数使列向量组合等于 b”。
- 这和你之前笔记中的 Row/Column 两种写法是一致的，只是表达更统一。

### 例题（含解法）
已知
$$
\begin{cases}
2x-y=0\\
-x+2y-z=-1\\
-3y+4z=4
\end{cases}
$$
写成 $Ax=b$ 并判断是否可能有解。
- 解：
  $$
  A=\begin{bmatrix}
 2&-1&0\\-1&2&-1\\0&-3&4
 \end{bmatrix},\quad
 b=\begin{bmatrix}0\\-1\\4\end{bmatrix}
  $$
- 只要 $b\in C(A)$ 就有解；后续通过消元可以具体判定。

### 易错点
- 把“有交点”直接当“唯一解”。
- 忽略矩阵形状（`m×n`）就讨论结论。
- 只算数值，不看几何结构。

### 与00_factor关联
- [[Linear Algebra-hub|线性代数-hub]]
- [[Column Space|列空间]]
- [[Linear Combination|线性组合]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[2, -1, 0], [-1, 2, -1], [0, -3, 4]], dtype=float)
b = np.array([0, -1, 4], dtype=float)
x = np.linalg.solve(A, b)
print("x =", x)
print("A@x =", A @ x)
```

## Session 1.2 Elimination with matrices

### 课程目标
- 掌握高斯消元（Gaussian elimination）流程。
- 理解主元（pivot）与行变换的意义。
- 会从增广矩阵判断无解/多解。

### 核心定义
- 初等行变换：交换、倍乘、行加。
- 主元列（pivot columns）：提供独立约束。
- 阶梯形矩阵（echelon form）：便于回代。

### 关键推导
- 用消元矩阵记步骤：
  $$E_k\cdots E_2E_1A=U$$
- 对增广矩阵同时行变换：
  $$[A\mid b]\to[U\mid c]$$
- 若出现 `[0 0 ... 0 | 非0]`，则无解。

### 例题（含解法）
$$
\begin{cases}
x+2y+z=2\\
2x+5y+3z=5\\
x+y+z=1
\end{cases}
$$
- 消元：`R2 <- R2-2R1`, `R3 <- R3-R1`。
- 得到上三角后回代：$y=1, z=0, x=0$。

### 易错点
- 求解 `Ax=b` 时把列变换混进来。
- 主元为0不换行。
- 回代时符号错误。

### 与00_factor关联
- [[Matrix Operations|矩阵运算]]
- [[Matrix Rank|矩阵秩]]
- [[Linear Systems|线性系统]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1, 2, 1], [2, 5, 3], [1, 1, 1]], dtype=float)
b = np.array([2, 5, 1], dtype=float)
Ab = np.c_[A, b]
print("augmented matrix:\n", Ab)
print("solution:", np.linalg.solve(A, b))
```

## Session 1.3 Multiplication and inverse matrices

### 课程目标
- 理解矩阵乘法是线性变换复合。
- 理解逆矩阵的代数与几何意义。
- 能通过增广矩阵求 $A^{-1}$。

### 核心定义
- 乘法复合：$(AB)x=A(Bx)$。
- 逆矩阵：$A^{-1}A=I=AA^{-1}$（仅方阵且可逆时）。
- 可逆等价于 `det(A) ≠ 0`。

### 关键推导
- 若 $A$ 可逆，则 $Ax=b$ 唯一解为：
  $$x=A^{-1}b$$
- 用行变换：
  $$[A\mid I]\to[I\mid A^{-1}]$$

### 例题（含解法）
对
$$A=\begin{bmatrix}1&3\\2&7\end{bmatrix}$$
求逆。
- 结果：
  $$A^{-1}=\begin{bmatrix}7&-3\\-2&1\end{bmatrix}$$
- 验证：$AA^{-1}=I$。

### 易错点
- 误以为 $AB=BA$。
- 非方阵讨论双侧逆。
- 把“左逆存在”直接当“可逆”。

### 与00_factor关联
- [[Matrix Inverse|逆矩阵]]
- [[Determinant|行列式]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1, 3], [2, 7]], dtype=float)
A_inv = np.linalg.inv(A)
print("A_inv:\n", A_inv)
print("A @ A_inv:\n", A @ A_inv)
```

## Session 1.4 Factorization into A = LU

### 课程目标
- 掌握 LU 分解的计算逻辑。
- 理解 `L` 记录消元过程、`U` 记录结果。
- 会用 `LU` 快速解多个右端向量。

### 核心定义
- 无换行时：
  $$A=LU$$
- 求解流程：先 `Ly=b`，再 `Ux=y`。

### 关键推导
- 每一步消元乘子（multiplier）写入 `L` 的下三角位置。
- `L` 对角线取1，`U` 为上三角。

### 例题（含解法）
$$A=\begin{bmatrix}2&1\\4&3\end{bmatrix}$$
- 消元乘子 $m_{21}=2$。
- 得
  $$L=\begin{bmatrix}1&0\\2&1\end{bmatrix},\ U=\begin{bmatrix}2&1\\0&1\end{bmatrix}$$
- 验证 $LU=A$。

### 易错点
- 忘记 `L` 对角线为1。
- 把 `L` 与 `U` 顺序写反。
- 有行交换时仍写 `A=LU`。

### 与00_factor关联
- [[LU Decomposition|LU分解]]
- [[Matrix Operations|矩阵运算]]

### 代码块（完整保留）
```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 1], [4, 3]], dtype=float)
P, L, U = lu(A)
print("P:\n", P)
print("L:\n", L)
print("U:\n", U)
print("P@A:\n", P @ A)
print("L@U:\n", L @ U)
```

## Session 1.5 Transposes, permutations, spaces Rn

### 课程目标
- 理解转置与置换矩阵的作用。
- 掌握含换行消元时的标准形式 `PA=LU`。
- 明确 `R^n` 向量空间语境。

### 核心定义
- 转置：$(AB)^T=B^TA^T$。
- 置换矩阵（Permutation matrix）：单位矩阵换行。
- 含换行的分解：
  $$PA=LU$$

### 关键推导
- 行交换可由左乘 `P` 表示。
- 置换不改变线性系统解集，只改变方程顺序。

### 例题（含解法）
$$P=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$$
- 计算：
  $$PA=\begin{bmatrix}3&4\\1&2\end{bmatrix}$$
- 且 $P^{-1}=P^T=P$。

### 易错点
- 把 $(AB)^T$ 写成 $A^TB^T$。
- 混淆行交换和列交换。
- 把置换矩阵当一般随机矩阵。

### 与00_factor关联
- [[Permutation Matrix|置换矩阵]]
- [[Vector Space|向量空间]]

### 代码块（完整保留）
```python
import numpy as np

P = np.array([[0, 1], [1, 0]])
A = np.array([[1, 2], [3, 4]])
print("P@A=\n", P @ A)
print("(A.T).T=\n", (A.T).T)
```

## Session 1.6 Column space and nullspace

### 课程目标
- 理解列空间 `C(A)` 与零空间 `N(A)`。
- 掌握“`Ax=b` 有解当且仅当 `b∈C(A)`”。
- 建立维数直觉。

### 核心定义
- 列空间：`A` 列向量线性组合的全体。
- 零空间：满足 `Ax=0` 的所有向量。
- 都是子空间（subspace）。

### 关键推导
- 你原笔记中的关键句：`Ax=b` 只有在 `b` 属于列空间时有解。
- 通过消元确定 pivot columns，即列空间的基来源。

### 例题（含解法）
$$A=\begin{bmatrix}1&1&2\\2&1&3\\3&1&4\\4&1&5\end{bmatrix}$$
- 第3列 = 第1列 + 第2列，列空间维数不变。
- 所以 `C(A)` 由前两列张成，是 `R^4` 中二维子空间。

### 易错点
- 把“列空间在 `R^m`”误记成 `R^n`。
- 用行变换后的列直接当原列空间基（需谨慎）。
- 把“零空间只有0”当所有矩阵都成立。

### 与00_factor关联
- [[Column Space|列空间]]
- [[Null Space|零空间]]
- [[Subspace|子空间]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,1,2],[2,1,3],[3,1,4],[4,1,5]], dtype=float)
rank = np.linalg.matrix_rank(A)
print("rank(A)=", rank)
print("col3 - col1 - col2 =", A[:,2] - A[:,0] - A[:,1])
```

## Session 1.7 Solving Ax = 0: pivot variables, special solutions

### 课程目标
- 会通过 RREF 找主变量与自由变量。
- 理解特解（special solutions）构造法。
- 会写齐次方程通解。

### 核心定义
- 主变量：对应主元列。
- 自由变量：可自由指定。
- 特解：每次让一个自由变量为1，其余自由变量为0所得解。

### 关键推导
- 若 `n` 个未知量、`r` 个主元，则自由变量个数 `n-r`。
- 齐次通解是所有特解的线性组合。

### 例题（含解法）
$$
A=\begin{bmatrix}1&2&3\\2&4&6\\2&6&8\\2&8&10\end{bmatrix}
$$
- 消元后有2个主元、1个自由变量。
- 令 $x_3=1$ 得特解 $(-1,-1,1)^T$。
- 通解：$x=t(-1,-1,1)^T$。

### 易错点
- 把自由变量也当回代目标。
- 忘记通解参数（例如 `t`）。
- 把一个特解误当全部解。

### 与00_factor关联
- [[Null Space|零空间]]
- [[Matrix Rank|矩阵秩]]

### 代码块（完整保留）
```python
import sympy as sp

A = sp.Matrix([[1,2,3],[2,4,6],[2,6,8],[2,8,10]])
print("rref:", A.rref())
print("nullspace:", A.nullspace())
```

## Session 1.8 Solving Ax = b: row reduced form R

### 课程目标
- 在非齐次系统中区分“特解 + 齐次通解”。
- 用简化行阶梯形（RREF）读解结构。
- 会判断无解/唯一解/无穷多解。

### 核心定义
- 非齐次通解：
  $$x=x_p+x_n,\quad Ax_p=b,\ Ax_n=0$$
- RREF 直接给线性关系。

### 关键推导
- `Ax=b` 的全部解 = 一个特解 + 零空间平移。
- 这也是你原笔记中“特解线 + 零空间方向”的几何表达。

### 例题（含解法）
设
$$
A=\begin{bmatrix}1&2&1\\2&4&2\end{bmatrix},\ b=\begin{bmatrix}3\\6\end{bmatrix}
$$
- 第二行与第一行等价，系统一致。
- 一组特解：$x_p=(3,0,0)^T$。
- 零空间方向来自 `x1+2x2+x3=0`，故有无穷多解。

### 易错点
- 看到秩不足就断言无解（还要看增广列）。
- 特解选择后忘记加上零空间部分。
- 把 `R` 当原矩阵去解释列空间。

### 与00_factor关联
- [[Linear Systems|线性系统]]
- [[Null Space|零空间]]
- [[Column Space|列空间]]

### 代码块（完整保留）
```python
import sympy as sp

A = sp.Matrix([[1,2,1],[2,4,2]])
b = sp.Matrix([3,6])
print("A.rank =", A.rank(), "aug.rank =", A.row_join(b).rank())
print("parametric solution:", sp.linsolve((A,b)))
```

## Session 1.9 Independence, basis, and dimension

### 课程目标
- 区分线性无关/相关。
- 理解基（basis）是“最小生成集”。
- 理解维数（dimension）是自由方向数量。

### 核心定义
- 线性无关：只有零系数组合为零向量。
- 基：既生成空间又线性无关。
- 维数：任意基的向量个数。

### 关键推导
- `r` 个主元列对应一个无关集合。
- 对于子空间，任意两组基长度相同（维数定义成立）。

### 例题（含解法）
向量组 $(1,0,1),(0,1,1),(1,1,2)$。
- 第3个 = 前两个之和，故相关。
- 去掉第3个后前两个构成该子空间一组基，维数为2。

### 易错点
- “向量个数少于维数”与“必无关”混淆。
- 把生成集直接当基，不检验无关。
- 在不同空间里混比维数。

### 与00_factor关联
- [[Linear Independence|线性无关]]
- [[Basis Risk|基]]
- [[Vector Space|向量空间]]

### 代码块（完整保留）
```python
import sympy as sp

V = sp.Matrix([[1,0,1],[0,1,1],[1,1,2]])
print("rank =", V.rank())
print("columns are independent?", V.rank() == V.shape[1])
```

## Session 1.10 The four fundamental subspaces

### 课程目标
- 掌握四大基本子空间的定义与所在空间。
- 理解正交关系与维数关系。
- 把 rank-nullity 定理放进总框架。

### 核心定义
- `C(A)`：列空间，位于 `R^m`。
- `N(A)`：零空间，位于 `R^n`。
- `C(A^T)`：行空间，位于 `R^n`。
- `N(A^T)`：左零空间，位于 `R^m`。

### 关键推导
- 维数关系：
  $$\dim C(A)=\dim C(A^T)=r$$
  $$\dim N(A)=n-r,\ \dim N(A^T)=m-r$$
- 正交关系：`C(A)` 与 `N(A^T)` 互为正交补；`C(A^T)` 与 `N(A)` 互为正交补。

### 例题（含解法）
若 `A` 是 `4×3` 且 `rank=2`：
- `dim C(A)=2`，`dim N(A)=1`。
- `dim C(A^T)=2`，`dim N(A^T)=2`。

### 易错点
- 忘记每个子空间属于哪个 `R^k`。
- 只记秩，不记 `n-r` 与 `m-r`。
- 混淆零空间和左零空间。

### 与00_factor关联
- [[Column Space|列空间]]
- [[Null Space|零空间]]
- [[Matrix Rank|矩阵秩]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,2,0],[2,4,1],[0,0,1],[1,2,1]], dtype=float)
r = np.linalg.matrix_rank(A)
m, n = A.shape
print("rank", r)
print("dim N(A)=", n-r, "dim N(A^T)=", m-r)
```

## Session 1.11 Matrix spaces; rank 1; small world graphs

### 课程目标
- 把“矩阵也可以作为向量”这个观点建立起来。
- 理解 rank-1 矩阵的结构与用途。
- 初步连接图网络（small world graphs）与线性代数。

### 核心定义
- 矩阵空间：所有 `m×n` 矩阵构成维数 `mn` 的向量空间。
- rank-1 矩阵可写成外积：
  $$A=uv^T$$

### 关键推导
- rank-1 矩阵每一列都与 `u` 共线，每一行都与 `v^T` 共线。
- 低秩结构是后续 SVD/压缩的先导。

### 例题（含解法）
令 $u=(1,2,3)^T$, $v=(2,-1)^T$，
$$A=uv^T=\begin{bmatrix}2&-1\\4&-2\\6&-3\end{bmatrix}$$
- 直接看出第2列是第1列的 `-1/2` 倍，秩为1。

### 易错点
- 把元素乘法误当外积。
- 只看一行/一列就判断秩。
- 忽略数值误差导致“近似低秩”判断错误。

### 与00_factor关联
- [[Matrix Rank|矩阵秩]]
- [[Spectral Decomposition|谱分解]]

### 代码块（完整保留）
```python
import numpy as np

u = np.array([[1],[2],[3]], dtype=float)
v = np.array([[2,-1]], dtype=float)
A = u @ v
print(A)
print("rank:", np.linalg.matrix_rank(A))
```

## Session 1.12 Graphs, networks, incidence matrices

### 课程目标
- 理解图（graph）的关联矩阵（incidence matrix）表示。
- 把网络流约束写成线性方程。
- 理解图连通性与矩阵秩的关系。

### 核心定义
- 关联矩阵 `B`：节点×边，列表示一条有向边在起点/终点的 `-1/+1`。
- 流守恒：
  $$Bf=s$$

### 关键推导
- 若图有 `c` 个连通分量，`rank(B)=n-c`（`n` 是节点数）。
- `N(B^T)` 与连通分量结构有关。

### 例题（含解法）
3个节点，边 `1→2, 2→3, 1→3`：
$$
B=\begin{bmatrix}
-1&0&-1\\
1&-1&0\\
0&1&1
\end{bmatrix}
$$
- 可验证 `rank(B)=2=n-1`（连通图）。

### 易错点
- 边方向约定不一致。
- 行列定义互换导致符号全错。
- 把邻接矩阵与关联矩阵混淆。

### 与00_factor关联
- [[Matrix Rank|矩阵秩]]
- [[Linear Systems|线性系统]]

### 代码块（完整保留）
```python
import numpy as np

B = np.array([[-1,0,-1],[1,-1,0],[0,1,1]], dtype=float)
print("B=\n", B)
print("rank(B)=", np.linalg.matrix_rank(B))
```

## Session 1.13 An overview of key ideas

### 课程目标
- 汇总 Unit1 主线：`Ax=b`、消元、子空间、秩。
- 建立“结构优先于计算”的复习方式。

### 核心定义
- 线性系统不是孤立题，而是同一结构在不同表述下的重复出现。
- 核心对象：`C(A)`, `N(A)`, `rank`, `pivot`, `basis`。

### 关键推导
- 从 `Ax=b` 出发：
  - 是否有解：看 `b` 是否在 `C(A)`。
  - 解有多唯一：看 `N(A)` 维数。
- 一句话：列空间决定“可达”，零空间决定“自由度”。

### 例题（含解法）
设 `rank(A)=r`，`A` 为 `m×n`。
- 若 `b∈C(A)`，解集维数是 `n-r`。
- 若 `b∉C(A)`，无解。

### 易错点
- 复习时只刷算术，不刷结构关系图。
- 把“解法步骤”与“判定逻辑”混在一起。

### 与00_factor关联
- [[Column Space|列空间]]
- [[Null Space|零空间]]
- [[Matrix Rank|矩阵秩]]
- [[Linear Independence|线性无关]]

### 代码块（完整保留）
```python
import numpy as np

A = np.array([[1,2,3],[2,4,6],[1,1,1]], dtype=float)
r = np.linalg.matrix_rank(A)
print("rank(A)=", r)
print("unknowns n=", A.shape[1], "=> solution dimension for Ax=0 is", A.shape[1]-r)
```

## Session 1.14 Exam 1 review

### 课程目标
- 用“考前清单”压缩 Unit1 全部知识点。
- 明确题型与对应方法。

### 核心定义
- 高频题型：
  - 消元与 RREF
  - `Ax=0` 与 `Ax=b`
  - 子空间/维数/秩
  - `A^{-1}`, `LU`, `PA=LU`

### 关键推导
- 快速流程：
  1. 先看矩阵形状和秩。
  2. 再看增广列是否一致。
  3. 最后写“特解 + 零空间通解”。

### 例题（含解法）
给 `A` 与 `b`，要求：是否有解、解结构、是否唯一。
- 标准答案模板：
  - `rank(A)=?`, `rank([A|b])=?`
  - 若一致则有解，否则无解
  - 若有解，维数 `n-rank(A)`

### 易错点
- rank 比较少写一个。
- 只给数值答案，不给结构说明。
- 忘记检查结果回代。

### 与00_factor关联
- [[Linear Systems|线性系统]]
- [[Matrix Inverse|逆矩阵]]
- [[LU Decomposition|LU分解]]
- [[Permutation Matrix|置换矩阵]]

### 代码块（完整保留）
```python
import sympy as sp

A = sp.Matrix([[1,2,1],[2,4,2],[1,1,1]])
b = sp.Matrix([1,2,0])
print("rank(A)=", A.rank())
print("rank([A|b])=", A.row_join(b).rank())
print("solution set:", sp.linsolve((A,b)))
```
