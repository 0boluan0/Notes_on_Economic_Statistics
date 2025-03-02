
# Matrix Algebra and Random Vectors

### 1. Basics of Matrix and Vector Algebra
- Definition of Vectors
- Vector Length and Direction
- Cosine of the Angle Between Two Vectors
- Scalar Multiplication and Unit Vectors
- Linear Dependence and Independence of Vectors
- Projection of a Vector

### 2. Matrices
- Definition and Notation
- Identity Matrix
- Diagonal Matrix
- Matrix Operations:
  - Scalar Multiplication
  - Matrix Addition
  - Matrix Multiplication
- Matrix Inverse
- Orthogonal Matrices
- Rank of a Matrix

### 3. Determinants
- Definition of Determinants
- Properties:
  - Determinant of Product of Matrices
  - Determinant of Inverse
  - Determinant of Diagonal Matrices
  - Determinants and Orthogonal Matrices

### 4. Eigenvalues and Eigenvectors
- Definitions
- Properties:
  - Spectral Representation
  - Trace and Determinant Using Eigenvalues

### 5. Positive Definite and Non-Negative Definite Matrices
- Definitions and Properties
- Quadratic Forms

### 6. Square Root of a Matrix
- Definition
- Symmetry and Properties
- Transformation of Quadratic Forms

### 7. Random Vectors and Random Matrices
- Definitions
- Mean and Variance-Covariance Matrix
- Linear Functions of Random Vectors

### 8. Correlation Matrix
- Definition and Properties
- Relationship with Variance-Covariance Matrix

---

# Basics of Matrix and Vector Algebra 

## 1. **向量的定义与表示**
   
   - 向量是包含 $n$ 个实数的数组：
     $$x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}, \quad x' = (x_1, x_2, \cdots, x_n)$$

## 2. **向量的长度**
   
   - 向量 $x$ 的长度公式：
     $$L_x = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} = \sqrt{x'x}$$

## 3. **向量间的夹角与余弦**
   
   - 两向量 $x$ 和 $y$ 的夹角 $\theta$ 的余弦公式：
     $$\cos\theta = \frac{x'y}{L_x L_y}$$
     - $\cos\theta = 1$：$x = cy, c > 0$
     - $\cos\theta = -1$：$x = cy, c < 0$
     - $\cos\theta = 0$：$x$ 和 $y$ 正交。

## 4. **向量的缩放与单位向量**
   
   - 缩放公式：
     $$cx = \begin{pmatrix} cx_1 \\ cx_2 \\ \vdots \\ cx_n \end{pmatrix}, \quad L_{cx} = |c|L_x$$
   
   - 单位向量公式：
     $$x^* = \frac{x}{L_x}$$
     单位向量的长度为 $1$，方向与 $x$ 相同。

## 5. **线性相关与无关**
   
   - 向量组 $\{x_1, x_2, \cdots, x_k\}$ 线性相关当且仅当：
     $$c_1x_1 + c_2x_2 + \cdots + c_kx_k = 0$$
     且系数 $c_1, c_2, \cdots, c_k$ 不全为零。
   - 若无此关系，则为线性无关。

## 6. **向量的投影**
   
   - 向量 $x$ 在 $y$ 上的投影公式：
     $$\text{Projection of } x \text{ on } y = \frac{x'y}{y'y}y$$
   - 投影的长度：
     $$\text{Length of projection} = \frac{|x'y|}{L_y} = L_x |\cos\theta|$$

## 7.正交向量

### **正交向量的定义**

在向量空间中，**正交向量**是指两个向量之间的**内积为零**的向量。这意味着：

$$

\mathbf{u} \cdot \mathbf{v} = 0

$$

其中：

• $\mathbf{u} = (u_1, u_2, \ldots, u_n)$
• $\mathbf{v} = (v_1, v_2, \ldots, v_n)$
• 内积定义为：

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
$$
如果内积等于 0，则称 $\mathbf{u}$ 和 $\mathbf{v}$ 是正交的。

### 标准正交向量

两向量的长度均为1且二者正交,则二者为标准正交向量.

### **正交向量的几何意义**

在二维或三维欧几里得空间中，正交向量的几何意义是这两个向量之间的夹角为 $90^\circ$。从向量的内积公式可以看出：

$$
\mathbf{u} \cdot \mathbf{v} = |\mathbf{u}| |\mathbf{v}| \cos\theta
$$

当 $\cos\theta = 0$ 时，夹角 $\theta = 90^\circ$。

### **正交向量的性质**

1. **正交向量之间没有投影关系**：

如果 $\mathbf{u}$ 和 $\mathbf{v}$ 正交，则 $\mathbf{u}$ 在 $\mathbf{v}$ 上的投影为 0，反之亦然。

2. **线性独立性**：

两个正交的非零向量必定线性无关。这是因为如果它们线性相关，其中一个必定可以表示为另一个的倍数，而倍数关系无法满足正交条件。

3. **正交基**：

在向量空间中，如果一组向量两两正交，并且非零，则这组向量构成正交基。

  

**例子**  

**二维空间**

假设 $\mathbf{u} = (1, 2)$，$\mathbf{v} = (-2, 1)$，计算其内积：

$$
\mathbf{u} \cdot \mathbf{v} = 1 \cdot (-2) + 2 \cdot 1 = -2 + 2 = 0
$$
因此，$\mathbf{u}$ 和 $\mathbf{v}$ 是正交的。

**三维空间**

假设 $\mathbf{u} = (1, -1, 0)$，$\mathbf{v} = (1, 1, 2)$，计算内积：

$$

\mathbf{u} \cdot \mathbf{v} = 1 \cdot 1 + (-1) \cdot 1 + 0 \cdot 2 = 1 - 1 + 0 = 0

$$

因此，$\mathbf{u}$ 和 $\mathbf{v}$ 是正交的。

---

# Matrices 

## 1. **矩阵的定义与表示**
  
   - 矩阵是 $n \times p$ 的数值数组，由 $n$ 行和 $p$ 列组成：
     $$
     A_{n \times p} =
     \begin{pmatrix}
     a_{11} & a_{12} & \cdots & a_{1p} \\
     a_{21} & a_{22} & \cdots & a_{2p} \\
     \vdots & \vdots & \ddots & \vdots \\
     a_{n1} & a_{n2} & \cdots & a_{np}
     \end{pmatrix}
     $$
   - 方阵：若 $n = p$，则称矩阵为方阵（Square Matrix）。

## 2. **特殊矩阵**
  
   - **单位矩阵 (Identity Matrix)**:
     $$I_p = \begin{pmatrix}
     1 & 0 & \cdots & 0 \\
     0 & 1 & \cdots & 0 \\
     \vdots & \vdots & \ddots & \vdots \\
     0 & 0 & \cdots & 1
     \end{pmatrix}$$
     对角线上全为 $1$，其余元素为 $0$。
   - **对角矩阵 (Diagonal Matrix)**:
     $$A = \begin{pmatrix}
     a_{11} & 0 & \cdots & 0 \\
     0 & a_{22} & \cdots & 0 \\
     \vdots & \vdots & \ddots & \vdots \\
     0 & 0 & \cdots & a_{pp}
     \end{pmatrix}$$
     非对角线元素为 $0$。

## 3. **矩阵运算**
   
   - **标量乘法 (Scalar Multiplication)**:
     $$c \cdot A = \begin{pmatrix}
     c \cdot a_{11} & c \cdot a_{12} & \cdots & c \cdot a_{1p} \\
     c \cdot a_{21} & c \cdot a_{22} & \cdots & c \cdot a_{2p} \\
     \vdots & \vdots & \ddots & \vdots \\
     c \cdot a_{n1} & c \cdot a_{n2} & \cdots & c \cdot a_{np}
     \end{pmatrix}$$
   - **矩阵加法 (Matrix Addition)**:
     若 $A_{n \times p}$ 和 $B_{n \times p}$，则：
     $$A + B = \begin{pmatrix}
     a_{11} + b_{11} & \cdots & a_{1p} + b_{1p} \\
     \vdots & \ddots & \vdots \\
     a_{n1} + b_{n1} & \cdots & a_{np} + b_{np}
     \end{pmatrix}$$
   - **矩阵乘法 (Matrix Multiplication)**:
     若 $A_{n \times p}$ 和 $B_{p \times m}$，则 $C = A \cdot B$，$C_{n \times m}$ 的元素为：
     $$c_{ik} = \sum_{j=1}^p a_{ij} b_{jk}$$
 
## 4. **矩阵的转置**
   
   - 矩阵转置将行变为列：
     $$
        A =
     \begin{pmatrix}
     a_{11} & a_{12} & \cdots & a_{1p} \\
     a_{21} & a_{22} & \cdots & a_{2p} \\
     \vdots & \vdots & \ddots & \vdots \\
     a_{n1} & a_{n2} & \cdots & a_{np}
     \end{pmatrix} 
     A' = \begin{pmatrix}
     a_{11} & a_{21} & \cdots & a_{n1} \\
     a_{12} & a_{22} & \cdots & a_{n2} \\
     \vdots & \vdots & \ddots & \vdots \\
     a_{1p} & a_{2p} & \cdots & a_{np}
     \end{pmatrix}$$

特别的,对称矩阵的转置等于它本身

## 5. **矩阵的逆**
   
   - 若方阵 $A$ 存在矩阵 $B$，使得：
     $$AB = BA = I_p$$
     则 $B$ 是 $A$ 的逆，记为 $A^{-1}$。
   - $A$ 的逆存在的条件：列向量线性无关，即 $A$ 非奇异。

对角矩阵的逆矩阵好求：

$$  
D = \begin{bmatrix}

d_1 & 0 & 0 & \cdots & 0 \\

0 & d_2 & 0 & \cdots & 0 \\

0 & 0 & d_3 & \cdots & 0 \\

\vdots & \vdots & \vdots & \ddots & 0 \\

0 & 0 & 0 & \cdots & d_n

\end{bmatrix}$$

逆矩阵为:$$  

D^{-1} = \begin{bmatrix}

\frac{1}{d_1} & 0 & 0 & \cdots & 0 \\

0 & \frac{1}{d_2} & 0 & \cdots & 0 \\

0 & 0 & \frac{1}{d_3} & \cdots & 0 \\

\vdots & \vdots & \vdots & \ddots & 0 \\

0 & 0 & 0 & \cdots & \frac{1}{d_n}

\end{bmatrix}$$
对角线上元素的倒数

==总体协方差矩阵一定可逆,但是样本协方差矩阵不一定有逆.==

## 6. **正交矩阵 (Orthogonal Matrix)**
  
   - 矩阵 $Q$ 满足：
     $$Q Q' = Q' Q = I_p$$
     则称 $Q$ 为正交矩阵。
   - 正交矩阵的列向量和行向量均为单位正交向量。

Q的转置等于他的逆.他的行和列都是正交的.

## 7. **矩阵的秩 (Rank of a Matrix)**
   
   - 矩阵 $A_{n \times p}$ 的秩是其线性无关列向量的最大数目，记为 $\text{rank}(A)$，满足：
     $$\text{rank}(A) \leq \min(n, p)$$
   - 当 $\text{rank}(A) = \min(n, p)$ 时，$A$ 为满秩矩阵；否则为秩亏矩阵。

## 8.**矩阵的迹**

矩阵的迹(trace)是一个方阵的对角线元素之和。对于一个 $n \times n$ 的方阵 $\mathbf{A}$：
$$
\mathbf{A} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{bmatrix}
$$

矩阵的迹记为 $\text{tr}(\mathbf{A})$，其定义为：

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii} = a_{11} + a_{22} + \cdots + a_{nn}
$$

**矩阵的迹的求法**

1. **确定矩阵为方阵**：只有方阵才定义有迹。
2. **提取对角线元素**：将矩阵的主对角线元素取出。
3. **计算对角线元素的和**：将这些元素相加，得到迹。

**例子**

**例子 1: 二阶矩阵**

$$
\mathbf{A} =
\begin{bmatrix} 
1 & 2 \\
3 & 4
\end{bmatrix}
$$

矩阵的对角线元素为 $1, 4$，因此：

$$
\text{tr}(\mathbf{A}) = 1 + 4 = 5
$$

**例子 2: 三阶矩阵**

$$

\mathbf{B} =

\begin{bmatrix}

2 & 5 & 1 \\

0 & 3 & -1 \\

4 & 2 & 6

\end{bmatrix}

$$

矩阵的对角线元素为 $2, 3, 6$，因此：

$$
\text{tr}(\mathbf{B}) = 2 + 3 + 6 = 11
$$ 
**矩阵的迹的性质**

1. **可加性**：
$$
\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})
$$
2. **数乘分配性**：
$$
\text{tr}(c\mathbf{A}) = c \cdot \text{tr}(\mathbf{A}), \quad c \in \mathbb{R}
$$
3. **转置不变性**：
$$
\text{tr}(\mathbf{A}) = \text{tr}(\mathbf{A}^\top)
$$

4. **相似矩阵的迹相等**：

如果 $\mathbf{B} = \mathbf{P}^{-1} \mathbf{A} \mathbf{P}$，则：

$$

\text{tr}(\mathbf{B}) = \text{tr}(\mathbf{A})

$$

5. **矩阵乘积的迹交换性**（但注意 $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ 时迹仍然相等）：

$$
\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})
$$

---

# Determinants 

## 1. **行列式的定义**

   - 对于一个方阵 $A$，行列式 $|A|$ 是一个标量，反映矩阵的某些代数性质。
   - 不同阶矩阵的行列式计算方式：
     - 若 $p = 1$：
       $$|A| = a_{11}$$
     - 若 $p = 2$：
       $$|A| = a_{11}a_{22} - a_{12}a_{21}$$
     - 若 $p = 3$：
       $$
       |A| = a_{11} \begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix}
       - a_{12} \begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix}
       + a_{13} \begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}
       $$

## 2. **行列式的一般计算公式**
   
   - 对于 $n \times n$ 矩阵 $A$：
     $$
     |A| = \sum_{j=1}^n (-1)^{1+j}a_{1j}|A_{1j}|
     $$
     其中 $A_{1j}$ 是删除第 $1$ 行和第 $j$ 列后的子矩阵。

## 3. **行列式的性质**
   
   - **乘法性质**：
     若 $A$ 和 $B$ 为方阵，则：
     $$|AB| = |A||B|$$
   - **逆矩阵性质**：
     若 $A$ 可逆，则：
     $$|A^{-1}| = \frac{1}{|A|}$$
   - **常数乘法性质**：
     若 $c$ 是常数，则：
     $$|cA| = c^p|A|$$
     其中 $p$ 为矩阵的阶数。
   - **对角矩阵的行列式**：
     若 $A$ 是对角矩阵：
     $$
     A = \begin{pmatrix}
     a_{11} & 0 & \cdots & 0 \\
     0 & a_{22} & \cdots & 0 \\
     \vdots & \vdots & \ddots & \vdots \\
     0 & 0 & \cdots & a_{pp}
     \end{pmatrix}
     $$
     则 $|A|$ 为对角线元素的乘积：
     $$|A| = a_{11}a_{22}\cdots a_{pp}$$

## 4. **行列式与矩阵特性的关系**
   
   - **非奇异矩阵**：
     若 $|A| \neq 0$，则 $A$ 可逆，且列向量线性无关。
   - **正交矩阵**：
     对于正交矩阵 $Q$，其行列式为：
     $$|Q| = \pm 1$$

证明:

正交矩阵 $Q$ 满足以下行列式的性质：
 $$|Q’Q| = |I| = 1,$$

因为单位矩阵的行列式为 $1$。

• 行列式的乘积性质：$$|Q’Q| = |Q’||Q|.$$

因为 $|Q’| = |Q|$（转置矩阵的行列式与原矩阵相等），所以有：

$$|Q|^2 = 1.$$

• 从上式可得：$$|Q| = \pm 1.$$

## 5. **行列式与特征值的关系**
  
   - 若 $A$ 的特征值为 $\lambda_1, \lambda_2, \cdots, \lambda_p$，则：
     - 行列式为所有特征值的乘积：
       $$|A| = \prod_{i=1}^p \lambda_i$$
     - 矩阵的迹为特征值之和：
       $$\text{tr}(A) = \sum_{i=1}^p \lambda_i$$

## 6. **行列式的应用**
   
- **判断矩阵的可逆性**：
     若 $|A| \neq 0$，则 $A$ 可逆；否则不可逆。
   - **线性变换的几何意义**：
     行列式的绝对值表示线性变换对体积的缩放比例。

---

# Eigenvalues and Eigenvectors 

## 1. **特征值与特征向量的定义**
   
   - 对于一个方阵 $A$，若存在非零向量 $x$ 和标量 $\lambda$，使得：
     $$Ax = \lambda x$$
     则 $\lambda$ 称为 $A$ 的**特征值 (eigenvalue)**，$x$ 称为对应的**特征向量 (eigenvector)**。
   - **几何意义**：
     - 特征向量是矩阵 $A$ 作用下**方向不变**的向量。
     - 特征值表示 $A$ 在对应特征向量上的拉伸或缩放比例。

## 2. **特征值与特征向量的性质**
   
   - 一个 $p \times p$ 矩阵 $A$ 有 $p$ 对特征值与特征向量：
     $$(\lambda_1, e_1), (\lambda_2, e_2), \cdots, (\lambda_p, e_p)$$
   - 特征向量可以标准化为单位向量：
     $$e = \frac{x}{L_x}, \quad L_x = \sqrt{x'x}$$
   - 当 $A$ 是对称矩阵时，不同特征值对应的特征向量是正交的（互相垂直）。

## 3. **特征值与特征向量的计算方法**
   
   - 特征值的求解：
     - 解以下特征方程：
       $$|A - \lambda I| = 0$$
       其中 $I$ 是单位矩阵，$\lambda$ 是特征值。
   - 特征向量的求解：
     - 对每个特征值 $\lambda$，解线性方程组：
       $$(A - \lambda I)\begin{pmatrix} x \\ x \end{pmatrix} = 0$$
     - 注意：$x$ 必须是非零解。

   **举例**：设 $A = \begin{pmatrix} 2 & -3 \\ -3 & 2 \end{pmatrix}$。
   - 求特征值：解 $|A - \lambda I| = 0$：
     $$\begin{vmatrix} 2-\lambda & -3 \\ -3 & 2-\lambda \end{vmatrix} = (2-\lambda)^2 - 9 = 0$$
     解得 $\lambda_1 = 5, \lambda_2 = -1$。
   - 求特征向量：
     - 对 $\lambda_1 = 5$，解 $(A - 5I)x = 0$，得 $e_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} -1 \\ 1 \end{pmatrix}$。
     - 对 $\lambda_2 = -1$，解 $(A + I)x = 0$，得 $e_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$。

## 4. **特征值与特征向量的应用**
   
### **谱分解 (Spectral Decomposition)**：
    
#### 1. **谱分解的定义**

**谱分解**（Spectral Decomposition）是线性代数中的一个重要概念，用于将一个矩阵分解成由其特征值和特征向量定义的形式。谱分解的核心思想是利用矩阵的特征值和特征向量，将矩阵表示为一个基于这些特征的分解形式。

对于一个 $n \times n$ 的对称矩阵 $\mathbf{A}$，它可以表示为：

$$
\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top
$$
其中：

• $\mathbf{Q}$ 是由矩阵 $\mathbf{A}$ 的正交特征向量组成的正交矩阵.
• $\mathbf{\Lambda}$ 是一个对角矩阵，其对角元素是矩阵 $\mathbf{A}$ 的特征值。

#### 2. **谱分解的具体形式**

假设 $\mathbf{A}$ 是一个对称矩阵，其特征值为 $\lambda_1, \lambda_2, \dots, \lambda_n$，对应的特征向量为 $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$。那么：
$$
\mathbf{A} = \sum_{i=1}^n \lambda_i \mathbf{v}_i \mathbf{v}_i^\top
$$

其中：
• $\lambda_i$ 是特征值；
• $\mathbf{v}_i$ 是单位化的特征向量，即 $||\mathbf{v}_i|| = 1$；
• $\mathbf{v}_i \mathbf{v}_i^\top$ 是一个秩为 1 的矩阵。


**谱分解的步骤**

1. **求特征值和特征向量**：

对矩阵 $\mathbf{A}$，解特征方程 $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$，得到特征值 $\lambda_i$。随后计算对应的特征向量 $\mathbf{v}_i$。

2. **构造矩阵 $\mathbf{\Lambda}$ 和 $\mathbf{Q}$**：

• 将特征值 $\lambda_1, \lambda_2, \dots, \lambda_n$ 排列为对角矩阵 $\mathbf{\Lambda}$；

• 将特征向量按列排列，构成正交矩阵 $\mathbf{Q}$。

3. **构造谱分解形式**：

利用 $\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top$，完成分解。

#### 3.  **谱分解例子**

**示例 1: 二阶对称矩阵**

给定矩阵：

$$

\mathbf{A} = \begin{bmatrix}

4 & 1 \\

1 & 3

\end{bmatrix}

$$

1. **求特征值**：

解 $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$：

$$

\det\begin{bmatrix}

4-\lambda & 1 \\

1 & 3-\lambda

\end{bmatrix} = (4-\lambda)(3-\lambda) - 1 = \lambda^2 - 7\lambda + 11 = 0

$$

解得特征值 $\lambda_1 = 5, \lambda_2 = 2$。

2. **求特征向量**：

对 $\lambda_1 = 5$，解 $(\mathbf{A} - 5\mathbf{I})\mathbf{v} = 0$：

$$

(\mathbf{A} - 5\mathbf{I}) = \begin{bmatrix}

-1 & 1 \\

1 & -2

\end{bmatrix}

$$

解得特征向量 $\mathbf{v}_1 = \begin{bmatrix} 1  \\  1 \end{bmatrix}$。

对 $\lambda_2 = 2$，解 $(\mathbf{A} - 2\mathbf{I})\mathbf{v} = 0$：

$$
(\mathbf{A} - 2\mathbf{I}) = \begin{bmatrix}
2 & 1 \\
1 & 1
\end{bmatrix}
$$

解得特征向量 $\mathbf{v}_2 = \begin{bmatrix} -1  \\  1 \end{bmatrix}$。

3. **构造 $\mathbf{\Lambda}$ 和 $\mathbf{Q}$**：

$$
\mathbf{\Lambda} = \begin{bmatrix}
5 & 0 \\
0 & 2
\end{bmatrix}, \quad
\mathbf{Q} = \begin{bmatrix}
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}
\end{bmatrix}
$$
4. **谱分解结果**：
$$
\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top
$$

#### 4.**谱分解的性质**

1. **对称矩阵的谱定理**：

任意实对称矩阵都可以通过正交矩阵对角化，其特征值全为实数，特征向量互相正交。

2. **正定矩阵的特征值**：

如果 $\mathbf{A}$ 是正定矩阵，则其特征值均为正数。

3. **矩阵的迹与行列式**：

• 矩阵 $\mathbf{A}$ 的迹等于所有特征值的和：

$$

\text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i

$$

• 矩阵 $\mathbf{A}$ 的行列式等于所有特征值的乘积：

$$

\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i

$$

证明:

**1. 矩阵的迹等于特征值之和**

**结论：**
$$
\text{tr}(A) = \sum_{i=1}^p \lambda_i
$$

这里，$\text{tr}(A)$ 表示矩阵 $A$ 的迹，$\lambda_i$ 是矩阵 $A$ 的特征值。

**证明：**

1. 假设 $A$ 是一个 $p \times p$ 的矩阵，其谱分解形式为：

$$
A = P \Lambda P^{-1}
$$

其中：
• $P$ 是由 $A$ 的特征向量组成的矩阵。
• $\Lambda = \text{diag}(\lambda_1, \lambda_2, \cdots, \lambda_p)$ 是对角矩阵，对角线上的元素是 $A$ 的特征值。

2. 矩阵的迹具有如下性质：

$$

\text{tr}(A) = \text{tr}(P \Lambda P^{-1})

$$

因为迹在矩阵相似变换（如 $P \Lambda P^{-1}$）下保持不变：
$$
\text{tr}(P \Lambda P^{-1}) = \text{tr}(\Lambda)
$$

3. 对角矩阵 $\Lambda$ 的迹是其对角线元素的和：
$$
\text{tr}(\Lambda) = \sum_{i=1}^p \lambda_i
$$

因此：
$$
\text{tr}(A) = \sum_{i=1}^p \lambda_i
$$

**2. 行列式等于特征值的乘积**

**结论：**
$$
|A| = \prod_{i=1}^p \lambda_i
$$

这里，$|A|$ 是矩阵 $A$ 的行列式，$\lambda_i$ 是 $A$ 的特征值。

**证明：**

1. 假设矩阵 $A$ 的谱分解为：
$$
A = P \Lambda P^{-1}
$$
其中：
• $P$ 是由特征向量组成的矩阵。
• $\Lambda = \text{diag}(\lambda_1, \lambda_2, \cdots, \lambda_p)$。

2. 行列式的性质：行列式在矩阵相似变换下不变，即：
$$
|A| = |P \Lambda P^{-1}|
$$
由于行列式的计算规则为：
$$
|P \Lambda P^{-1}| = |P| \cdot |\Lambda| \cdot |P^{-1}|
$$

而 $|P^{-1}| = \frac{1}{|P|}$，所以：

$$
|P \Lambda P^{-1}| = |\Lambda|
$$

3. 对角矩阵 $\Lambda$ 的行列式是其对角线元素的乘积：

$$

|\Lambda| = \prod_{i=1}^p \lambda_i

$$

因此：

$$
|A| = \prod_{i=1}^p \lambda_i
$$

**总结：**

4. **矩阵的幂**：

利用谱分解可以高效计算矩阵的幂：

$$

\mathbf{A}^k = \mathbf{Q} \mathbf{\Lambda}^k \mathbf{Q}^\top

$$
   
   - **迹和行列式的关系**：
     - 矩阵的迹等于特征值之和：
       $$\text{tr}(A) = \sum_{i=1}^p \lambda_i$$
     - 行列式等于特征值的乘积：
       $$|A| = \prod_{i=1}^p \lambda_i$$
   - **线性变换的几何解释**：
     - 特征值和特征向量可以帮助理解矩阵 $A$ 对空间的拉伸、旋转和缩放作用。例如：
       - 正的特征值表示沿特征向量方向的拉伸；
       - 负的特征值表示方向翻转；
       - 特征值的绝对值越大，拉伸或压缩程度越显著。

## 5.奇异值分解

我们对一个 $2 \times 2$ 矩阵 $A$ 进行 **奇异值分解**，即将矩阵分解为：

$$
A = U \Sigma V^T
$$

其中：
• $U$ 是左奇异向量矩阵。
• $\Sigma$ 是奇异值矩阵。
• $V$ 是右奇异向量矩阵。

**1. 给定矩阵 $A$**

$$
A = \begin{pmatrix} 3 & 0  \\  4 & 0 \end{pmatrix}.
$$

**2. 计算 $A^T A$ 和 $A A^T$**

为求奇异值，先计算 $A^T A$ 和 $A A^T$：

1. **计算 $A^T A$：**

$$
A^T = \begin{pmatrix} 3 & 4  \\  0 & 0 \end{pmatrix}, \quad A^T A = \begin{pmatrix} 3 & 4  \\  0 & 0 \end{pmatrix} \begin{pmatrix} 3 & 0  \\  4 & 0 \end{pmatrix}.
$$

矩阵乘法结果为：
$$
A^T A = \begin{pmatrix} 9 + 16 & 0  \\  0 & 0 \end{pmatrix} = \begin{pmatrix} 25 & 0  \\  0 & 0 \end{pmatrix}.
$$

2. **计算 $A A^T$：**

$$
A A^T = \begin{pmatrix} 3 & 0 \\  4 & 0 \end{pmatrix} \begin{pmatrix} 3 & 4  \\  0 & 0 \end{pmatrix}.
$$

矩阵乘法结果为：

$$
A A^T = \begin{pmatrix} 9 & 12  \\  12 & 16 \end{pmatrix}.
$$

**3. 计算奇异值**

奇异值是 $A^T A$ 或 $A A^T$ 的 **特征值的平方根**。

1. 从 $A^T A$：

$$
A^T A = \begin{pmatrix} 25 & 0  \\  0 & 0 \end{pmatrix}.
$$

特征值为：

$$
\lambda_1 = 25, \quad \lambda_2 = 0.
$$

2. 奇异值为特征值的平方根：

$$
\sigma_1 = \sqrt{25} = 5, \quad \sigma_2 = \sqrt{0} = 0.
$$

因此，奇异值矩阵为：

$$
\Sigma = \begin{pmatrix} 5 & 0  \\  0 & 0 \end{pmatrix}.
$$

**4. 计算 $V$ 和 $U$**

1. **求右奇异向量矩阵 $V$：**

• $V$ 是 $A^T A$ 的特征向量组成的矩阵。
• $A^T A = \begin{pmatrix} 25 & 0  \\  0 & 0 \end{pmatrix}$，对应特征值 $\lambda_1 = 25$ 和 $\lambda_2 = 0$。

• 对应的特征向量为：

$$
v_1 = \begin{pmatrix} 1  \\  0 \end{pmatrix}, \quad v_2 = \begin{pmatrix} 0  \\  1 \end{pmatrix}.
$$

所以：

$$
V = \begin{pmatrix} 1 & 0  \\  0 & 1 \end{pmatrix}.
$$

2. **求左奇异向量矩阵 $U$：**

• $U$ 是 $A A^T$ 的特征向量组成的矩阵。
• $A A^T = \begin{pmatrix} 9 & 12  \\  12 & 16 \end{pmatrix}$。
• 通过计算可知，$A A^T$ 的特征值为 $\lambda_1 = 25$ 和 $\lambda_2 = 0$：
• 对应特征值 $\lambda_1 = 25$，特征向量为：

$$
u_1 = \begin{pmatrix} \frac{3}{5}  \\  \frac{4}{5} \end{pmatrix}.
$$

• 对应特征值 $\lambda_2 = 0$，特征向量为：

$$
u_2 = \begin{pmatrix} -\frac{4}{5}  \\  \frac{3}{5} \end{pmatrix}.
$$

所以：

$$
U = \begin{pmatrix} \frac{3}{5} & -\frac{4}{5}  \\  \frac{4}{5} & \frac{3}{5} \end{pmatrix}.
$$

**5. 组合奇异值分解**

将 $U$、$\Sigma$、$V$ 组合起来，得到矩阵 $A$ 的奇异值分解：

$$
A = U \Sigma V^T.
$$

具体为：

$$
\begin{pmatrix} 3 & 0  \\  4 & 0 \end{pmatrix} =
\begin{pmatrix} \frac{3}{5} & -\frac{4}{5}  \\  \frac{4}{5} & \frac{3}{5} \end{pmatrix}
\begin{pmatrix} 5 & 0  \\  0 & 0 \end{pmatrix}
\begin{pmatrix} 1 & 0  \\  0 & 1 \end{pmatrix}^T.
$$

**6. 小结**

  

通过奇异值分解：

1. 计算奇异值 $\sigma_1$ 和 $\sigma_2$，构造 $\Sigma$。
2. 求解 $A^T A$ 的特征向量，构造 $V$。
3. 求解 $A A^T$ 的特征向量，构造 $U$。
4. 最终将矩阵 $A$ 表示为：
$$
A = U \Sigma V^T.
$$

---

# Positive Definite and Non-Negative Definite Matrices 

## 1. **定义**
   
   - **正定矩阵 (Positive Definite Matrix)**:
     若 $A$ 是一个 $p \times p$ 的对称矩阵，且对于任意非零向量 $x \neq 0$，满足：
     $$x'Ax > 0$$
     则 $A$ 被称为正定矩阵。
   - **半正定矩阵 (Non-Negative Definite Matrix)**:
     若 $A$ 是一个 $p \times p$ 的对称矩阵，且对于任意非零向量 $x \neq 0$，满足：
     $$x'Ax \geq 0$$
     则 $A$ 被称为半正定矩阵。

## 2. **几何意义**
   
   - **正定矩阵**：对应于一个严格凸的二次型（即 $x'Ax$ 在任意方向都大于零）。
   - **半正定矩阵**：对应于一个不严格凸的二次型（即 $x'Ax$ 可以等于零，但不能为负）。

## 3. **特征值的判定条件**
   
   - 一个矩阵是否为正定或半正定可以通过其特征值判断：
     - 若 $A$ 的所有特征值 $\lambda_i > 0$，则 $A$ 是正定矩阵。
     - 若 $A$ 的所有特征值 $\lambda_i \geq 0$，则 $A$ 是半正定矩阵。

## 4. **二次型 (Quadratic Form)**
   
   - $x'Ax$ 被称为一个关于 $p$ 个变量 $x_1, x_2, \cdots, x_p$ 的二次型：
    $$
     x'Ax = \sum_{i=1}^p \sum_{j=1}^p a_{ij}x_ix_j
     $$
   - 在正定矩阵的情况下，二次型 $x'Ax > 0$ 表示该函数无论 $x$ 如何变化，其值总为正。

## 5. **正定矩阵的等价条件**
   
   - 若 $A$ 是对称矩阵，下列条件等价于 $A$ 是正定矩阵：
     1. 所有特征值 $\lambda_i > 0$。
     2. 对任意非零向量 $x$，$x'Ax > 0$。
     3. 所有主子矩阵（leading principal minors）的行列式均为正。
   - 半正定矩阵的等价条件：
     1. 所有特征值 $\lambda_i \geq 0$。
     2. 对任意非零向量 $x$，$x'Ax \geq 0$。

**证明**

 如果$A$是对称矩阵，可以对$A$进行谱分解：$$ A = Q \Lambda Q^\top $$其中$Q$是正交矩阵，$\Lambda$是对角矩阵，对角线上是$A$的特征值$\lambda_1, \lambda_2, \dots, \lambda_n$。

对于任意非零向量$x$，有：$$x^\top A x = x^\top (Q \Lambda Q^\top) x = (Q^\top x)^\top \Lambda (Q^\top x) $$令$y = Q^\top x$，因$Q$是正交矩阵，其列向量组成正交基，满足$Q^\top Q = I$，因此$x \neq 0 \implies y \neq 0$。于是： $$x^\top A x = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2$$
若$\lambda_i > 0$，则$x^\top A x > 0$。
则证明: 当且仅当所有特征值$\lambda_i > 0$时，$x^\top A x > 0$对任意$x \neq 0$成立。因此，矩阵$A$是正定矩阵。

## 7. **例子与直观理解**
   
   - **例子 1: 对称正定矩阵**  
     $$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$  
     - $A$ 的特征值为 $\lambda_1 = 3, \lambda_2 = 1$，均为正，因此 $A$ 是正定矩阵。
     - 对任意向量 $x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$：
       $$x'Ax = 2x_1^2 + 2x_2^2 + 2x_1x_2 > 0$$
       可见二次型恒为正。

   - **例子 2: 半正定矩阵**  
     $$B = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$  
     - $B$ 的特征值为 $\lambda_1 = 1, \lambda_2 = 0$，均为非负，因此 $B$ 是半正定矩阵。
     - 对于 $x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$：
       $$x'Bx = x_1^2 \geq 0$$

## 8. **正定矩阵的平方根**
   
   - 若 $A$ 是正定矩阵，则可以定义 $A$ 的平方根矩阵 $A^{1/2}$，满足：
     $$A^{1/2}A^{1/2} = A$$
     其构造方式基于特征分解：
     $$A = P \Lambda P', \quad A^{1/2} = P \Lambda^{1/2} P'$$
     其中 $\Lambda^{1/2}$ 是对角矩阵，包含 $\sqrt{\lambda_i}$。

---

# Square Root of a Matrix 

## 1. **定义**
   
   - 矩阵的平方根是一个矩阵 $A^{1/2}$，满足：
     $$A^{1/2}A^{1/2} = A$$
   - 条件：平方根矩阵的存在性要求 $A$ 必须是**对称正定矩阵**。

## 2. **构造方法**
   
   - 使用特征值分解：
     $$A = P \Lambda P'$$
     其中：
     - $P$ 是 $A$ 的特征向量构成的正交矩阵。
     - $\Lambda = \text{diag}(\lambda_1, \lambda_2, \cdots, \lambda_p)$ 是 $A$ 的特征值构成的对角矩阵。
   - 定义平方根矩阵为：
     $$A^{1/2} = P \Lambda^{1/2} P'$$
     其中 $\Lambda^{1/2} = \text{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \cdots, \sqrt{\lambda_p})$。

## 3. **性质**
   
   - **对称性**：如果 $A$ 是对称矩阵，则 $A^{1/2}$ 也是对称矩阵。
     $$ (A^{1/2})' = A^{1/2} $$
   - **正定性**：如果 $A$ 是正定矩阵，则 $A^{1/2}$ 也是正定矩阵。
   - **逆矩阵与平方根**：
     $$ A^{-1} = (A^{1/2})^{-1}(A^{1/2})^{-1}, \quad A^{-1/2} = (A^{1/2})^{-1} $$

## 4. **几何意义**
   
   - 矩阵平方根可以被视为将 $A$ 的作用分解为两个连续的“半作用”。
   - 对于正定矩阵 $A$，$A^{1/2}$ 是一种等价于线性变换 $A$ 的简化操作。

## 5. **例子**
   
   - **例子 1: 简单对称正定矩阵**  
     $$A = \begin{pmatrix} 4 & 0 \\ 0 & 9 \end{pmatrix}$$  
     - 特征值：$\lambda_1 = 4, \lambda_2 = 9$。
     - 特征值平方根：$\sqrt{\lambda_1} = 2, \sqrt{\lambda_2} = 3$。
     - 矩阵平方根：
       $$A^{1/2} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$$
     - 验证：$A^{1/2}A^{1/2} = A$。

   - **例子 2: 非对角矩阵**  
     $$A = \begin{pmatrix} 5 & 4 \\ 4 & 5 \end{pmatrix}$$  
     - 特征值分解：$\lambda_1 = 9, \lambda_2 = 1$，特征向量构成正交矩阵：
       $$P = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$
     - 矩阵平方根：
       $$A^{1/2} = P \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} P' = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}$$

## 6. **应用**
   
   - **协方差矩阵的变换**：
     - 协方差矩阵 $\Sigma$ 的平方根 $\Sigma^{1/2}$ 用于生成服从多元正态分布的随机向量：
       $$X \sim N(0, \Sigma), \quad X = \Sigma^{1/2}Z$$
       其中 $Z$ 是标准正态分布随机向量。
   - **正定矩阵的线性变换**：
     - 在优化中，平方根矩阵常用于条件数分析和预处理。

## 7. **重要公式**
   
   - 矩阵平方根与特征值：
     $$A^{1/2} = \sum_{i=1}^p \sqrt{\lambda_i} e_i e_i'$$
     其中 $e_i$ 是特征向量，$\lambda_i$ 是对应特征值。
   - 逆平方根矩阵：
     $$A^{-1/2} = \sum_{i=1}^p \frac{1}{\sqrt{\lambda_i}} e_i e_i'$$

---

# **Quadratic Transformation**

## **1. 定义**

一个 **Quadratic Transformation** 表示为：

$$
y = Q(x) = x^T A x + b^T x + c,
$$
其中：

• $x \in \mathbb{R}^n$ 是一个 $n \times 1$ 的向量。
• $A \in \mathbb{R}^{n \times n}$ 是对称矩阵（通常要求）。
• $b \in \mathbb{R}^n$ 是线性变换的系数向量。
• $c \in \mathbb{R}$ 是常数项。

## **2. 几何意义**

二次变换的几何意义在于它可以生成二次曲线或曲面：

1. **二次型的作用：**

• $x^T A x$ 描述输入变量 $x$ 的二次关系，决定了曲线或曲面的形状。
• 矩阵 $A$ 的正定性或不定性决定几何性质：
• $A$ 正定：生成椭圆或椭球。
• $A$ 不定：生成双曲线或双曲面。

2. **线性部分 $b^T x$：**

• $b^T x$ 表示线性变换，决定几何的偏移方向。

3. **常数项 $c$：**

• $c$ 是整体的平移量，影响曲线或曲面的位置。

## **3. 常见形式**

### **3.1 简化形式**

最简单的二次变换为：
$$
y = x^T A x,
$$

其中没有线性项和常数项。这种变换仅包含二次型部分。

### **3.2 标准形式**

对于二维变量 $x = (x_1, x_2)^T$，二次变换可以写为：

$$

Q(x) = ax_1^2 + bx_1x_2 + cx_2^2 + dx_1 + ex_2 + f,

$$

这可转换为矩阵形式：

$$

Q(x) = x^T A x + b^T x + c,

$$

其中 $x = \begin{pmatrix} x_1 \ x_2 \end{pmatrix}$，$A$ 是对称矩阵。

  
## 4.**正定矩阵 Rayleigh 商的最大值与最小值**

### 4.1结论：

1. $\max_{x \neq 0} \frac{x^T B x}{x^T x} = \lambda_1$，当 $x = e_1$。
2. $\min_{x \neq 0} \frac{x^T B x}{x^T x} = \lambda_p$，当 $x = e_p$。

其中：
• $B$ 是一个 $p \times p$ 的正定矩阵。
• $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_p$ 是 $B$ 的特征值。
• $e_1, e_2, \dots, e_p$ 是 $B$ 的单位正交特征向量（即 $e_i^T e_i = 1$）。
• $x$ 是任意的非零向量（$x \neq 0$）。

### **4.2. 预备知识**

• 任意向量 $x \in \mathbb{R}^p$ 可以表示为矩阵 $B$ 的特征向量的线性组合：

$$

x = \sum_{i=1}^p c_i e_i, \quad \text{其中 } c_i = e_i^T x.

$$
• 特征向量 $e_i$ 是正交归一化的，即：

$$
e_i^T e_j = \begin{cases}
1, & \text{若 } i = j,  \\
0, & \text{若 } i \neq j.
\end{cases}
$$

• 正定矩阵的特征值 $\lambda_i$ 均为正实数。

### **4.3 Rayleigh 商的展开**

Rayleigh 商定义为：

$$
R(x) = \frac{x^T B x}{x^T x}.
$$

将 $x$ 展开为 $x = \sum_{i=1}^p c_i e_i$，代入 $R(x)$ 中，得到：

$$

x^T B x = \left( \sum_{i=1}^p c_i e_i \right)^T B \left( \sum_{j=1}^p c_j e_j \right).

$$

由于 $B e_i = \lambda_i e_i$，有：

$$
B \left( \sum_{j=1}^p c_j e_j \right) = \sum_{j=1}^p c_j \lambda_j e_j.
$$

因此：
$$
x^T B x = \left( \sum_{i=1}^p c_i e_i \right)^T \left( \sum_{j=1}^p c_j \lambda_j e_j \right) = \sum_{i=1}^p \lambda_i c_i^2.
$$

同时，$x^T x$ 为：
$$
x^T x = \left( \sum_{i=1}^p c_i e_i \right)^T \left( \sum_{j=1}^p c_j e_j \right) = \sum_{i=1}^p c_i^2.
$$

将 $x^T B x$ 和 $x^T x$ 代入 Rayleigh 商，得到
$$
R(x) = \frac{x^T B x}{x^T x} = \frac{\sum_{i=1}^p \lambda_i c_i^2}{\sum_{i=1}^p c_i^2}.
$$

### **4.4 最大值、最小值的证明**

要最大化 $R(x)$，观察到：
• 在分子 $\sum_{i=1}^p \lambda_i c_i^2$ 中，$\lambda_i$ 按降序排列，且 $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_p$。
• 当 $c_1 = 1$ 且 $c_2 = c_3 = \dots = c_p = 0$ 时，$x$ 完全沿着特征向量 $e_1$ 的方向。

此时：
$$
R(x) = \frac{\lambda_1 \cdot 1^2}{1} = \lambda_1.
$$  

因此，Rayleigh 商的最大值为：

$$
\max_{x \neq 0} \frac{x^T B x}{x^T x} = \lambda_1, \quad \text{当 } x = e_1.
$$

要最小化 $R(x)$，同理可得：
• 在分子 $\sum_{i=1}^p \lambda_i c_i^2$ 中，$\lambda_p$ 是最小的特征值。
• 当 $c_p = 1$ 且 $c_1 = c_2 = \dots = c_{p-1} = 0$ 时，$x$ 完全沿着特征向量 $e_p$ 的方向。

此时：
$$
R(x) = \frac{\lambda_p \cdot 1^2}{1} = \lambda_p.
$$

因此，Rayleigh 商的最小值为：

$$

\min_{x \neq 0} \frac{x^T B x}{x^T x} = \lambda_p, \quad \text{当 } x = e_p.

$$

------------


# Random Vectors and Random Matrices 

## 1. **随机向量 (Random Vectors)**
   
   - 随机向量是由多个随机变量组成的列向量：
     $$x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{pmatrix}$$
     其中 $x_1, x_2, \cdots, x_p$ 是随机变量。
   - **期望值**：
     $$E(x) = \begin{pmatrix} E(x_1) \\ E(x_2) \\ \vdots \\ E(x_p) \end{pmatrix} = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_p \end{pmatrix}$$
   - **性质**：
     - 若 $y = Ax + b$，其中 $A$ 是常数矩阵，$b$ 是常数向量，则：
       $$E(y) = AE(x) + b$$

## 2. **随机向量的协方差矩阵 (Variance-Covariance Matrix)**
   
   - 定义为随机变量之间的方差与协方差的矩阵：
     $$\Sigma = \begin{pmatrix} 
     \text{Var}(x_1) & \text{Cov}(x_1, x_2) & \cdots & \text{Cov}(x_1, x_p) \\ 
     \text{Cov}(x_2, x_1) & \text{Var}(x_2) & \cdots & \text{Cov}(x_2, x_p) \\ 
     \vdots & \vdots & \ddots & \vdots \\ 
     \text{Cov}(x_p, x_1) & \text{Cov}(x_p, x_2) & \cdots & \text{Var}(x_p)
     \end{pmatrix}$$
     其中 $\text{Cov}(x_i, x_j) = E[(x_i - \mu_i)(x_j - \mu_j)]$。
   - **性质**：
     - $\Sigma$ 是对称矩阵：
       $$\Sigma_{ij} = \Sigma_{ji}$$
     - 若 $x$ 的随机变量是线性无关的，则 $\Sigma$ 是正定矩阵。
   - **示例**：对于随机向量 $x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}$：
     $$\Sigma = \begin{pmatrix} 
     \sigma_{11} & \sigma_{12} & \sigma_{13} \\ 
     \sigma_{21} & \sigma_{22} & \sigma_{23} \\ 
     \sigma_{31} & \sigma_{32} & \sigma_{33} 
     \end{pmatrix}$$
     其中 $\sigma_{ij} = \text{Cov}(x_i, x_j)$。

## 3. **线性函数的方差 (Variance of a Linear Function)**
   
   - 若 $x$ 是一个 $p$ 维随机向量，$b$ 是一个常数向量，则：
     $$\text{Var}(b'x) = b'\Sigma b$$
   - 几何意义：这描述了向量 $x$ 在方向 $b$ 上的分布宽度。

## 4. **独立性与非相关性**
   
   - 如果 $x_i$ 和 $x_j$ 是独立的，则它们一定是非相关的，即 $\text{Cov}(x_i, x_j) = 0$。
   - 非相关性并不一定意味着独立性。

## 5. **随机矩阵 (Random Matrices)**
  
   - 随机矩阵是元素为随机变量的矩阵。例如，一个 $p \times q$ 随机矩阵：
     $$X = \begin{pmatrix} x_{11} & x_{12} & \cdots & x_{1q} \\ 
     x_{21} & x_{22} & \cdots & x_{2q} \\ 
     \vdots & \vdots & \ddots & \vdots \\ 
     x_{p1} & x_{p2} & \cdots & x_{pq} 
     \end{pmatrix}$$
   - 每一列可以看作一个随机向量。

## 6. **相关矩阵 (Correlation Matrix)**
   
   - 相关矩阵描述随机变量之间的相关性：
     $$\rho_{ij} = \frac{\text{Cov}(x_i, x_j)}{\sqrt{\text{Var}(x_i) \text{Var}(x_j)}} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii} \sigma_{jj}}}$$
   - 相关矩阵的性质：
     - 对角元素 $\rho_{ii} = 1$。
     - 相关系数满足 $-1 \leq \rho_{ij} \leq 1$。
   - **相关矩阵与协方差矩阵的关系**：
     $$\rho = V^{-1/2} \Sigma V^{-1/2}, \quad \Sigma = V^{1/2} \rho V^{1/2}$$
     其中 $V = \text{diag}(\sigma_{11}, \sigma_{22}, \cdots, \sigma_{pp})$。

---

# Correlation Matrix 

## 1. **定义**
   
   - 相关矩阵描述随机向量中各变量之间的相关性
   - 若随机向量 $x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_p \end{pmatrix}$，其相关矩阵为：
     $$
     \rho = \begin{pmatrix} 
     1 & \rho_{12} & \cdots & \rho_{1p} \\ 
     \rho_{21} & 1 & \cdots & \rho_{2p} \\ 
     \vdots & \vdots & \ddots & \vdots \\ 
     \rho_{p1} & \rho_{p2} & \cdots & 1 
     \end{pmatrix}
     $$
     其中 $\rho_{ij}$ 是 $x_i$ 和 $x_j$ 的相关系数。

## 2. **相关系数的公式**
   
   - 两随机变量 $x_i$ 和 $x_j$ 的相关系数定义为：
     $$
     \rho_{ij} = \frac{\text{Cov}(x_i, x_j)}{\sqrt{\text{Var}(x_i) \cdot \text{Var}(x_j)}} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii} \cdot \sigma_{jj}}}
     $$
   - **性质**：
     - $\rho_{ij}$ 的取值范围为 $-1 \leq \rho_{ij} \leq 1$。
     - $\rho_{ij} = 1$ 表示 $x_i$ 和 $x_j$ 完全正相关（线性关系为正）。
     - $\rho_{ij} = -1$ 表示 $x_i$ 和 $x_j$ 完全负相关（线性关系为负）。
     - $\rho_{ij} = 0$ 表示 $x_i$ 和 $x_j$ 不相关（线性关系为零）。

## 3. **相关矩阵的性质**
   
   - 对角线元素 $\rho_{ii} = 1$（每个变量与自身的相关系数为 1）。
   - $\rho_{ij} = \rho_{ji}$，即相关矩阵是对称矩阵。
   - 相关矩阵是正半定矩阵（所有特征值均非负）。
   - 相关矩阵可以通过协方差矩阵标准化得到。

## 4. **协方差矩阵与相关矩阵的关系**
  
   - 若随机向量 $x$ 的协方差矩阵为 $\Sigma$，则相关矩阵为：
     $$
     \rho = V^{-1/2} \Sigma V^{-1/2}
     $$
     其中 $V = \text{diag}(\sigma_{11}, \sigma_{22}, \cdots, \sigma_{pp})$，$\sigma_{ii} = \text{Var}(x_i)$。
   - 协方差矩阵可以从相关矩阵中恢复：
     $$
     \Sigma = V^{1/2} \rho V^{1/2}
     $$


## 5. **例子**
   
   - 假设随机向量 $x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}$ 的协方差矩阵为：
     $$
     \Sigma = \begin{pmatrix} 
     4 & 2 & 1 \\ 
     2 & 3 & 1 \\ 
     1 & 1 & 2 
     \end{pmatrix}
     $$
     计算对应的相关矩阵：
     - 方差向量 $V = \text{diag}(4, 3, 2)$。
     - 相关矩阵为：
       $$
       \rho = V^{-1/2} \Sigma V^{-1/2} = \begin{pmatrix} 
       1 & \frac{2}{\sqrt{12}} & \frac{1}{\sqrt{8}} \\ 
       \frac{2}{\sqrt{12}} & 1 & \frac{1}{\sqrt{6}} \\ 
       \frac{1}{\sqrt{8}} & \frac{1}{\sqrt{6}} & 1 
       \end{pmatrix}
       $$
