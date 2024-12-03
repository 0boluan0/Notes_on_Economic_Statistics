
# Content

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

## 6. **正交矩阵 (Orthogonal Matrix)**
  
   - 矩阵 $Q$ 满足：
     $$Q Q' = Q' Q = I_p$$
     则称 $Q$ 为正交矩阵。
   - 正交矩阵的列向量和行向量均为单位正交向量。

7. **矩阵的秩 (Rank of a Matrix)**
   - 矩阵 $A_{n \times p}$ 的秩是其线性无关列向量的最大数目，记为 $\text{rank}(A)$，满足：
     $$\text{rank}(A) \leq \min(n, p)$$
   - 当 $\text{rank}(A) = \min(n, p)$ 时，$A$ 为满秩矩阵；否则为秩亏矩阵。

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
