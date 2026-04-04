---
aliases:
  - 线性代数复习路线
  - MIT 18.06SC review
tags:
  - 线性代数
  - mit-ocw
  - review
---

# Review and exam roadmap

## 这页是做什么的

- 这页不是重复讲义，而是把整门线性代数变成可执行的复习路线。
- 使用顺序默认是：
  1. 先按 `整门课 15 分钟回忆` 把主线拉起来。
  2. 再按 `按题型复习` 找你最薄弱的模块。
  3. 最后回到 final review 和 final exam 做总验收。

## 资料现实情况

- 本地仓库里有三次 `Exam Review` 的 summary PDF，也有 `Final Course Review` 和 `Final Exam`。
- 本地仓库里没有 Exam 1 / Exam 2 / Exam 3 的试卷 PDF，因此复习主轴放在 review summary、作业题和 final 上。
- 题目总入口：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]

## 整门课 15 分钟回忆

### 第 1 步：先把三条主线说出来

- `Ax=b` 与四个基本子空间。
- 正交、投影、least squares、QR。
- determinant、eigen、positive definite、Jordan、SVD、pseudoinverse。

### 第 2 步：再把每条主线各举一个代表问题

- `Ax=b`：为什么有的系统无解，有的系统不唯一。
- projection / least squares：为什么误差必须正交。
- eigen / SVD：为什么某些方向被保留下来，某些矩阵必须换一种分解看。

### 第 3 步：最后把题型和工具对应起来

- 解结构问题 -> elimination / rref / four fundamental subspaces。
- 最佳逼近问题 -> projection / least squares / QR。
- 长期行为问题 -> eigen / diagonalization / $e^{At}$ / Markov。
- 一般矩阵结构问题 -> positive definite / Jordan / SVD / pseudoinverse。

## 按 unit 复习

### Unit I

- 主笔记：[[01_Ax = b and the Four Subspaces]]
- Review PDF：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf|Exam 1 review]]
- 核心能力：
  - 用消元判断是否有解、是否唯一、是否有自由变量。
  - 从原矩阵读 column space，从 rref 读 row space / nullspace。
  - 用 rank、nullity、orthogonality 解释四个基本子空间。
- 先回这些 session：
  - 1.3-1.5 算法线
  - 1.7-1.11 子空间与解结构
  - 1.13 incidence matrix

### Unit II

- 主笔记：[[02_Least Squares, Determinants and Eigenvalues]]
- Review PDF：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|Exam 2 review]]
- 核心能力：
  - 把 best fit 问题翻译成 projection / least squares。
  - 熟练使用 `P=A(A^TA)^{-1}A^T` 和 `A^TA\hat{x}=A^Tb`。
  - 用 eigen / diagonalization / matrix exponential 分析矩阵幂和长期行为。
- 先回这些 session：
  - 2.1-2.4 正交主线
  - 2.5-2.7 determinant 主线
  - 2.8-2.10 eigen 主线

### Unit III

- 主笔记：[[03_Positive Definite Matrices and Applications]]
- Review PDF：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf|Exam 3 review]]
- 核心能力：
  - 判断 symmetric / positive definite / similar / diagonalizable / rank-deficient 的结构含义。
  - 解释 SVD 与 pseudoinverse 在一般矩阵中的角色。
  - 把 linear transformation、change of basis、matrix representation 区分清楚。
- 先回这些 session：
  - 3.1-3.3 对称与正定
  - 3.4-3.5 Jordan 与 SVD
  - 3.6-3.8 linear transformation / change of basis / pseudoinverse

## 按题型复习

### 题型 1：解结构与四个基本子空间

- 入口：[[01_Ax = b and the Four Subspaces]]
- 必会题：
  - `Ax=0` 的通解
  - `Ax=b` 的相容条件与一般解
  - 从原矩阵和 rref 读四个基本子空间
- 配套卡：
  - [[Linear system solution structure]]
  - [[Reading the Four Fundamental Subspaces from RREF]]

### 题型 2：projection、least squares、QR

- 入口：[[02_Least Squares, Determinants and Eigenvalues#Session 2.1 Orthogonal vectors and subspaces|Unit II 正交主线]]
- 必会题：
  - 线投影和子空间投影
  - 正规方程
  - Gram-Schmidt 与 QR
- 配套卡：
  - [[Orthogonal Projection]]
  - [[Projection Matrix]]
  - [[Least Squares]]
  - [[Least Squares via Normal Equations]]
  - [[Gram-Schmidt Orthogonalization]]

### 题型 3：determinant、cofactor、inverse、volume

- 入口：[[02_Least Squares, Determinants and Eigenvalues#Session 2.5 Properties of determinants|Unit II determinant 主线]]
- 必会题：
  - 行操作对 determinant 的影响
  - cofactor / adjugate
  - Cramer’s rule 与 inverse 的关系
- 配套卡：
  - [[Determinant]]
  - [[Cramer's Rule]]
  - [[Matrix Inverse]]

### 题型 4：eigen、diagonalization、$e^{At}$、Markov

- 入口：[[02_Least Squares, Determinants and Eigenvalues#Session 2.8 Eigenvalues and eigenvectors|Unit II eigen 主线]]
- 必会题：
  - 求 eigenvalues / eigenvectors
  - 判断是否可对角化
  - 用 diagonalization 求幂、矩阵指数和长期行为
- 配套卡：
  - [[Eigenvalues]]
  - [[Diagonalization]]
  - [[Matrix Exponential]]
  - [[Markov Matrix]]

### 题型 5：positive definite、Jordan、SVD、pseudoinverse

- 入口：[[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Unit III 结构主线]]
- 必会题：
  - positive definite 的三种判别视角
  - 相似与 Jordan 结构
  - SVD 的来源、几何解释和 rank 信息
  - pseudoinverse 的公式与适用条件
- 配套卡：
  - [[Positive Definite Matrix]]
  - [[Testing Positive Definiteness]]
  - [[Jordan Form]]
  - [[Singular Value Decomposition]]
  - [[Pseudoinverse]]
  - [[Choosing Matrix Decompositions]]

## Final course review

- Review PDF：[[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_FinalRevsum.pdf|Final course review]]
- Final exam：[[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_final_ex.pdf|final exam]] | [[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_final_exs.pdf|solutions]]
- Final 额外范围：
  - [[Linear Transformation]]
  - [[Change of Basis]]
  - [[Pseudoinverse]]
  - 复数特征值与复内积

## 推荐复习动作

1. 先读 [[00_MIT OCW 18.06SC course map|course map]] 的 `三条主线`。
2. 再读 3 篇 unit note 的 `单元速览` 和 `Session 回忆索引`。
3. 对薄弱题型，只回相关 session，不从头重读整篇。
4. 看对应 `Exam Review` summary，检查自己能否先口述再看 PDF。
5. 最后做 `Final course review` 和 `final exam` 做总验收。

## 考前最后自检

- 我能把这门课压缩成 3 条主线和 5 类题型。
- 我知道什么时候该用 elimination，什么时候该用 least squares，什么时候该切换到 eigen / SVD。
- 我能在几何语言、子空间语言、矩阵语言之间切换，而不是只会算。
- 我能说清楚“为什么这个工具适合这道题”，而不只是套公式。
