---
aliases:
  - MIT 18.06SC course map
  - 线性代数课程总览
  - MIT OCW 18.06SC 课程地图
tags:
  - 线性代数
  - mit-ocw
  - course-note
---

# MIT OCW 18.06SC course map

## 如果你一时想不起整门课，先看这页

- 这门课不是“很多章节”，而是三条连续主线：
  - `Ax=b` 与四个基本子空间：先搞清楚什么叫有解、唯一解、自由变量、秩。
  - 正交与最佳逼近：当精确解不存在时，转到 projection / least squares / QR。
  - 特征结构与矩阵分解：用 determinant、eigenvalues、positive definite、Jordan、SVD 解释矩阵的长期行为和标准形。
- 复习顺序默认是：
  1. 先读本页的 `三条主线` 和 `单元速览`。
  2. 再跳到对应 unit note 的 `单元速览` 和 `Session 回忆索引`。
  3. 最后用 [[04_Review and exam roadmap|Review and exam roadmap]] 按题型回收。

## 课程定位

- 这套笔记对应 MIT OpenCourseWare 的 18.06SC Linear Algebra（Fall 2011），课程骨架以官方 syllabus 为准。
- 官方 syllabus：<https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/syllabus/>
- 学完后你应该能把一切题目先翻译成这几个问题之一：
  - `Ax=b` 有没有解，为什么。
  - 哪个子空间 / 哪个方向在控制结果。
  - 当没有精确解时，最佳逼近是什么。
  - 这个矩阵的长期行为由什么特征结构决定。
  - 应该用 inverse、least squares、diagonalization、Jordan 还是 SVD。

## 三条主线

### 主线一：从解方程到子空间结构

- 起点：[[01_Ax = b and the Four Subspaces|Unit I]]
- 核心问题：什么决定 `Ax=b` 的可解性、唯一性与自由度。
- 必会语言：pivot、rank、nullity、column space、null space、basis、dimension。
- 看到这类题时要先想到：
  - `b` 是否在 [[Column Space]] 里。
  - 解是否等于“particular solution + [[Null Space]]”。
  - 该矩阵的四个基本子空间分别在哪个空间里。

### 主线二：从正交到最佳逼近

- 起点：[[02_Least Squares, Determinants and Eigenvalues|Unit II 前半]]
- 核心问题：原系统无解时，怎样找到最佳近似；为什么正交是最佳逼近的语言。
- 必会语言：[[Orthogonality]]、[[Orthogonal Projection]]、[[Least Squares]]、[[Projection Matrix]]、[[Gram-Schmidt Orthogonalization]]、QR。
- 看到这类题时要先想到：
  - 误差必须与目标子空间正交。
  - 正规方程来自 `A^T(b-A\hat{x})=0`。
  - 如果题目在问“best fit / closest / minimum error”，几乎都该走 projection / least squares 路线。

### 主线三：从不变量到矩阵分解

- 起点：[[02_Least Squares, Determinants and Eigenvalues|Unit II 后半]] + [[03_Positive Definite Matrices and Applications|Unit III]]
- 核心问题：矩阵如何缩放体积、保留方向、决定长期行为，以及在一般情形下如何被标准化。
- 必会语言：[[Determinant]]、[[Eigenvalues]]、[[Diagonalization]]、[[Matrix Exponential]]、[[Positive Definite Matrix]]、[[Jordan Form]]、[[Singular Value Decomposition]]、[[Pseudoinverse]]。
- 看到这类题时要先想到：
  - determinant 管体积和奇异性。
  - eigen / diagonalization 管反复作用和动力系统。
  - symmetric / positive definite 是“最好处理”的矩阵。
  - 一般矩阵最后统一到 SVD 与 pseudoinverse。

## 单元速览

| Unit | 你在解决什么问题 | 你最后应该会什么 | 主笔记 |
| --- | --- | --- | --- |
| I | 什么样的线性系统有解、为什么不唯一、自由度从哪来 | 消元、读主元与自由变量、解释四个基本子空间 | [[01_Ax = b and the Four Subspaces]] |
| II | 没有精确解时怎样做最佳逼近；方阵有哪些重要不变量 | projection / least squares / QR / determinant / eigen / matrix exponential | [[02_Least Squares, Determinants and Eigenvalues]] |
| III | 怎样把前两单元统一到更强的结构视角 | positive definite、similarity / Jordan、SVD、linear transformation、change of basis、pseudoinverse | [[03_Positive Definite Matrices and Applications]] |

## Session 回忆索引

### Unit I：Ax = b and the Four Subspaces

- 1.1 几何图像：row picture / column picture / matrix picture。
- 1.2 课程路线图：为什么 elimination、subspace、orthogonality、eigen、SVD 会串成一门课。
- 1.3-1.5 算法线：消元、主元、LU、可逆性。
- 1.6-1.10 结构线：vector spaces、column space、null space、special solutions、basis、dimension。
- 1.11-1.13 收束：four fundamental subspaces、rank-1、graphs / networks / incidence matrices。
- 1.14 考前闭环：把“解结构 -> 子空间 -> 维数 -> 网络”串成一套解释。

### Unit II：Least Squares, Determinants and Eigenvalues

- 2.1-2.4 正交主线：orthogonality -> projection -> least squares -> Gram-Schmidt / QR。
- 2.5-2.7 determinant 主线：三条定义性性质、cofactor、inverse 与 volume。
- 2.8-2.10 eigen 主线：eigenvalues -> diagonalization -> powers of A / $e^{At}$。
- 2.11 应用收束：Markov / Fourier 都是在“选对基底”后解耦。
- 2.12 考前闭环：题型在 projection、determinant、eigen 三块里来回切。

### Unit III：Positive Definite Matrices and Applications

- 3.1-3.3 对称与正定：spectral decomposition、quadratic form、minimum。
- 3.4-3.5 标准形：similarity / Jordan 处理一般方阵，SVD 处理任意矩阵。
- 3.6-3.7 变换与换基：矩阵是线性变换在某组坐标下的表示。
- 3.8 逆的推广：left inverse / right inverse / pseudoinverse。
- 3.9 考前闭环：symmetric、positive definite、Jordan、SVD、pseudoinverse 同时归到“找对坐标和结构”。

## 本地资料入口

- 课程 PDF 总目录：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]
- 讲义 summary：`MIT_OCW_18.06SC_PDF/05_Session_Summaries`
- 练习与作业：`MIT_OCW_18.06SC_PDF/02_Exercises`、`03_Homework_Problems`、`04_Homework_Solutions`
- Final review / final exam：`MIT_OCW_18.06SC_PDF/01_Exams`
- 教材 1：[[Introduction_to_Linear_Algebra(Strang).pdf|Introduction to Linear Algebra (Strang)]]
- 教材 2：[[linear_algebra_and_its_applications_4th.pdf|Linear Algebra and Its Applications (4th)]]

## 主笔记入口

- [[01_Ax = b and the Four Subspaces|01 Ax = b and the Four Subspaces]]
- [[02_Least Squares, Determinants and Eigenvalues|02 Least Squares, Determinants and Eigenvalues]]
- [[03_Positive Definite Matrices and Applications|03 Positive Definite Matrices and Applications]]
- [[04_Review and exam roadmap|04 Review and exam roadmap]]

## 题型入口

- 解结构与四个基本子空间：[[01_Ax = b and the Four Subspaces]]
- 投影、最小二乘、QR：[[02_Least Squares, Determinants and Eigenvalues#Session 2.1 Orthogonal vectors and subspaces|Unit II 正交主线]]
- determinant、cofactor、volume：[[02_Least Squares, Determinants and Eigenvalues#Session 2.5 Properties of determinants|Unit II determinant 主线]]
- eigen、diagonalization、$e^{At}$：[[02_Least Squares, Determinants and Eigenvalues#Session 2.8 Eigenvalues and eigenvectors|Unit II eigen 主线]]
- positive definite、Jordan、SVD、pseudoinverse：[[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Unit III 结构主线]]

## 配套卡片入口

- Hub：[[Linear Algebra-hub|线性代数 Hub]]
- 方法卡：
  - [[Gram-Schmidt Orthogonalization]]
  - [[Reading the Four Fundamental Subspaces from RREF]]
  - [[Least Squares via Normal Equations]]
  - [[Testing Positive Definiteness]]
  - [[Choosing Matrix Decompositions]]
  - [[Linear Algebra Problem-Type Map]]
- 核心概念：
  - [[Column Space]]
  - [[Null Space]]
  - [[Orthogonal Projection]]
  - [[Least Squares]]
  - [[Projection Matrix]]
  - [[Determinant]]
  - [[Eigenvalues]]
  - [[Positive Definite Matrix]]
  - [[Singular Value Decomposition]]
  - [[Pseudoinverse]]

## 使用建议

- 想恢复课程全貌：先看本页，再看各 unit note 的 `单元速览`。
- 想按题型复习：直接跳 [[04_Review and exam roadmap|Review and exam roadmap]]。
- 想补概念：从 [[Linear Algebra-hub|hub]] 进卡片，而不是回正文里搜。

## 关于旧稿

- 目录下原有的 `00_perface.md`、`01_matrices and Gaussian Elimination.md`、`02_vector spaces and subspace.md` 保留为素材。
- 正式学习路径仍以本文件和 `01` 到 `04` 这 4 份主笔记为准。
