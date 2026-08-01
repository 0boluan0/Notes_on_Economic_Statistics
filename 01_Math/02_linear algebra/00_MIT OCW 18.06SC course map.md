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

> [!info] 课程来源
> 本套笔记对应 MIT OpenCourseWare **18.06SC Linear Algebra, Fall 2011**，由 Gilbert Strang 教授主讲。
>
> - [Official syllabus](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/syllabus/)
> - [Official resource index](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/resource-index/)
> - [Official problem sets](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/problem-sets/)
> - [Official exams](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/exams/)
> <!-- bilingual-en:start -->
> These notes correspond to MIT OpenCourseWare **18.06SC Linear Algebra, Fall 2011**, taught by Professor Gilbert Strang.
> <!-- bilingual-en:end -->

## 从哪里开始
<!-- bilingual-en:start -->
*Where to begin*
<!-- bilingual-en:end -->

- 第一次学习：按 Unit I → Unit II → Unit III → Final 的顺序阅读，每个 Session 完成正文、自检和对应 Homework。
- 复习某个主题：使用下方“题型入口”直接跳到对应 Unit。
- 考前复习：进入 [[04_Review and exam roadmap|Final Course Review and Final Exam]]。
- 查找原始资料：进入 [[MIT_OCW_18.06SC_PDF/index|PDF and transcript index]]。

## 四篇主笔记
<!-- bilingual-en:start -->
*The four main notes*
<!-- bilingual-en:end -->

| 顺序 | 主笔记 | 核心问题 | 结尾验收 |
|---|---|---|---|
| 1 | [[01_Ax = b and the Four Subspaces]] | $Ax=b$ 何时有解、何时唯一，秩与四个基本子空间怎样统一这些现象 | Exam 1 |
| 2 | [[02_Least Squares, Determinants and Eigenvalues]] | 无精确解时如何最佳逼近；方阵怎样缩放体积并保留特殊方向 | Exam 2 |
| 3 | [[03_Positive Definite Matrices and Applications]] | 对称、正定、Jordan、SVD 与伪逆怎样描述更一般的矩阵 | Exam 3 |
| 4 | [[04_Review and exam roadmap]] | 把全课程压成题型判断系统，并完成九道 Final Exam | Final Exam |
<!-- bilingual-en:start -->
| Order | Main note | Central question | Final checkpoint |
|---|---|---|---|
| 1 | [[01_Ax = b and the Four Subspaces]] | When does $Ax=b$ have a solution, when is it unique, and how do rank and the four fundamental subspaces unify these cases? | Exam 1 |
| 2 | [[02_Least Squares, Determinants and Eigenvalues]] | How do we find the best approximation when no exact solution exists, and how does a square matrix scale volume while preserving special directions? | Exam 2 |
| 3 | [[03_Positive Definite Matrices and Applications]] | How do symmetry, positive definiteness, Jordan form, SVD, and the pseudoinverse describe more general matrices? | Exam 3 |
| 4 | [[04_Review and exam roadmap]] | Compress the whole course into a problem-type decision system and complete the nine problems on the final exam. | Final Exam |
<!-- bilingual-en:end -->

## 全课程的结构主线
<!-- bilingual-en:start -->
*Structural threads across the course*
<!-- bilingual-en:end -->

### 主线 1：存在性与唯一性
<!-- bilingual-en:start -->
*Thread 1: existence and uniqueness*
<!-- bilingual-en:end -->

设 $A\in\mathbb F^{m\times n}$。求解 $Ax=b$ 时必须把两个问题分开：
<!-- bilingual-en:start -->
Let $A\in\mathbb F^{m\times n}$. When solving $Ax=b$, keep two questions separate:
<!-- bilingual-en:end -->

$$
b\in C(A)
\quad\Longleftrightarrow\quad
\text{至少有一个解},
$$

$$
N(A)=\{0\}
\quad\Longleftrightarrow\quad
\text{相容时解唯一}.
$$

消元负责计算，[[线性方程组与四个基本子空间#四个基本子空间|列空间]]与[[线性方程组与四个基本子空间#四个基本子空间|零空间]]负责解释。
<!-- bilingual-en:start -->
Elimination performs the calculation; the column space and nullspace explain the result.
<!-- bilingual-en:end -->

### 主线 2：精确解到最佳近似
<!-- bilingual-en:start -->
*Thread 2: from exact solutions to best approximations*
<!-- bilingual-en:end -->

若 $b\notin C(A)$，方程无解，但仍可寻找最接近 $b$ 的列空间向量：
<!-- bilingual-en:start -->
If $b\notin C(A)$, the equation has no exact solution, but we can still find the vector in the column space closest to $b$:
<!-- bilingual-en:end -->

$$
p=A\hat x,
\qquad
A^T(b-A\hat x)=0.
$$

这条线依次经过正交、投影、[[正交投影与最小二乘#最小二乘与正规方程|最小二乘]]、Gram--Schmidt 与 QR。
<!-- bilingual-en:start -->
This thread passes through orthogonality, projection, least squares, Gram--Schmidt, and QR in sequence.
<!-- bilingual-en:end -->

### 主线 3：反复作用与自然坐标
<!-- bilingual-en:start -->
*Thread 3: repeated action and natural coordinates*
<!-- bilingual-en:end -->

[[特征值、对角化与线性动力系统#特征值与特征向量|特征值]]寻找在 $A$ 作用下方向不变的向量；对角化把矩阵幂和 $e^{At}$ 解耦。对称矩阵拥有正交特征基，缺陷方阵需要 Jordan 结构，而任意矩形矩阵最终都可使用 [[奇异值分解与低秩近似#SVD 的三层结构|SVD]]。
<!-- bilingual-en:start -->
Eigenvalues identify vectors whose directions remain unchanged under $A$; diagonalization decouples matrix powers and $e^{At}$. Symmetric matrices have orthogonal eigenbases, defective square matrices require Jordan structure, and every rectangular matrix admits an SVD.
<!-- bilingual-en:end -->

### 主线 4：同一个矩阵的四种身份
<!-- bilingual-en:start -->
*Thread 4: four identities of the same matrix*
<!-- bilingual-en:end -->

矩阵可以同时被看作：
<!-- bilingual-en:start -->
A matrix can simultaneously be viewed as:
<!-- bilingual-en:end -->

1. 线性方程组的系数；
2. 一组列向量；
3. 线性变换在给定基下的坐标表示；
4. 将输入空间的方向拉伸、旋转、压扁到输出空间的规则。
<!-- bilingual-en:start -->

&nbsp;
**1.** the coefficient array of a linear system;<br>
**2.** a collection of column vectors;<br>
**3.** the coordinate representation of a linear transformation in chosen bases;<br>
**4.** a rule that stretches, rotates, and flattens directions from the input space into the output space.<br>
<!-- bilingual-en:end -->

真正理解线性代数，意味着能在这四种语言之间切换，而不是只会执行矩阵运算。
<!-- bilingual-en:start -->
Understanding linear algebra means being able to move among these four languages, not merely performing matrix operations.
<!-- bilingual-en:end -->

## 官方课程顺序

### Unit I：$Ax=b$ and the Four Subspaces

> [!warning] Unit I 的文件编号不是学习顺序
> `Ses1.13` 是 Overview，官网把它放在 Geometry 后、Elimination 前。不能把本地 PDF 按文件名机械排序。
> <!-- bilingual-en:start -->
> `Ses1.13` is the overview. The official site places it after Geometry and before Elimination, so the local PDFs must not be ordered mechanically by filename.
> <!-- bilingual-en:end -->

| 笔记位置 | 官方主题 | 本地 summary | 作业 |
|---:|---|---|---|
| 1.1 | The Geometry of Linear Equations | `Ses1.1` | `Ses1.1` |
| 1.2 | An Overview of Key Ideas | `Ses1.13` | 无 |
| 1.3 | Elimination with Matrices | `Ses1.2` | `Ses1.2` |
| 1.4 | Multiplication and Inverse Matrices | `Ses1.3` | `Ses1.3` |
| 1.5 | Factorization into $A=LU$ | `Ses1.4` | `Ses1.4` |
| 1.6 | Transposes, Permutations, Vector Spaces | `Ses1.5` | `Ses1.5` |
| 1.7 | Column Space and Nullspace | `Ses1.6` | `Ses1.6` |
| 1.8 | Solving $Ax=0$ | `Ses1.7` | `Ses1.7` |
| 1.9 | Solving $Ax=b$ | `Ses1.8` | `Ses1.8` |
| 1.10 | Independence, Basis and Dimension | `Ses1.9` | `Ses1.9` |
| 1.11 | Four Fundamental Subspaces | `Ses1.10` | `Ses1.10` |
| 1.12 | Matrix Spaces; Rank 1; Small World Graphs | `Ses1.11` | `Ses1.11` |
| 1.13 | Graphs, Networks, Incidence Matrices | `Ses1.12` | `Ses1.12` |
| 1.14 | Exam 1 Review | `Ses1.14` | 无 |
| Exam 1 | Unit 1 Exam | `ex1/ex1s` | 完整题解在 Unit I |

### Unit II：Least Squares, Determinants and Eigenvalues

| 笔记位置 | 官方主题 |
|---:|---|
| 2.1 | Orthogonal Vectors and Subspaces |
| 2.2 | Projections onto Subspaces |
| 2.3 | Projection Matrices and Least Squares |
| 2.4 | Orthogonal Matrices and Gram--Schmidt |
| 2.5 | Properties of Determinants |
| 2.6 | Determinant Formulas and Cofactors |
| 2.7 | Cramer's Rule, Inverse Matrix and Volume |
| 2.8 | Eigenvalues and Eigenvectors |
| 2.9 | Diagonalization and Powers of $A$ |
| 2.10 | Differential Equations and $e^{At}$ |
| 2.11 | Markov Matrices; Fourier Series |
| 2.12 | Exam 2 Review |
| Exam 2 | Unit 2 Exam，完整题解在 Unit II |

### Unit III：Positive Definite Matrices and Applications

| 笔记位置 | 官方主题 | 原课堂讲次 |
|---:|---|---:|
| 3.1 | Symmetric Matrices and Positive Definiteness | Lecture 25 |
| 3.2 | Complex Matrices; FFT | Lecture 26 |
| 3.3 | Positive Definite Matrices and Minima | Lecture 27 |
| 3.4 | Similar Matrices and Jordan Form | Lecture 28 |
| 3.5 | Singular Value Decomposition | Lecture 29 |
| 3.6 | Linear Transformations and their Matrices | Lecture 30 |
| 3.7 | Change of Basis; Image Compression | Lecture 31 |
| 3.8 | Left and Right Inverses; Pseudoinverse | Lecture 33 |
| 3.9 | Exam 3 Review | Lecture 32 |
| Exam 3 | Unit 3 Exam，重点范围截至 SVD | — |

OCW Scholar 版主动把 Lecture 33 移到 Exam 3 Review 前，因此本笔记遵循官网导航，而不是原始录像讲次的数值顺序。
<!-- bilingual-en:start -->
The OCW Scholar version deliberately moves Lecture 33 before the Exam 3 Review. These notes therefore follow the official site navigation rather than the numerical order of the original recorded lectures.
<!-- bilingual-en:end -->

### Course close

1. Final Course Review（Lecture 34）；
2. Final Exam：3 小时、九道综合题；
3. 2023 年的 Final 18.06 Lecture 是后加退休讲座，不属于 Fall 2011 主课程，本套笔记不把它算作 Session。

## 本地资料体系

| 目录 | 数量 | 用途 |
|---|---:|---|
| `01_Exams` | 3 | Final Review、Final Exam、Final Solutions |
| `02_Exercises` | 6 | 实际为 Exam 1–3 及答案，保留原文件位置 |
| `03_Homework_Problems` | 31 | 每个常规 Session 的题目 |
| `04_Homework_Solutions` | 31 | 与 Homework 一一对应的官方答案 |
| `05_Session_Summaries` | 35 | 全部 Session summary |
| `06_Lecture_Transcripts` | 36 | 35 个 Session 加 Final Review 的讲课转录 |
| `07_Recitation_Transcripts` | 36 | 对应 problem-solving / recitation 转录 |
| `08_Supplementary_Transcripts` | 2 | 课程介绍与 Strang 教学访谈 |
| `99_Books` | 3 | 本地教材 |

全部 transcript 已按 Unit、Session 和 Lecture/Recitation 语义化命名；每节正文末尾直接链接对应文件。
<!-- bilingual-en:start -->
Every transcript has been named semantically by unit, session, and lecture or recitation; each session note links directly to its corresponding files at the end.
<!-- bilingual-en:end -->

## 题型入口
<!-- bilingual-en:start -->
*Entry points by problem type*
<!-- bilingual-en:end -->

| 想解决的问题 | 进入哪里 | 关键工具 |
|---|---|---|
| 解的存在性、唯一性、自由度 | [[01_Ax = b and the Four Subspaces]] | elimination、RREF、rank、four subspaces |
| 最接近的向量或直线 | [[02_Least Squares, Determinants and Eigenvalues]] | projection、normal equations、QR |
| 体积、可逆性与余子式 | [[02_Least Squares, Determinants and Eigenvalues]] | determinant、cofactor、Cramer |
| 矩阵幂、稳态、微分方程 | [[02_Least Squares, Determinants and Eigenvalues]] | eigen、diagonalization、$e^{At}$ |
| 二次型、极小值与对称结构 | [[03_Positive Definite Matrices and Applications]] | spectral theorem、positive definiteness |
| 缺陷方阵、矩形矩阵或压缩 | [[03_Positive Definite Matrices and Applications]] | Jordan、SVD、pseudoinverse |
| 全课程综合题 | [[04_Review and exam roadmap]] | 题型识别、条件检查、交叉验算 |
<!-- bilingual-en:start -->
| Problem to solve | Where to go | Key tools |
|---|---|---|
| Existence, uniqueness, and degrees of freedom of solutions | [[01_Ax = b and the Four Subspaces]] | elimination, RREF, rank, four subspaces |
| Closest vector or line | [[02_Least Squares, Determinants and Eigenvalues]] | projection, normal equations, QR |
| Volume, invertibility, and cofactors | [[02_Least Squares, Determinants and Eigenvalues]] | determinant, cofactor, Cramer's rule |
| Matrix powers, steady states, and differential equations | [[02_Least Squares, Determinants and Eigenvalues]] | eigenvalues, diagonalization, $e^{At}$ |
| Quadratic forms, minima, and symmetric structure | [[03_Positive Definite Matrices and Applications]] | spectral theorem, positive definiteness |
| Defective square matrices, rectangular matrices, or compression | [[03_Positive Definite Matrices and Applications]] | Jordan form, SVD, pseudoinverse |
| Integrated whole-course problems | [[04_Review and exam roadmap]] | problem-type recognition, condition checks, cross-checking |
<!-- bilingual-en:end -->

## 配套知识卡入口

- Hub：[[线性代数 Course Atlas|Linear Algebra Hub]]
- 题型选择：[[线性代数 Course Atlas|Linear Algebra Problem-Type Map]]
- 分解选择：[[线性代数 Course Atlas|Choosing Matrix Decompositions]]
- 解结构：[[线性方程组与四个基本子空间#可解性与完整解|Linear system solution structure]]
- 四子空间读取：[[线性方程组与四个基本子空间#四个基本子空间|Reading the Four Fundamental Subspaces from RREF]]
- 最小二乘：[[正交投影与最小二乘#最小二乘与正规方程|Least Squares via Normal Equations]]
- 正定判别：[[对称矩阵与正定二次型#二次型与正定性|Testing Positive Definiteness]]

**课程知识链：**线性组合 → $Ax=b$ → 消元与子空间 → 正交投影 → 行列式与特征结构 → 对称正定 → SVD 与伪逆。
<!-- bilingual-en:start -->
**Course knowledge chain:** linear combinations → $Ax=b$ → elimination and subspaces → orthogonal projection → determinants and eigenstructure → symmetry and positive definiteness → SVD and pseudoinverse.
<!-- bilingual-en:end -->
