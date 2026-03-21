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

## 课程定位

- 这套笔记对应 MIT OpenCourseWare 的 18.06SC Linear Algebra（Fall 2011），课程骨架以官方 syllabus 为准。
- 官方 syllabus：<https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/syllabus/>
- 课程主线非常清晰：先从 `Ax=b`、消元、四个基本子空间出发，再进入 [[Orthogonality|正交]]、[[Least Squares|最小二乘]]、[[Determinant|行列式]]、[[Eigenvalues|特征值]]，最后收束到 [[Positive Definite Matrix|正定矩阵]]、[[Similar Matrix|相似]]、[[Singular Value Decomposition|SVD]]、[[Linear Transformation|线性变换]] 与 [[Pseudoinverse|伪逆]]。

## 本地资料入口

- 课程 PDF 总目录：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]
- 讲义 summary：`MIT_OCW_18.06SC_PDF/05_Session_Summaries`
- 练习与作业：`MIT_OCW_18.06SC_PDF/02_Exercises`、`03_Homework_Problems`、`04_Homework_Solutions`
- Final review / final exam：`MIT_OCW_18.06SC_PDF/01_Exams`
- 教材 1：[[MIT_OCW_18.06SC_PDF/99_Other/Introduction_to_Linear_Algebra(Strang).pdf|Introduction to Linear Algebra (Strang)]]
- 教材 2：[[MIT_OCW_18.06SC_PDF/99_Other/linear_algebra_and_its_applications_4th.pdf|Linear Algebra and Its Applications (4th)]]

## 正式课程笔记

- [[01_Ax = b and the Four Subspaces|01 Ax = b and the Four Subspaces]]
- [[02_Least Squares, Determinants and Eigenvalues|02 Least Squares, Determinants and Eigenvalues]]
- [[03_Positive Definite Matrices and Applications|03 Positive Definite Matrices and Applications]]
- [[04_Review and exam roadmap|04 Review and exam roadmap]]

## 学习顺序

- 先通读一个 unit note，理解该单元主线和 session 顺序。
- 每个 session 再回看对应的 summary PDF，确认定义、公式、例题和 Strang 的表述。
- 做题时同步使用 [[MIT_OCW_18.06SC_PDF/index|题目总索引]]，先做题、再看 solution。
- 遇到核心知识点时，优先跳转到 `00_factor` 中的知识卡；如果知识卡不够用，本轮已经补建或重建。

## 三个 unit 的主线

### Unit I: Ax = b and the Four Subspaces

- 核心问题：什么样的线性方程组有解，解为什么有这样的结构。
- 关键工具：消元、主元与自由变量、列空间、零空间、基、维数、四个基本子空间。
- 最重要的思维转换：把“解方程”看成“研究矩阵作用在空间上的结构”。

### Unit II: Least Squares, Determinants and Eigenvalues

- 核心问题：无解时怎样找最佳近似；方阵有哪些更深的代数不变量。
- 关键工具：[[Orthogonality|正交]]、[[Orthogonal Projection|投影]]、[[Least Squares|最小二乘]]、[[Gram-Schmidt Orthogonalization|Gram-Schmidt]]、[[Determinant|行列式]]、[[Eigenvalues|特征值]] 与 [[Diagonalization|对角化]]、[[Matrix Exponential|矩阵指数]]、[[Markov Matrix|Markov 矩阵]]。
- 最重要的思维转换：从“解得出来”转向“如何最好地逼近”和“矩阵长期行为是什么”。

### Unit III: Positive Definite Matrices and Applications

- 核心问题：哪些矩阵最稳定、最好用；非方阵和一般线性变换如何统一看待。
- 关键工具：[[Symmetric Matrix|对称矩阵]]、[[Positive Definite Matrix|正定矩阵]]、二次型、[[Similar Matrix|相似]] 与 [[Jordan Form|Jordan 形式]]、[[Singular Value Decomposition|SVD]]、[[Linear Transformation|变换矩阵]]、[[Change of Basis|换基]]、[[Pseudoinverse|伪逆]]。
- 最重要的思维转换：把前两单元的秩、行列式、特征值、正交结构统一起来。

## 相关知识卡入口

- Hub：[[Linear Algebra-hub|线性代数 Hub]]
- Unit I 常用：[[Vector Space]]、[[Subspace]]、[[Column Space]]、[[Null Space]]、[[Matrix Inverse]]、[[LU Decomposition]]、[[Permutation Matrix]]、[[Incidence Matrix]]
- Unit II 常用：[[Orthogonality]]、[[Orthogonal Projection]]、[[Projection Matrix]]、[[Least Squares]]、[[Orthogonal Matrix]]、[[Gram-Schmidt Orthogonalization]]、[[Determinant]]、[[Cramer's Rule]]、[[Diagonalization]]
- Unit III 常用：[[Eigenvalues]]、[[Eigenvectors]]、[[Positive Definite Matrix]]、[[Spectral Decomposition]]、[[Singular Value Decomposition]]、[[Linear Transformation]]、[[Change of Basis]]、[[Pseudoinverse]]

## 关于旧稿

- 目录下原有的 `00_perface.md`、`01_matrices and Gaussian Elimination.md`、`02_vector spaces and subspace.md` 保留。
- 它们在这轮被当作早期草稿和素材库使用，正式学习路径以本文件和 `01` 到 `04` 这 4 份主笔记为准。
