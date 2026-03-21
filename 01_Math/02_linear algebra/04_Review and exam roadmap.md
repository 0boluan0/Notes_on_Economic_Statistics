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

## 资料现实情况

- 本地仓库里有三次 `Exam Review` 的 summary PDF，也有 `Final Course Review` 和 `Final Exam`。
- 本地仓库里没有 Exam 1 / Exam 2 / Exam 3 的试卷 PDF，因此本笔记会把复习路线重点放在 review summary、作业题和 final 上。
- 题目总入口：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]

## Unit I review

- 主笔记：[[01_Ax = b and the Four Subspaces]]
- Review PDF：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf|Exam 1 review]]
- 最关键的能力：
  - 用消元判断是否有解、是否唯一、是否有自由变量。
  - 从原矩阵中读出 column space 的基，从 rref 读出 nullspace 的基。
  - 用 rank 和四个基本子空间解释所有结论，而不是只写答案。
- 重点题型：
  - `Ax=0` 的通解
  - `Ax=b` 的解结构
  - basis / dimension / independence
  - incidence matrix 与网络流

## Unit II review

- 主笔记：[[02_Least Squares, Determinants and Eigenvalues]]
- Review PDF：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|Exam 2 review]]
- 最关键的能力：
  - 把 best fit 问题写成投影与 least squares。
  - 熟练使用 `P=A(A^TA)^{-1}A^T` 和 `A^TA\hat{x}=A^Tb`。
  - 用特征值和特征向量分析矩阵幂、迭代和长期行为。
- 重点题型：
  - projection matrix 的 rank、eigenvalues、column space
  - Gram-Schmidt 正交化
  - determinant 性质和 cofactor
  - [[Diagonalization|diagonalization]] 与 [[Markov Matrix|Markov]] 长期稳态

## Unit III review

- 主笔记：[[03_Positive Definite Matrices and Applications]]
- Review PDF：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf|Exam 3 review]]
- 最关键的能力：
  - 判断对称矩阵、正定矩阵、相似矩阵的结构含义。
  - 用 `e^{At}`、对角化或相似来分析动态系统。
  - 熟练解释 [[Singular Value Decomposition|SVD]] 的来源、意义和子空间信息。
- 重点题型：
  - symmetric / positive definite / semidefinite 判断
  - [[Similar Matrix|similar matrices]] 与 [[Jordan Form|Jordan]] 视角
  - [[Singular Value Decomposition|SVD]] 的手算与解释
  - differential equations 与矩阵指数

## Final course review

- Review PDF：[[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_FinalRevsum.pdf|Final course review]]
- Final exam：[[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_final_ex.pdf|final exam]] | [[MIT_OCW_18.06SC_PDF/01_Exams/MIT18_06SCF11_final_exs.pdf|solutions]]
- Final 的额外范围：
  - [[Linear Transformation]]
  - [[Change of Basis]]
  - [[Pseudoinverse]]
  - 复数特征值与复内积
- Final 的复习策略：
  - 先回 Unit I 的四个基本子空间框架
  - 再回 Unit II 的正交 / least squares / eigenvalues
  - 最后回 Unit III 的 symmetric / positive definite / [[Singular Value Decomposition|SVD]] / [[Linear Transformation|linear transformations]]
  - 做 final 前至少过一遍 `Final course review`

## 推荐复习顺序

1. 先读对应 unit note 的“本单元复习清单”。
2. 再看相应 `Exam Review` summary。
3. 回到 [[MIT_OCW_18.06SC_PDF/index|题目总索引]]，补做对应单元的作业题。
4. 最后用 final review 和 final exam 做总验收。

## 总复习 checklist

- 我能从 `Ax=b`、四个基本子空间、[[Orthogonal Projection|正交投影]]、[[Eigenvalues|特征值]]、[[Singular Value Decomposition|SVD]] 这五条主线把整门课串起来。
- 我知道什么时候该用 elimination，什么时候该用 [[Least Squares|least squares]]，什么时候该转向 eigen / [[Singular Value Decomposition|SVD]]。
- 我能在几何语言、代数语言和矩阵语言之间自由切换。
