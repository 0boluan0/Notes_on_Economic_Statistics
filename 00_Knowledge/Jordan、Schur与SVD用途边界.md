---
aliases:
  - "Jordan 标准形用于精确相似分类，Schur 分解用于数值特征问题，SVD 用于奇异方向几何"
  - Jordan, Schur, and SVD serve different purposes
  - Jordan、Schur 与 SVD 的用途边界
student_os: knowledge-atom
atom_id: LA-EIG-015
atom_set: eigenvalues-linear-dynamics
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jordan标准形]]"
  - "[[Schur分解]]"
  - "[[Schur分解存在性]]"
  - "[[奇异值分解]]"
related:
  - "[[Normal 矩阵谱定理]]"
  - "[[Jordan标准形的域条件]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 标准形用于精确相似分类，Schur 分解用于数值特征问题，SVD 用于奇异方向几何
<!-- bilingual-en:start -->
*Jordan form classifies exact similarity structure, Schur form supports numerical eigenproblems, and SVD describes singular-direction geometry*
<!-- bilingual-en:end -->

> [!summary] 三种分解各回答什么
> - Jordan 标准形：在精确代数中记录广义特征向量链和块尺寸，用于区分相似类；
> - Schur 分解：用正交或酉（unitary）相似变换把一般方阵约化为三角或准三角形，用于数值特征值与不变子空间计算；
> - SVD：用两个空间中的正交方向和非负奇异值描述矩形或方形线性映射，用于秩、条件数、最小二乘与低秩近似。
> <!-- bilingual-en:start -->
> Jordan, Schur, and SVD preserve and expose different structures, so none is a drop-in replacement for the others.
> <!-- bilingual-en:end -->

Jordan 块尺寸对微小扰动不稳定。矩阵
$$\begin{bmatrix}1&1\\0&1\end{bmatrix}$$
有一个二阶 Jordan 块；把右下角改成 $1+\varepsilon$ 后，任意非零 $\varepsilon$ 都产生两个不同特征值，使矩阵可对角化。浮点数据无法可靠判断一个极小差异究竟应当视为精确的零还是非零，因此直接计算 Jordan 块通常不是稳健的数值目标。

Schur 分解使用条件良好的正交或酉变换，并保留方阵的相似结构。LAPACK 对一般非对称特征问题先约化到 Hessenberg 形，再求 Schur 形。这样做不意味着每个特征向量都良态，而是避免把病态特征向量基强行当作主要数值表示。

SVD 则不做相似变换：$A=U\Sigma V^*$ 的左右方向来自不同空间，$\Sigma$ 上是奇异值而不是方阵的特征值。因此，“SVD 是更稳定的 Jordan 形”是错误类比；它回答的是另一类几何问题。
<!-- bilingual-en:start -->
Jordan block sizes are perturbation-sensitive, Schur form is the stable orthogonal or unitary reduction for square eigenproblems, and SVD studies singular values between input and output spaces.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 想数值计算一般非对称方阵的特征值与不变子空间，应先选 Jordan、Schur 还是 SVD？
>
> **答案：** Schur。Jordan 用于精确相似分类；SVD 研究奇异方向，不是一般方阵特征问题的相似约化。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf|MIT Lecture 28 transcript]]：核对 Jordan 结构对精确重根与微小扰动敏感。
- [LAPACK Users' Guide, Nonsymmetric Eigenproblems](https://www.netlib.org/lapack95/lug95/node34.html)：核对一般非对称特征问题的 Schur 路线。
- [LAPACK Users' Guide, Eigenvalues and Schur Factorization](https://www.netlib.org/lapack/lug/node50.html)：核对实、复 Schur 形。
- [[奇异值分解|两空间 SVD]]：核对 SVD 描述输入、输出奇异方向而非 Jordan 相似类。
