---
aliases:
- Frobenius norm from singular values
student_os: knowledge-atom
atom_id: LA-SVD-008
atom_set: singular-value-decomposition-low-rank
atom_type: identity
status: source-checked
requires:
- '[[奇异值]]'
- '[[Frobenius范数]]'
- '[[迹与特征值总和]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[奇异值能量比]]'
- '[[截断SVD最佳低秩近似]]'
related:
- '[[谱范数与最大奇异值]]'
---

<!-- bilingual-en:start -->
# Frobenius norm and singular values
<!-- bilingual-en:end -->

*Frobenius 范数与奇异值*

<!-- bilingual-en:start -->
The [[Frobenius范数|Frobenius norm]] is found by squaring every entry’s magnitude, adding, and taking a square root. The same number can be found from the [[奇异值|singular values]], the nonnegative stretch amounts in an SVD: square them all, add, and take the square root. This connects a total measured entry by entry with a total measured along perpendicular stretch directions.
<!-- bilingual-en:end -->

[[Frobenius范数]]的算法是：每个元素的模先平方，求和，再开平方。同一个数也可以从[[奇异值]]，即 SVD 中的非负伸缩量得到：把所有奇异值平方，求和，再开平方。这把逐元素衡量的整体大小与沿相互垂直伸缩方向衡量的整体大小联系起来。

> [!example] The same total in two descriptions · 两种描述给出同一个总量
> <!-- bilingual-en:start -->
> For the [[对角矩阵|diagonal matrix]] $A=\operatorname{diag}(4,3,0)$ (main diagonal $4,3,0$; all other entries zero), the singular values are $4,3,0$. The total squared magnitude is $16+9=25$, so the Frobenius norm is $5$. The [[谱范数与最大奇异值|spectral norm]] is instead $4$, because it keeps only the largest stretch.
> <!-- bilingual-en:end -->
>
> 对[[对角矩阵]] $A=\operatorname{diag}(4,3,0)$（主对角线为 $4,3,0$，其他元素为零），奇异值为 $4,3,0$。模平方总量为 $16+9=25$，所以 Frobenius 范数为 $5$；[[谱范数与最大奇异值|谱范数]]则为 $4$，因为它只保留最大伸缩量。

<!-- bilingual-en:start -->
## Write the identity with its square
<!-- bilingual-en:end -->

*把范数平方写清楚*

<!-- bilingual-en:start -->
Let $A\in\mathbb F^{m\times n}$ for positive integers $m,n$, where $\mathbb F$ is $\mathbb R$ or $\mathbb C$. Put $p=\min(m,n)$ and list all $p$ singular values, including repeated values and zeros. The identity is
<!-- bilingual-en:end -->

设 $A\in\mathbb F^{m\times n}$，$m,n$ 为正整数，$\mathbb F$ 为 $\mathbb R$ 或 $\mathbb C$。令 $p=\min(m,n)$，列出全部 $p$ 个奇异值，包括重复值和零。恒等式为

> [!important] Squared total and norm · 平方总量与范数
> <!-- bilingual-en:start -->
> The trace $\operatorname{tr}(M)$ of a square matrix is the sum of its diagonal entries. The star in $A^*$ means transpose and complex-conjugate every entry; for a real matrix it is simply $A^T$.
> <!-- bilingual-en:end -->
>
> 方阵的迹 $\operatorname{tr}(M)$ 是其对角元素之和；$A^*$ 表示转置并对每个元素取复共轭，实矩阵时就是 $A^T$。
>
> $$\|A\|_F^2=\operatorname{tr}(A^*A)=\sum_{i=1}^{p}\sigma_i^2,\qquad \|A\|_F=\sqrt{\sum_{i=1}^{p}\sigma_i^2}.$$

<!-- bilingual-en:start -->
## Why the trace adds the entry squares
<!-- bilingual-en:end -->

*为什么迹会把元素模平方加起来*

<!-- bilingual-en:start -->
The product $A^*A$ is an $n\times n$ square matrix, even when $A$ is rectangular. By [[矩阵乘法|matrix multiplication]], its $k$th diagonal entry pairs column $k$ of $A$ with itself:
<!-- bilingual-en:end -->

即使 $A$ 是矩形，$A^*A$ 也是 $n\times n$ 方阵。按[[矩阵乘法]]，它的第 $k$ 个对角元素把 $A$ 的第 $k$ 列与自身配对：

$$(A^*A)_{kk}=\sum_{j=1}^{m}\overline{a_{jk}}a_{jk}=\sum_{j=1}^{m}|a_{jk}|^2.$$

<!-- bilingual-en:start -->
Adding these diagonal entries over $k=1,\ldots,n$ counts every entry square exactly once. For the matrix $B=\begin{bmatrix}1&2\\-2&0\end{bmatrix}$, the two column totals are $1+4=5$ and $4+0=4$. Thus $\operatorname{tr}(B^TB)=9=\|B\|_F^2$.
<!-- bilingual-en:end -->

再对 $k=1,\ldots,n$ 把这些对角元素相加，就把每个元素的模平方恰好数一遍。对 $B=\begin{bmatrix}1&2\\-2&0\end{bmatrix}$，两列的平方总量分别为 $1+4=5$、$4+0=4$，所以 $\operatorname{tr}(B^TB)=9=\|B\|_F^2$。

<!-- bilingual-en:start -->
## Why the same trace adds the singular squares
<!-- bilingual-en:end -->

*为什么同一个迹又等于奇异值平方和*

<!-- bilingual-en:start -->
A full [[奇异值分解|SVD]] writes $A=U\Sigma V^*$ with $U$ an $m\times m$ and $V$ an $n\times n$ [[Unitary 矩阵|unitary matrix]], so $U^*U=I$ and $V^*V=I$. The diagonal factor $\Sigma$ has size $m\times n$. Multiplication gives the [[奇异值与Gram谱|Gram spectral identity]]
<!-- bilingual-en:end -->

完整[[奇异值分解|SVD]]写成 $A=U\Sigma V^*$，其中 $U$ 为 $m\times m$、$V$ 为 $n\times n$ 的[[Unitary 矩阵|酉矩阵]]，满足 $U^*U=I$、$V^*V=I$；对角因子 $\Sigma$ 为 $m\times n$。相乘得到[[奇异值与Gram谱|Gram 谱恒等式]]

$$A^*A=V\Sigma^*\Sigma V^*.$$

<!-- bilingual-en:start -->
The diagonal matrix $\Sigma^*\Sigma$ has entries $\sigma_1^2,\ldots,\sigma_p^2$, followed by $n-p$ extra zeros if $n>p$. These are the eigenvalues of $A^*A$, counted with multiplicity. By [[迹与特征值总和|the trace–eigenvalue identity]], their sum is its trace. Combining this with the entry calculation proves the formula. A repeated singular value contributes once for each occurrence; zeros contribute nothing.
<!-- bilingual-en:end -->

对角矩阵 $\Sigma^*\Sigma$ 的元素为 $\sigma_1^2,\ldots,\sigma_p^2$，若 $n>p$，后面还有 $n-p$ 个零。这些是按重数计算的 $A^*A$ 特征值。根据[[迹与特征值总和]]，其总和就是迹，再与逐元素的计算相接，便证明了公式。重复的奇异值每出现一次就计算一次，零则贡献零。

<!-- bilingual-en:start -->
For example, a $2\times3$ matrix $[\operatorname{diag}(4,3)\ \ 0]$ has two singular values $4,3$, while $A^*A$ has three eigenvalues $16,9,0$. Both calculations still give squared Frobenius norm $25$. The zero matrix also satisfies the identity, with all values zero.
<!-- bilingual-en:end -->

例如 $2\times3$ 矩阵 $[\operatorname{diag}(4,3)\ \ 0]$ 只有两个奇异值 $4,3$，而 $A^*A$ 有三个特征值 $16,9,0$；两种算法仍然给出 Frobenius 范数平方 $25$。零矩阵也满足该恒等式，此时所有量均为零。

> [!warning] Do not drop squares, roots or conjugates · 不要遗漏平方、根号或共轭
> <!-- bilingual-en:start -->
> Nonzero singular values $5,2,2$ give $\|A\|_F^2=25+4+4=33$ and $\|A\|_F=\sqrt{33}$, not $9$ or $33$. For complex matrices, $A^TA$ cannot replace $A^*A$: for $A=[i]$, the former is $[-1]$, while the latter is $[1]$ and the norm is $1$.
> <!-- bilingual-en:end -->
>
> 非零奇异值为 $5,2,2$ 时，$\|A\|_F^2=25+4+4=33$，$\|A\|_F=\sqrt{33}$，不是 $9$，也不是 $33$。复矩阵不能把 $A^*A$ 换成 $A^TA$：若 $A=[i]$，后者为 $[-1]$，前者为 $[1]$，范数为 $1$。

<!-- bilingual-en:start -->
The formula also explains the language of squared “energy” in [[奇异值能量比|singular-value energy ratios]]: the total being divided up is $\|A\|_F^2$. For an approximation error matrix $E=A-B$, the same identity can be applied to the singular values of $E$ itself. Choosing the best low-rank approximation is the separate result in [[截断SVD最佳低秩近似|truncated-SVD approximation]].
<!-- bilingual-en:end -->

该公式也解释了[[奇异值能量比]]为什么使用平方“能量”：分配的总量是 $\|A\|_F^2$。对近似误差矩阵 $E=A-B$，同一个恒等式可以用于 $E$ 自己的奇异值；如何选出最佳低秩近似则是[[截断SVD最佳低秩近似]]中的独立结论。

## 来源与核验

<!-- bilingual-en:start -->
- [Boyd and Vandenberghe, *Convex Optimization*, Appendix A.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), A.1.1, printed pp. 633–634 (PDF pp. 647–648): entrywise Frobenius definition and trace expression. [Axler, *Linear Algebra Done Right*, 4th ed.](https://linear.axler.net/LADR4e.pdf), 8.47–8.52, printed pp. 326–328 (PDF pp. 340–342): trace and its eigenvalue sum.
<!-- bilingual-en:end -->

- Boyd 与 Vandenberghe A.1.1，印刷页 633–634（PDF 647–648）核对逐元素 Frobenius 定义与迹表达；Axler 8.47–8.52，印刷页 326–328（PDF 340–342）核对迹及其特征值求和性质。

<!-- bilingual-en:start -->
- [LAPACK Users’ Guide, Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html): full real/complex rectangular SVD conventions; [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf#page=1|MIT 18.06SC Session 3.5, pp. 1–3]] checks the course Gram construction. The rectangular Gram formula and zero counts above are explicitly derived with full factor dimensions.
<!-- bilingual-en:end -->

- LAPACK 核对实、复矩形 SVD 的完整约定；MIT 第 3.5 节第 1–3 页核对课程 Gram 构造。正文按完整因子尺寸明确推导矩形 Gram 公式及零值数量。

<!-- bilingual-en:start -->
- [Cornell CS 6241, SVD and other low-rank decompositions](https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-02-13.html), SVD norm list: explicitly states the squared Frobenius norm as the sum of squared singular values. All numerical examples and the complex one-entry boundary are checked directly.
<!-- bilingual-en:end -->

- Cornell 的 SVD 范数列表明确给出 Frobenius 范数平方等于奇异值平方和；数值例子与单元素复矩阵边界直接核验。
