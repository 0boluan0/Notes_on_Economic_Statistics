---
aliases:
- Frobenius norm
student_os: knowledge-atom
atom_id: LA-SVD-017
atom_set: singular-value-decomposition-low-rank
atom_type: definition
status: source-checked
requires:
- '[[欧氏范数]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[Frobenius范数与奇异值]]'
- '[[奇异值能量比]]'
- '[[截断SVD最佳低秩近似]]'
related:
- '[[谱范数]]'
---

<!-- bilingual-en:start -->
# Frobenius norm
<!-- bilingual-en:end -->

*Frobenius 范数*

<!-- bilingual-en:start -->
The **Frobenius norm** measures a matrix’s overall size by treating all its entries as one long vector. Square the magnitude of each entry, add the results, then take the square root. Unlike a signed sum, positive and negative entries cannot cancel each other in this measure.
<!-- bilingual-en:end -->

**Frobenius 范数**把矩阵中的全部元素当成一个长向量，测量矩阵的整体大小：每个元素的模先平方，全部相加，再开平方。与带符号求和不同，这样计算时正负元素不会互相抵消。

> [!example] Four entries make one length · 四个元素组成一个长度
> <!-- bilingual-en:start -->
> For $A=\begin{bmatrix}1&2\\-2&0\end{bmatrix}$, list the entries as $(1,2,-2,0)$. Its [[欧氏范数|Euclidean length]] gives the Frobenius norm:
> <!-- bilingual-en:end -->
>
> 对 $A=\begin{bmatrix}1&2\\-2&0\end{bmatrix}$，把元素排成 $(1,2,-2,0)$，它的[[欧氏范数|欧氏长度]]就是 Frobenius 范数：
>
> $$\|A\|_F=\sqrt{1^2+2^2+(-2)^2+0^2}=3,\qquad \|A\|_F^2=9.$$

<!-- bilingual-en:start -->
The result is unchanged if those same entries occupy different positions, because addition does not depend on their order. More generally, a matrix with just three nonzero entries $1,-2,2$ also has Frobenius norm $3$.
<!-- bilingual-en:end -->

只要元素相同，改变它们的位置不会改变结果，因为求和不依赖排列顺序。更一般地，只有三个非零元素 $1,-2,2$ 的矩阵，Frobenius 范数也为 $3$。

<!-- bilingual-en:start -->
## Keep the square root in the definition
<!-- bilingual-en:end -->

*定义中保留最后的平方根*

<!-- bilingual-en:start -->
Let $A=(a_{jk})\in\mathbb F^{m\times n}$ with positive integers $m,n$ and $\mathbb F=\mathbb R$ or $\mathbb C$. Here $j$ labels a row, $k$ labels a column, and $a_{jk}$ is the entry at their intersection. The subscript $F$ names this particular matrix norm:
<!-- bilingual-en:end -->

设 $A=(a_{jk})\in\mathbb F^{m\times n}$，$m,n$ 为正整数，$\mathbb F=\mathbb R$ 或 $\mathbb C$。$j$ 是行号，$k$ 是列号，$a_{jk}$ 是该行该列的元素。下标 $F$ 表示这一种矩阵范数：

$$\|A\|_F=\left(\sum_{j=1}^{m}\sum_{k=1}^{n}|a_{jk}|^2\right)^{1/2}.$$

<!-- bilingual-en:start -->
For real entries, $|a_{jk}|^2=a_{jk}^2$. For a complex entry $z=a+bi$, its squared magnitude is $|z|^2=a^2+b^2=\overline z z$, where $\overline z=a-bi$ is its conjugate. For example, the one-entry matrix $[i]$ has Frobenius norm $1$; using $i^2=-1$ in place of $|i|^2$ would not produce a length.
<!-- bilingual-en:end -->

实元素满足 $|a_{jk}|^2=a_{jk}^2$。复元素 $z=a+bi$ 的模平方为 $|z|^2=a^2+b^2=\overline z z$，其中 $\overline z=a-bi$ 是其共轭。例如只有一个元素的矩阵 $[i]$ 的 Frobenius 范数为 $1$；若错用 $i^2=-1$ 代替 $|i|^2$，得到的就不是长度。

> [!important] A norm and its square are different · 范数与范数平方不同
> <!-- bilingual-en:start -->
> The sum of squared magnitudes is $\|A\|_F^2$. Its square root is $\|A\|_F$. If every entry is multiplied by $c$, the norm is multiplied by $|c|$, while its square is multiplied by $|c|^2$.
> <!-- bilingual-en:end -->
>
> 元素模平方之和是 $\|A\|_F^2$，再开平方才是 $\|A\|_F$。所有元素都乘以 $c$ 时，范数乘以 $|c|$，而范数平方乘以 $|c|^2$。

<!-- bilingual-en:start -->
As a norm on the [[矩阵空间|space of fixed-size matrices]], it is nonnegative, vanishes exactly when every entry is zero, and satisfies $\|A+B\|_F\le\|A\|_F+\|B\|_F$ for matrices of the same size. These are the ordinary vector norm properties applied to the lists of entries. They remain valid with complex magnitudes.
<!-- bilingual-en:end -->

作为[[矩阵空间|固定尺寸矩阵空间]]上的范数，它非负，且当且仅当所有元素都是零时取零；同尺寸矩阵满足 $\|A+B\|_F\le\|A\|_F+\|B\|_F$。把各元素列成向量后，这些就是向量范数的性质；复数情形使用模，仍然成立。

<!-- bilingual-en:start -->
## Measure the error in a whole matrix
<!-- bilingual-en:end -->

*测量整张矩阵的误差*

<!-- bilingual-en:start -->
To compare a matrix $A$ with an approximation $B$ of the same size, compute the entrywise difference $E=A-B$. Then $\|A-B\|_F$ is the square root of the sum of squared entry errors. If the only nonzero errors are $0.1$ and $-0.2$, this distance is $\sqrt{0.01+0.04}=\sqrt{0.05}$, whereas the sum of squared errors is $0.05$. This is an aggregate error, not an average per entry.
<!-- bilingual-en:end -->

要比较矩阵 $A$ 和同尺寸近似矩阵 $B$，先逐元素相减得到 $E=A-B$。$\|A-B\|_F$ 就是全部元素误差平方和的平方根。若只有两处非零误差 $0.1$、$-0.2$，该距离为 $\sqrt{0.01+0.04}=\sqrt{0.05}$，而误差平方和为 $0.05$。这里得到的是整体误差，不是每个元素的平均误差。

<!-- bilingual-en:start -->
The [[谱范数|spectral norm]] answers another question: how much can the matrix stretch a unit input vector? For the [[对角矩阵|diagonal matrix]] $\operatorname{diag}(4,3,0)$, that largest stretch is $4$, but the Frobenius norm is $\sqrt{16+9}=5$. A separate identity, [[Frobenius范数与奇异值|the Frobenius norm in terms of singular values]], explains why these entry totals can also be read from all the squared singular stretches.
<!-- bilingual-en:end -->

[[谱范数]]回答另一个问题：矩阵最多能把单位输入向量伸长到多少？对[[对角矩阵]] $\operatorname{diag}(4,3,0)$，最大伸缩是 $4$，但 Frobenius 范数为 $\sqrt{16+9}=5$。独立恒等式[[Frobenius范数与奇异值]]说明，为什么逐元素累计的同一个总量也能从全部奇异伸缩量的平方读出。

## 来源与核验

<!-- bilingual-en:start -->
- [Boyd and Vandenberghe, *Convex Optimization*, Appendix A.1](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), A.1.1–A.1.2, printed pp. 633–634 (PDF pp. 647–648): entry-list Euclidean norm, Frobenius definition, norm properties and norm distance; A.1.5 distinguishes the induced 2-norm.
<!-- bilingual-en:end -->

- Boyd 与 Vandenberghe A.1.1–A.1.2，印刷页 633–634（PDF 647–648）：核对元素列表的欧氏长度、Frobenius 定义、范数性质与范数距离；A.1.5 区分诱导二范数。

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 4th ed.](https://linear.axler.net/LADR4e.pdf), §6A, example 6.3, definition 6.7 and properties 6.9/6.17, printed pp. 184–190 (PDF pp. 198–204): complex coordinate inner product and norm properties used after listing the entries. All displayed numerical examples are computed directly.
<!-- bilingual-en:end -->

- Axler 第 6A 节例 6.3、定义 6.7 与性质 6.9、6.17，印刷页 184–190（PDF 198–204）：核对把元素列成向量后所用的复坐标内积与范数性质；所有数值例子直接计算。
