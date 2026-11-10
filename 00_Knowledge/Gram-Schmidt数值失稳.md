---
aliases:
  - "CGS loss of orthogonality"
  - "Gram–Schmidt loss of orthogonality"
  - "Gram-Schmidt 正交性损失"
student_os: knowledge-atom
atom_id: LA-PROJ-009
atom_set: orthogonal-projection-least-squares
atom_type: numerical-boundary
status: source-checked
requires:
  - "[[Gram-Schmidt正交化]]"
related:
  - "[[薄QR最小二乘]]"
  - "[[正规方程条件数平方]]"
  - "[[条件性与算法稳定性]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

<!-- bilingual-en:start -->
# Loss of orthogonality in Gram–Schmidt
<!-- bilingual-en:end -->

*Gram–Schmidt 数值失稳*

<!-- bilingual-en:start -->
An exact [[Gram-Schmidt正交化|Gram–Schmidt construction]] removes every old direction before adding a new one. A computer instead subtracts rounded quantities. If the genuinely new part is tiny, a small subtraction error can become large relative to that part; normalization can then turn it into a substantial unwanted direction. Thus a computed factorization can reconstruct the input closely while its supposedly perpendicular columns are far from perpendicular.
<!-- bilingual-en:end -->

精确的[[Gram-Schmidt正交化]]在加入新方向前，会消去所有旧方向；计算机减去的却是已经舍入的量。如果真正新增的部分很小，微小的减法误差相对于它就可能很大，归一化还会把误差变成明显的多余方向。因此，算出的分解可以很好地重建输入，而号称垂直的列却并不垂直。

<!-- bilingual-en:start -->
## Two different numerical questions
<!-- bilingual-en:end -->

*分解残差与正交性误差*

<!-- bilingual-en:start -->
For an $m\times n$ input $A$ with $m\ge n\ge1$ and independent columns, let $\widehat Q,\widehat R$ denote the factors actually computed; the hats distinguish them from exact ones. A useful pair of diagnostics is

$$\rho=\frac{\|A-\widehat Q\widehat R\|_F}{\|A\|_F},\qquad
\omega=\|\widehat Q^*\widehat Q-I_n\|_F.$$

The [[Frobenius范数|Frobenius norm]] is the square root of the sum of squared absolute entries. The first quantity measures reconstruction error; the second measures loss of [[标准正交组|orthonormality]]. Here $I_n$ is the [[单位矩阵|identity]] and $^*$ is conjugate transpose, equal to $^T$ for real matrices. In complex arithmetic conjugation replaces $a+bi$ by $a-bi$, with $i^2=-1$. A small $\rho$ alone does not bound $\omega$ or justify treating $\widehat Q^*$ as an inverse on its column space.
<!-- bilingual-en:end -->

对 $m\ge n\ge1$、各列独立的 $m\times n$ 输入 $A$，以 $\widehat Q,\widehat R$ 表示实际计算的因子，帽号用来区别精确因子。上述两个检查量采用[[Frobenius范数]]，即各元素绝对值平方和的平方根。$\rho$ 衡量重建误差，$\omega$ 衡量[[标准正交组|标准正交性]]损失。$I_n$ 是[[单位矩阵]]，$^*$ 为共轭转置，实矩阵时等于 $^T$；复数共轭把 $a+bi$ 变为 $a-bi$，其中 $i^2=-1$。单凭很小的 $\rho$，既不能控制 $\omega$，也不能把 $\widehat Q^*$ 当成列空间上的逆。

<!-- bilingual-en:start -->
## Why subtracting projections can fail numerically
<!-- bilingual-en:end -->

## 失稳从哪里来

<!-- bilingual-en:start -->
Classical Gram–Schmidt (CGS) computes all coefficients from the original current column $a_k$:

$$r_{ik}=q_i^*a_k,\qquad v=a_k-\sum_{i<k}q_i r_{ik}.$$

In exact arithmetic the earlier $q_i$ are orthonormal, so each old component cancels. In floating point, both these vectors and the coefficients contain error. When $a_k$ nearly lies in the earlier span, the true remainder is much smaller than the terms being subtracted. This is [[灾难性消去|cancellation]]: the resulting direction may be inaccurate even when each original operation has only small relative rounding error. Dividing by the remainder's small norm makes the unwanted components more visible.
<!-- bilingual-en:end -->

经典 Gram–Schmidt（CGS）从当前原列 $a_k$ 计算全部投影系数，再按上式相减。精确算术中，先前 $q_i$ 标准正交，旧分量会恰好抵消；浮点计算中，方向与系数都带有误差。当 $a_k$ 几乎落在已有张成空间内，真正余量远小于相减各项，就发生[[灾难性消去]]：即使每个原始运算的相对舍入误差都小，余量方向仍可能不准。再除以很小的余量范数，多余分量会更明显。

> [!example] Three almost parallel columns · 三个几乎平行的列
> <!-- bilingual-en:start -->
> Consider the real matrix
>
> $$A=\begin{bmatrix}1&1&1\\ \varepsilon&0&0\\0&\varepsilon&0\\0&0&\varepsilon\end{bmatrix},\qquad \varepsilon=10^{-8}.$$
>
> Its columns are exactly independent because the last three rows form $\varepsilon I_3$. In a straightforward binary64 implementation, $1+\varepsilon^2$ rounds to $1$, so the first computed direction is approximately $(1,\varepsilon,0,0)^T$. CGS then produces directions close to
>
> $$\widehat q_2=(0,-1,1,0)^T/\sqrt2,\qquad
> \widehat q_3=(0,-1,0,1)^T/\sqrt2.$$
>
> Their dot product is $1/2$, nowhere near zero, although the factors can still reconstruct $A$ to rounding accuracy. This example describes the stated simple arithmetic order; it is not a claim that every library or precision returns identical entries.
> <!-- bilingual-en:end -->
>
> 取上述 $\varepsilon=10^{-8}$ 的实矩阵。末三行是 $\varepsilon I_3$，所以各列精确线性无关。直接 binary64 实现中，$1+\varepsilon^2$ 舍入为 $1$，第一计算方向约为 $(1,\varepsilon,0,0)^T$；随后 CGS 得到接近显示的 $\widehat q_2,\widehat q_3$。两者点积为 $1/2$，显然不接近零，但因子仍可能在舍入精度内重建 $A$。这是指定简单运算顺序的例子，并不声称每个库、每种精度都会返回完全相同的元素。

<!-- bilingual-en:start -->
## Changing the order changes the rounding
<!-- bilingual-en:end -->

## 怎样选实现

<!-- bilingual-en:start -->
**Modified Gram–Schmidt (MGS)** starts with $v=a_k$ and processes one earlier direction at a time:

$$r_{ik}=q_i^*v,\qquad v\leftarrow v-q_i r_{ik}\quad(i=1,\ldots,k-1).$$

The arrow means replace $v$ by the new remainder; after the last subtraction, normalize it to obtain $q_k$. The key change is the input to the inner product: it is the **updated remainder**, not the original $a_k$. Exact orthogonality makes the CGS and MGS coefficients equal in exact arithmetic, since subtracting earlier perpendicular directions cannot change the component along $q_i$. Rounded directions no longer have that property.

In the example, after removing $q_1$ from the third column, the remainder is approximately $(0,-\varepsilon,0,\varepsilon)^T$. Its inner product with $\widehat q_2$ is $\varepsilon/\sqrt2$, so MGS removes an additional component that the original-column CGS coefficient misses. This makes $\widehat q_2$ and the new $\widehat q_3$ much closer to perpendicular, but small unwanted components along $\widehat q_1$ can remain.
<!-- bilingual-en:end -->

**修改 Gram–Schmidt（MGS）**从 $v=a_k$ 出发，逐个旧方向执行上述计算；箭头表示用新余量替换 $v$；最后一次减法后，再归一化得到 $q_k$。关键是内积使用**已经更新的余量**，而不是原列 $a_k$。在精确标准正交性下，减去先前垂直方向不会改变沿 $q_i$ 的分量，所以 CGS 与 MGS 系数相同；已舍入的方向不再满足这一性质。

例子中，从第三列减去 $q_1$ 分量后，余量约为 $(0,-\varepsilon,0,\varepsilon)^T$，与 $\widehat q_2$ 的内积为 $\varepsilon/\sqrt2$。MGS 因而继续消去一部分原列系数遗漏的分量，使第二、第三计算方向更接近垂直；沿第一计算方向仍可能留下小分量。

<!-- bilingual-en:start -->
MGS generally improves the orthogonality behavior, but severe [[设计矩阵条件数|conditioning]] can still make a single pass inadequate. **Reorthogonalization** repeats the subtraction against the previously accepted directions; to preserve the reconstruction, its correction coefficients must be added to the earlier coefficients in $R$. Near-machine-precision guarantees require numerical-independence and rounding assumptions, not merely a command to repeat twice. A zero or unresolved remainder needs a rank/tolerance decision.

For a general dense QR calculation, a well-implemented Householder method is a standard robust choice; a reflector is a length-preserving reflection used to eliminate entries. Such methods do not rely on repeatedly normalizing a cancellation-dominated residual. They still do not remove [[条件性与算法稳定性|sensitivity of the original problem]]. For an incremental basis, reorthogonalized Gram–Schmidt may fit the task. In either case, check both the factorization and orthogonality at the precision and scale the application needs.
<!-- bilingual-en:end -->

MGS 通常改善正交性表现，但严重的[[设计矩阵条件数|条件性]]仍可能使一遍处理不足。**再正交化**再次针对先前已接受的方向做减法；要保持重建式，修正投影系数须加到 $R$ 中原系数上。接近机器精度的保证需要数值独立性及舍入假设，并不是无条件“重复两次就好”；余量为零或无法分辨时，需要作秩及容差判断。

通用稠密 QR 中，可靠的 Householder 实现是常见的稳健选择；反射是用来消去元素的保长变换，不依赖反复归一化严重消去后的余量。但它仍不能消除[[条件性与算法稳定性|原问题的敏感性]]。逐步增加基向量的任务中，再正交化的 Gram–Schmidt 可能更适合。无论哪种方法，都应按应用的精度和尺度，同时检查分解与正交性。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.335J Week 4, Lectures 9–10](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-4/): small reconstruction error does not establish orthogonality; Householder QR versus classical Gram–Schmidt.
- Giraud, Langou and Rozložník (2005), [*The Loss of Orthogonality in the Gram–Schmidt Orthogonalization Process*](https://doi.org/10.1016/j.camwa.2005.08.009), [university-hosted full text](https://disciplinas.ime.unicamp.br/ms993/wp-content/uploads/sites/8/2023/08/2005-CMA-giraud-langou-rozloznik.pdf), §2, pp. 1070–1072 / PDF pp. 2–4: CGS/MGS update order, condition-dependent loss and assumptions for reorthogonalization. The four-row example is directly derived and checked in binary64 here.
<!-- bilingual-en:end -->

- MIT 第 9–10 讲核对重建误差与正交性误差的区别，以及 Householder 与经典 Gram–Schmidt 的实现边界。
- Giraud 等（2005）第 2 节，印刷第 1070–1072 页 / PDF 第 2–4 页，核对 CGS/MGS 更新顺序、条件性相关损失及再正交化的前提；四行例子由正文直接推导并以 binary64 检查。
