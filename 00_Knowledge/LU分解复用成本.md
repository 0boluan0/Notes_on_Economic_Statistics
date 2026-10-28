---
student_os: knowledge-atom
atom_id: LA-LU-002
aliases:
  - Dense LU factorization costs cubic work and each additional right-hand side costs quadratic work
  - 稠密LU分解需立方量级工作而每个新增右端的求解需平方量级工作
  - Cost of reusing an LU factorization
status: source-checked
---

<!-- bilingual-en:start -->
# Dense LU factorization costs cubic work and each additional right-hand side costs quadratic work
<!-- bilingual-en:end -->
*稠密 LU 分解需立方量级工作，每个新增右端的求解需平方量级工作*

<!-- bilingual-en:start -->
For a fixed nonsingular dense $n\times n$ matrix $A$, classical [[消元与LU分解|LU factorization]] costs $O(n^3)$. Once $PA=LU$ is available, each new system $Ax=b$ is solved by $Ly=Pb$ and $Ux=y$, costing $O(n^2)$ for the two triangular solves. For $q$ right-hand sides, the total is $O(n^3+qn^2)$.
<!-- bilingual-en:end -->

对固定的非奇异稠密 $n\times n$ 矩阵 $A$，经典[[消元与LU分解|LU 分解]]需要 $O(n^3)$。一旦有 $PA=LU$，每个新系统 $Ax=b$ 都可通过 $Ly=Pb$、$Ux=y$ 求解，两次三角求解共需 $O(n^2)$；$q$ 个右端的总成本为 $O(n^3+qn^2)$。

<!-- bilingual-en:start -->
Elimination updates successively smaller two-dimensional blocks, giving a leading cubic count. The lecture models that growth with
<!-- bilingual-en:end -->

消元逐步更新越来越小的二维矩阵块，因此出现立方主项。课堂用下面的平方和说明其增长：

$$
\sum_{j=1}^{n}j^2=\frac{n(n+1)(2n+1)}6
=\frac13n^3+\frac12n^2+\frac16n
\sim\frac13n^3.
$$

<!-- bilingual-en:start -->
This sum is a cost model, not an exact count of every algorithmic operation. Counting each multiplication and subtraction separately gives the familiar leading factorization count $\frac23n^3+O(n^2)$; counting them as one update gives about $\frac13n^3$. Each triangular solve instead sums row lengths, $1+\cdots+n=O(n^2)$.
<!-- bilingual-en:end -->

此平方和是成本模型，并非算法全部操作的精确计数。把乘法与减法分别计数时，分解的常见主项为 $\frac23n^3+O(n^2)$；把二者合为一次更新时约为 $\frac13n^3$。三角求解累计的是行长度，$1+\cdots+n=O(n^2)$。

<!-- bilingual-en:start -->
The quadratic claim concerns a new right-hand side with the same factors. It does not cover arbitrary computations involving $A$, a changed coefficient matrix, or the cost of obtaining all $n$ columns of an inverse. Sparse or structured algorithms require their own cost model.
<!-- bilingual-en:end -->

平方量级结论针对复用同一分解求解新右端，并不涵盖任意关于 $A$ 的计算、系数矩阵发生变化的情况，或求出逆矩阵全部 $n$ 列的总成本。稀疏矩阵与结构化算法也需要另外分析。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=2|MIT lecture summary, p. 2]] · [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U1_S05_Lecture_Factorization_into_A_LU.pdf|MIT lecture transcript: operation counts]]
  <!-- bilingual-en:start -->
  Supports cubic elimination, quadratic cost per right-hand side, and the distinction between counting a multiply-subtract pair as one or two operations. The exact square-sum identity and total for $q$ right-hand sides are checked algebraically.
  <!-- bilingual-en:end -->
  核对立方消元成本、每个右端的平方成本，以及乘减合计一次或分别计数的区别；平方和恒等式与 $q$ 个右端总成本经代数核验。
