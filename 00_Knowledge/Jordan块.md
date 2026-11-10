---
aliases:
  - Jordan block
  - Jordan 块
student_os: knowledge-atom
atom_id: LA-EIG-031
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
requires:
  - "[[特征对]]"
related:
  - "[[Jordan链]]"
  - "[[Jordan链向量线性无关]]"
  - "[[Jordan标准形]]"
  - "[[Jordan块的幂]]"
  - "[[Jordan块的指数]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Jordan blocks
<!-- bilingual-en:end -->

*Jordan 块*

<!-- bilingual-en:start -->
A **Jordan block** is a small matrix pattern for one chain of coordinate directions. Every direction receives the same scalar multiplication, but each direction after the first also contributes to the preceding one. This small, repeated pattern lets us represent a linear map even when ordinary eigenvectors do not supply enough independent directions.
<!-- bilingual-en:end -->

**Jordan 块**是表示一条坐标方向链的小型矩阵模式。每个方向都接受相同的标量倍乘，但从第二个方向开始，还会向前一个方向贡献一份分量。这种重复的小结构，让我们在普通特征向量不足以提供全部独立方向时，仍能表示线性映射。

<!-- bilingual-en:start -->
## Read the action before the pattern
<!-- bilingual-en:end -->

*先看它怎样作用，再看一般格式*

<!-- bilingual-en:start -->
Consider this three-coordinate example. The [[矩阵乘法|matrix product]] tells us exactly what happens to a column vector:
<!-- bilingual-en:end -->

先看一个三坐标例子。[[矩阵乘法]]直接告诉我们，列向量经过作用后变成什么：

$$J_3(4)=\begin{pmatrix}4&1&0\\0&4&1\\0&0&4\end{pmatrix},\qquad
J_3(4)\begin{pmatrix}x_1\\x_2\\x_3\end{pmatrix}
=\begin{pmatrix}4x_1+x_2\\4x_2+x_3\\4x_3\end{pmatrix}.$$

<!-- bilingual-en:start -->
Here $x_1,x_2,x_3$ are the input coordinates. The diagonal entries multiply each by 4. The entry 1 immediately above each diagonal entry copies the next input coordinate into the preceding output coordinate. For example, $(0,0,1)^T$ becomes $(0,1,4)^T$. The superscript $T$ means [[矩阵转置|transpose]]; it writes a listed tuple as a column.
<!-- bilingual-en:end -->

这里 $x_1,x_2,x_3$ 是输入坐标。对角元把各坐标乘以4；紧挨主对角线上方的1，把下一个输入坐标加进前一个输出坐标。例如 $(0,0,1)^T$ 变成 $(0,1,4)^T$。上标 $T$ 表示[[矩阵转置|转置]]，把横写的元组写成列向量。

<!-- bilingual-en:start -->
Let $e_1=(1,0,0)^T$, $e_2=(0,1,0)^T$ and $e_3=(0,0,1)^T$ be the coordinate [[基|basis]] vectors. Reading the three columns gives
<!-- bilingual-en:end -->

令 $e_1=(1,0,0)^T$、$e_2=(0,1,0)^T$、$e_3=(0,0,1)^T$ 为坐标[[基|基向量]]。逐列读取就得到

$$J_3(4)e_1=4e_1,\qquad
J_3(4)e_2=4e_2+e_1,\qquad
J_3(4)e_3=4e_3+e_2.$$

<!-- bilingual-en:start -->
An [[特征对|eigenvector]] is a nonzero vector whose image is a scalar multiple of itself. Thus $e_1$ is an eigenvector with eigenvalue 4, but $e_2$ and $e_3$ are not: each image contains an extra direction. Those extra contributions are what the block records.
<!-- bilingual-en:end -->

[[特征对|特征向量]]是作用后仍为自身标量倍数的非零向量。因此 $e_1$ 是特征值4的特征向量，$e_2$、$e_3$ 却不是：它们的像都多出另一个方向。Jordan 块记录的正是这种额外贡献。

> [!important] One scalar and one chain length · 一个标量与一条链的长度
> <!-- bilingual-en:start -->
> For a positive integer $r$ and a scalar $\lambda$ in the chosen field, the Jordan block $J_r(\lambda)$ is the $r\times r$ matrix whose diagonal is $\lambda$, whose **superdiagonal** (the positions immediately above the diagonal) is 1, and whose other entries are zero:
> <!-- bilingual-en:end -->
>
> 给定正整数 $r$ 与所选数域中的标量 $\lambda$，Jordan 块 $J_r(\lambda)$ 是一个 $r\times r$ 方阵：主对角线全为 $\lambda$，**超对角线**（紧挨主对角线上方的位置）全为1，其余位置全为零：
>
> $$J_r(\lambda)=
> \begin{pmatrix}
> \lambda&1&0&\cdots&0\\
> 0&\lambda&1&\ddots&\vdots\\
> \vdots&\ddots&\ddots&\ddots&0\\
> 0&\cdots&0&\lambda&1\\
> 0&\cdots&\cdots&0&\lambda
> \end{pmatrix}=\lambda I_r+N.$$
>
> <!-- bilingual-en:start -->
> Here $I_r$ is the $r\times r$ [[单位矩阵|identity matrix]], and $N$ contains just the superdiagonal ones. The scalar may be real or complex; it is not restricted to positive numbers. The size-one convention is $J_1(\lambda)=[\lambda]$, with $N=[0]$.
> <!-- bilingual-en:end -->
>
> 这里 $I_r$ 是 $r\times r$ [[单位矩阵]]，$N$ 只保留超对角线上的1。标量可以是实数或复数，并不要求为正。尺寸为1时，约定 $J_1(\lambda)=[\lambda]$，此时 $N=[0]$。

<!-- bilingual-en:start -->
## Subtract the scalar part to see the chain
<!-- bilingual-en:end -->

*减去标量部分，链就显现出来*

<!-- bilingual-en:start -->
In the coordinate basis $e_1,\ldots,e_r$, subtracting $\lambda I_r$ leaves
<!-- bilingual-en:end -->

在坐标基 $e_1,\ldots,e_r$ 中，减去 $\lambda I_r$ 后，剩下

$$Ne_1=0,\qquad Ne_{j+1}=e_j\quad(1\le j<r).$$

<!-- bilingual-en:start -->
The index $j$ selects a position in the chain. Each application of $N$ moves a basis vector one position toward $e_1$, then sends it to zero. In the three-coordinate example, $e_3\mapsto e_2\mapsto e_1\mapsto0$. Consequently $N^r=0$, while $N^{r-1}e_r=e_1\ne0$. Such an operator is called **nilpotent**: some positive power is the zero operator. Its smallest such power here is $r$. For $r=1$, the preceding identity uses $N^0=I_1$.
<!-- bilingual-en:end -->

下标 $j$ 选择链中的位置。每作用一次 $N$，基向量就朝 $e_1$ 移动一个位置，最后归零。三坐标例子就是 $e_3\mapsto e_2\mapsto e_1\mapsto0$。因此 $N^r=0$，但 $N^{r-1}e_r=e_1\ne0$。这种算子称为**幂零**：某个正整数次幂是零算子；这里最小的这种次数正好为 $r$。当 $r=1$ 时，前一条等式使用 $N^0=I_1$。

<!-- bilingual-en:start -->
These are the defining equations of a [[Jordan链|Jordan chain]]. For a matrix $A$ acting in other coordinates, let $I$ denote the identity matrix of the same size as $A$. A chain $v_1,\ldots,v_r$ with $v_1\ne0$ satisfies $(A-\lambda I)v_1=0$ and $(A-\lambda I)v_{j+1}=v_j$. In that ordered chain basis, the action is exactly $J_r(\lambda)$. Later chain vectors are [[广义特征向量|generalized eigenvectors]]: repeated applications of $A-\lambda I$ eventually send them to zero, although one application need not. The [[Jordan链向量线性无关|chain-independence result]] explains why these vectors can serve as a basis for their span.
<!-- bilingual-en:end -->

这些就是 [[Jordan链]]的定义方程。对其他坐标中的矩阵 $A$，令 $I$ 表示与 $A$ 同尺寸的单位矩阵。一条链 $v_1,\ldots,v_r$ 要求 $v_1\ne0$，并满足 $(A-\lambda I)v_1=0$、$(A-\lambda I)v_{j+1}=v_j$；在这组有序链基中，作用恰好表示为 $J_r(\lambda)$。链中后续向量是[[广义特征向量]]：反复施加 $A-\lambda I$ 后归零，虽然一次未必归零。[[Jordan链向量线性无关]]解释了它们为什么可以作为自身张成空间的一组基。

<!-- bilingual-en:start -->
For the block itself, solving $Nx=0$ forces $x_2=\cdots=x_r=0$. Thus the ordinary eigenvectors for $\lambda$ are precisely the nonzero multiples of $e_1$: the eigenspace has **one independent direction**. Over the real or complex numbers, all nonzero scalar multiples remain valid eigenvectors. All $r$ coordinate directions belong to its generalized eigenspace because $N^r=0$. In particular, $J_2(\lambda)=\begin{pmatrix}\lambda&1\\0&\lambda\end{pmatrix}$ has one ordinary eigenvector direction but two generalized directions. Combining blocks into a whole-matrix representation is the job of [[Jordan标准形|Jordan canonical form]].
<!-- bilingual-en:end -->

对这个块本身，解 $Nx=0$ 会迫使 $x_2=\cdots=x_r=0$。因此特征值 $\lambda$ 的普通特征向量恰为 $e_1$ 的非零倍数：这个特征空间只有**一个独立方向**。在实数或复数范围内，所有非零标量倍数都仍是合法的特征向量。由于 $N^r=0$，全部 $r$ 个坐标方向都属于它的广义特征空间。特别地，$J_2(\lambda)=\begin{pmatrix}\lambda&1\\0&\lambda\end{pmatrix}$ 有一个普通特征方向，却有两个广义方向。把若干块组合成整个矩阵的表示，则由 [[Jordan标准形]]处理。

> [!warning] The shift part is nilpotent · 幂零的是移位部分
> <!-- bilingual-en:start -->
> The equation $N^r=0$ does not say $J_r(\lambda)^r=0$. When $\lambda\ne0$, for example, $J_r(\lambda)^r e_1=\lambda^r e_1\ne0$. Only at $\lambda=0$ is the whole block nilpotent. For $J_3(4)-4I_3$, the square still sends $e_3$ to $e_1$, but the cube sends every vector to zero. [[Jordan块的幂|Powers]] and [[Jordan块的指数|exponentials]] use this finite shift structure in their own formulas.
> <!-- bilingual-en:end -->
>
> $N^r=0$ 并不是说 $J_r(\lambda)^r=0$。例如当 $\lambda\ne0$ 时，$J_r(\lambda)^r e_1=\lambda^r e_1\ne0$。只有 $\lambda=0$ 时整个块才幂零。对 $J_3(4)-4I_3$，二次作用仍把 $e_3$ 送到 $e_1$，三次作用才把每个向量送到零。[[Jordan块的幂]]与[[Jordan块的指数]]在各自公式中利用这种有限移位结构。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 4th ed., §8C, examples 8.42–8.43 and definition 8.44, printed pp.321–322 / PDF pp.335–336](https://linear.axler.net/LADR4e.pdf#page=335): nilpotent chains, superdiagonal-one blocks, repeated eigenvalues and size-one blocks. The coordinate actions and power checks above are calculated directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=4|MIT 18.06SC Session 3.4 summary, p.4]]: the Jordan-block pattern and one independent eigenvector per block.
- [[03_Positive Definite Matrices and Applications#3.4.4 Jordan链与Jordan标准形|Permanent course notes, §3.4.4]]: the course convention $J_r(\lambda)=\lambda I+N$ and the ordered-chain equations.
<!-- bilingual-en:end -->

- Axler 第8C节例8.42–8.43与定义8.44，印刷321–322页／PDF335–336页：核对幂零链、超对角线为1的块、重复特征值与一阶块；上面的坐标作用及幂由显示矩阵直接核算。
- MIT Session3.4 摘要第4页：核对 Jordan 块的格式与每块一个独立特征方向。
- 永久课程第3.4.4节：保留课程中 $J_r(\lambda)=\lambda I+N$ 的约定与有序链方程。
