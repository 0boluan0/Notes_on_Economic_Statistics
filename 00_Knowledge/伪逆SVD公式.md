---
aliases:
  - "SVD formula for the Moore-Penrose pseudoinverse"
  - "Singular-value formula for the pseudoinverse"
student_os: knowledge-atom
atom_id: LA-PINV-016
atom_set: pseudoinverse-one-sided-inverses
atom_type: formula
status: source-checked
requires:
  - "[[Moore–Penrose 伪逆]]"
  - "[[紧SVD]]"
related:
  - "[[满列秩伪逆公式]]"
  - "[[满行秩伪逆公式]]"
  - "[[AA+列空间投影]]"
  - "[[A+A行空间投影]]"
leads_to:
  - "[[伪逆存在唯一性]]"
  - "[[小奇异值放大噪声]]"
  - "[[截断SVD反演]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
---

<!-- bilingual-en:start -->
# Pseudoinverse from the singular value decomposition
<!-- bilingual-en:end -->

*伪逆的 SVD 公式*

<!-- bilingual-en:start -->
The [[奇异值分解|singular value decomposition]] separates a matrix into independent input directions, their stretch factors, and corresponding output directions. To construct its [[Moore–Penrose 伪逆|Moore–Penrose pseudoinverse]], reverse each positive stretch and swap the roles of the input and output directions. A direction stretched to zero carries no recoverable information, so it is assigned zero instead of a reciprocal.
<!-- bilingual-en:end -->

[[奇异值分解]]把矩阵拆成一组独立输入方向、各自的伸缩量以及对应输出方向。构造 [[Moore–Penrose 伪逆]]时，撤销每个正的伸缩量，并交换输入与输出方向的角色。被压成零的方向没有可恢复信息，应该保持为零，不取倒数。

<!-- bilingual-en:start -->
## First reverse a visible stretch
<!-- bilingual-en:end -->

## 最小例子

$$
A=\begin{bmatrix}3&0\\0&0\end{bmatrix},\qquad
A^+=\begin{bmatrix}1/3&0\\0&0\end{bmatrix}.
$$

<!-- bilingual-en:start -->
The first coordinate is stretched by 3, so the reverse rule divides by 3. The second output is always zero. For target $b=(6,5)^T$, the rule gives $A^+b=(2,0)^T$, and applying $A$ again gives $(6,0)^T$. The unmatchable second output remains in the residual. An input's second coordinate cannot be inferred from this matrix; the minimum-norm choice sets it to zero. Here $T$ denotes [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

第一坐标被放大3倍，因此反向规则除以3；第二输出总为零。目标为 $b=(6,5)^T$ 时，规则给出 $A^+b=(2,0)^T$，再用 $A$ 得到 $(6,0)^T$；无法匹配的第二输出留在残差中。原输入的第二坐标无法从矩阵输出推知，最小范数选择把它设为零。这里 $T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
For a real or complex $m\times n$ matrix $A$ of [[矩阵秩|rank]] $r>0$, its [[紧SVD|compact SVD]] is
<!-- bilingual-en:end -->

对秩为 $r>0$ 的实或复 $m\times n$ 矩阵 $A$，[[紧SVD]]写成

$$
A=U_r\Sigma_rV_r^*,\qquad
U_r\in\mathbb F^{m\times r},\quad
V_r\in\mathbb F^{n\times r},\quad
\Sigma_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r),\quad
\sigma_i>0.
$$

<!-- bilingual-en:start -->
Here $\mathbb F$ is either the real or complex numbers. The columns $u_i$ of $U_r$ and $v_i$ of $V_r$ are [[标准正交组|orthonormal families]]: each has unit length, and distinct members are perpendicular. The star means conjugate transpose, replacing each $a+bi$ by $a-bi$ after transposition, with $i^2=-1$. Thus $U_r^*U_r=V_r^*V_r=I_r$, where $I_r$ is the [[单位矩阵|identity]]. The relation $Av_i=\sigma_i u_i$ says exactly which input direction reaches which output direction.
<!-- bilingual-en:end -->

$\mathbb F$ 表示实数域或复数域。$U_r$ 的列 $u_i$、$V_r$ 的列 $v_i$ 各自组成[[标准正交组]]：每个向量长度为1，不同向量彼此垂直。星号表示共轭转置，即转置后把每个 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$。因此 $U_r^*U_r=V_r^*V_r=I_r$，$I_r$ 为[[单位矩阵]]。关系 $Av_i=\sigma_i u_i$ 指明哪个输入方向到达哪个输出方向。

> [!important] Reverse the positive diagonal block / 撤销正对角块
> <!-- bilingual-en:start -->
> The pseudoinverse is
>
> $$A^+=V_r\Sigma_r^{-1}U_r^*
> =\sum_{i=1}^r\frac{1}{\sigma_i}v_i u_i^*.$$
>
> Its shape is $n\times m$. Only positive singular values occur in the inverse diagonal block. If $A=0$, use $A^+=0$ of the reversed shape; the sum is then empty.
> <!-- bilingual-en:end -->
>
> 伪逆是上述 $n\times m$ 矩阵，正对角块中的 $\sigma_i$ 分别换成 $1/\sigma_i$。若 $A=0$，直接取反向尺寸的零矩阵，求和为空，不对零作除法。

<!-- bilingual-en:start -->
## Read, rescale, rebuild
<!-- bilingual-en:end -->

## 方向上的作用

<!-- bilingual-en:start -->
For an output target $b$ with $m$ coordinates, $u_i^*b$ reads its component along output direction $u_i$. Divide that number by the stretch $\sigma_i$, then use it as the coefficient of input direction $v_i$:
<!-- bilingual-en:end -->

目标 $b$ 含 $m$ 个输出坐标。$u_i^*b$ 读取它沿输出方向 $u_i$ 的分量；除以伸缩量 $\sigma_i$，再把得到的数作为输入方向 $v_i$ 的系数：

$$
A^+b=\sum_{i=1}^r\frac{u_i^*b}{\sigma_i}v_i.
$$

<!-- bilingual-en:start -->
A component perpendicular to all the $u_i$ is outside the [[列空间|attainable output space]] and contributes zero. The result lies in the span of the $v_i$, namely $C(A^*)=N(A)^\perp$, so it contains none of the invisible [[零空间|nullspace]] input component. These statements explain the [[最小范数最小二乘解|minimum-norm least-squares choice]].
<!-- bilingual-en:end -->

垂直于所有 $u_i$ 的分量位于[[列空间|可达输出空间]]之外，贡献为零。结果落在 $v_i$ 张成的 $C(A^*)=N(A)^\perp$ 中，不包含看不见的[[零空间]]输入分量。这解释了[[最小范数最小二乘解|最小范数的最小二乘选择]]。

<!-- bilingual-en:start -->
The formula also verifies the definition. With $G=V_r\Sigma_r^{-1}U_r^*$, direct multiplication gives
<!-- bilingual-en:end -->

这个公式还能直接核对定义。令 $G=V_r\Sigma_r^{-1}U_r^*$，矩阵乘法给出

$$
AG=U_rU_r^*,\qquad GA=V_rV_r^*,\qquad
AGA=A,\qquad GAG=G.
$$

<!-- bilingual-en:start -->
The first two products equal their conjugate transposes; therefore $G$ satisfies all four Penrose conditions. They are respectively the [[AA+列空间投影|output-space]] and [[A+A行空间投影|input-space]] orthogonal projectors. Different valid bases inside a repeated-singular-value subspace give the same $A^+$, because the four-condition object is [[伪逆存在唯一性|unique]]. Pseudoinversion does not mean taking the reciprocal of each matrix entry.
<!-- bilingual-en:end -->

前两个乘积都等于各自的共轭转置，所以 $G$ 满足四个 Penrose 条件；它们分别是[[AA+列空间投影|输出空间]]与[[A+A行空间投影|输入空间]]中的正交投影。即使重复奇异值对应的子空间内选择不同合法基，所得 $A^+$ 也相同，因为四条件对象[[伪逆存在唯一性|唯一]]。伪逆并不是把原矩阵的每个元素逐项取倒数。

> [!example] A direction that is not a coordinate axis / 不沿坐标轴的方向
> <!-- bilingual-en:start -->
> For the row matrix $A=[1\ 1]$, take $\sigma_1=\sqrt2$, $u_1=1$ and $v_1=(1,1)^T/\sqrt2$. The formula yields $A^+=(1/2,1/2)^T$. A target sum of 2 is therefore split as $(1,1)^T$. The factor $1/2$ comes from both the reciprocal stretch and the unit direction; it is not the reciprocal of the entries of $A$.
> <!-- bilingual-en:end -->
>
> 对行矩阵 $A=[1\ 1]$，可取 $\sigma_1=\sqrt2$、$u_1=1$、$v_1=(1,1)^T/\sqrt2$。公式给出 $A^+=(1/2,1/2)^T$，因此把目标和2分成 $(1,1)^T$。系数 $1/2$ 同时来自伸缩量的倒数和单位方向，并非原元素取倒数。

<!-- bilingual-en:start -->
## Exact inversion and a rank decision
<!-- bilingual-en:end -->

## 精确公式与数值阈值的边界

<!-- bilingual-en:start -->
Every positive singular value is inverted in the exact formula. A tiny positive value can produce [[小奇异值放大噪声|large noise amplification]]. If a computation sets values at or below a chosen tolerance $\tau$ to zero, it uses the pseudoinverse of a thresholded matrix $A_\tau$, not the exact pseudoinverse of the original $A$. The threshold, scale convention and purpose must be stated; deliberate noise suppression is [[截断SVD反演|truncated SVD inversion]].
<!-- bilingual-en:end -->

精确公式会对每个正奇异值取倒数，很小的正值可能造成[[小奇异值放大噪声|很大的噪声放大]]。若计算把不超过给定容差 $\tau$ 的值设为零，使用的是截断矩阵 $A_\tau$ 的伪逆，不是原 $A$ 的精确伪逆。应说明阈值、尺度口径与目的；为抑制噪声主动截取方向，就是[[截断SVD反演]]。

<!-- bilingual-en:start -->
For example, $A=\operatorname{diag}(1,\varepsilon)$ with $\varepsilon>0$ has $A^+=\operatorname{diag}(1,1/\varepsilon)$. Dropping its second direction instead gives $\operatorname{diag}(1,0)$, which sends target $(0,1)^T$ to zero rather than $(0,1/\varepsilon)^T$. When only $A^+b$ is needed, its direction-by-direction sum can be evaluated without assembling the entire matrix $A^+$.
<!-- bilingual-en:end -->

例如 $\varepsilon>0$ 时，$A=\operatorname{diag}(1,\varepsilon)$ 的精确伪逆为 $\operatorname{diag}(1,1/\varepsilon)$；删去第二方向得到的却是 $\operatorname{diag}(1,0)$。后者把目标 $(0,1)^T$ 送到零，而非 $(0,1/\varepsilon)^T$。如果只需要 $A^+b$，可直接计算逐方向求和，不必先组装整个 $A^+$。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 7.75 and formula 7.78, PDF p.289](https://linear.axler.net/LADR4e.pdf#page=289): the positive-singular-value reciprocal expansion and its geometric proof.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8, PDF p.2]] and [LAPACK, SVD](https://netlib.org/lapack/lug/node32.html): rectangular factor dimensions and real/complex direction pairing. [LAPACK least-squares error bounds](https://netlib.org/lapack/lug/node83.html) explicitly separates a thresholded rank model.
- [Penrose (1955), Theorem 1, pp.406–407](https://doi.org/10.1017/S0305004100030401): the four-condition object and uniqueness. The verification and small examples above are direct multiplication.
<!-- bilingual-en:end -->

- Axler 第7.75条及公式7.78核对正奇异值倒数展开及其几何证明。
- MIT Session 3.8 第2页与 LAPACK SVD 章节核对长方因子尺寸、实复奇异方向配对；LAPACK 误差界章节另行声明阈值秩模型。
- Penrose 定理1核对四条件对象及唯一性；本文的四条件验证和小例子由直接乘法给出。
