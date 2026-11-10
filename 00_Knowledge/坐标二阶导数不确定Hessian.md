---
student_os: knowledge-atom
atom_id: OPT-TOOLS-011
aliases:
  - Hessian reconstruction from directional second derivatives
  - Coordinate curvature and mixed partials
status: source-checked
---

<!-- bilingual-en:start -->
# Recovering a Hessian from directional curvature
<!-- bilingual-en:end -->

*从方向曲率恢复 Hessian*

<!-- bilingual-en:start -->
Looking only along the coordinate axes can miss how two inputs interact. For $f(x,y)=xy$, keeping $y=0$ makes the function identically zero, and so does keeping $x=0$. Neither axis reveals any second-order change at the origin. Yet moving both inputs together, $(x,y)=(t,t)$, gives $f(t,t)=t^2$. The missing information is a **mixed partial derivative**, stored off the diagonal of the Hessian.
<!-- bilingual-en:end -->

只沿坐标轴查看，可能漏掉两个输入怎样共同作用。对 $f(x,y)=xy$，固定 $y=0$ 时函数恒为零，固定 $x=0$ 时也一样。两条轴都没有显示原点处的二阶变化；但让两个输入同时变化，取 $(x,y)=(t,t)$，就得到 $f(t,t)=t^2$。缺失的信息是**混合偏导数**，它存放在 Hessian 的非对角位置。

<!-- bilingual-en:start -->
## What one direction measures
<!-- bilingual-en:end -->

*一个方向测到什么*

<!-- bilingual-en:start -->
Let $f:U\to\mathbb R$, where $U\subseteq\mathbb R^n$ is [[开集|open]], $n\ge1$, and $a\in U$. Assume $f$ is $C^2$ near $a$: its partial derivatives through order two exist and are continuous there. Its [[Hessian矩阵|Hessian]] $H=H_f(a)$ is then [[Hessian对称条件|symmetric]]. For a fixed real column vector $v$, the [[Hessian方向二阶导数|second derivative along the line]] $a+tv$ is the number
<!-- bilingual-en:end -->

设 $f:U\to\mathbb R$，$U\subseteq\mathbb R^n$ 为[[开集]]，$n\ge1$，$a\in U$。假设 $f$ 在 $a$ 附近为 $C^2$，即直到二阶的偏导在附近存在且连续。此时它的 [[Hessian矩阵|Hessian]] $H=H_f(a)$ [[Hessian对称条件|对称]]。对固定实列向量 $v$，沿直线 $a+tv$ 的[[Hessian方向二阶导数|二阶导数]]是一个数：

$$q(v)=\left.\frac{d^2}{dt^2}f(a+tv)\right|_{t=0}=v^THv.$$

<!-- bilingual-en:start -->
Here $T$ means [[矩阵转置|transpose]]. The vector sets the velocity of the input path, so it need not have length one. This is a second derivative with respect to $t$, not necessarily a geometric curvature normalized by distance. Scaling $v$ by $c$ scales $q(v)$ by $c^2$.
<!-- bilingual-en:end -->

这里 $T$ 表示[[矩阵转置|转置]]。向量指定输入路径的速度，不必为单位长度。这是关于 $t$ 的二阶导数，不一定是按路程归一化的几何曲率。把 $v$ 乘以 $c$，会让 $q(v)$ 乘以 $c^2$。

<!-- bilingual-en:start -->
Write $e_i$ for the vector with a $1$ in coordinate $i$ and zeros elsewhere. Multiplying out gives $q(e_i)=H_{ii}$. Thus the $n$ axis measurements reveal only the $n$ diagonal entries. For $xy$, the full calculation is
<!-- bilingual-en:end -->

记 $e_i$ 为第 $i$ 个坐标是 $1$、其余为零的向量。相乘得到 $q(e_i)=H_{ii}$，所以 $n$ 个坐标轴测量只给出 $n$ 个对角元。对 $xy$，完整计算为

$$H=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
q((u,v)^T)=2uv.$$

<!-- bilingual-en:start -->
In particular $q(e_1)=q(e_2)=0$, but $q((1,1)^T)=2$ and $q((1,-1)^T)=-2$. Direct substitution gives $t^2$ and $-t^2$, whose second derivatives are indeed $2$ and $-2$. The zero function has the same axis measurements as $xy$ and a different Hessian. For $n\ge2$, axis data therefore do not determine the matrix, even for polynomials. When $n=1$, there is no mixed entry: the single coordinate measurement already determines the $1\times1$ Hessian.
<!-- bilingual-en:end -->

特别地，$q(e_1)=q(e_2)=0$，但 $q((1,1)^T)=2$，$q((1,-1)^T)=-2$。直接代入两条直线，得到 $t^2$ 与 $-t^2$，二阶导数确实为 $2$ 与 $-2$。恒零函数与 $xy$ 的坐标轴测量完全一样，Hessian 却不同；因此，在 $n\ge2$ 时，即使对多项式，轴上的数据也不能确定整个矩阵。若 $n=1$，没有混合项，唯一的坐标测量就已经确定 $1\times1$ Hessian。

<!-- bilingual-en:start -->
## Add paired directions to recover the mixed entries
<!-- bilingual-en:end -->

*增加成对和方向，补回混合项*

<!-- bilingual-en:start -->
For $i\ne j$, expand the measurement along $e_i+e_j$:
<!-- bilingual-en:end -->

当 $i\ne j$ 时，展开沿 $e_i+e_j$ 的测量：

$$\begin{aligned}
q(e_i+e_j)
&=(e_i+e_j)^TH(e_i+e_j)\\
&=H_{ii}+H_{ij}+H_{ji}+H_{jj}\\
&=H_{ii}+2H_{ij}+H_{jj}.
\end{aligned}$$

<!-- bilingual-en:start -->
The last equality uses symmetry. Subtracting the two already known diagonal values isolates the interaction:
<!-- bilingual-en:end -->

最后一步用到了对称性。减去已经知道的两个对角值，便隔离出交叉作用：

> [!important] A reconstruction formula · 恢复公式
> $$H_{ij}=\frac{q(e_i+e_j)-q(e_i)-q(e_j)}2,\qquad i<j.$$
>
> <!-- bilingual-en:start -->
> Measure the $n$ coordinate vectors and the $n(n-1)/2$ pairwise sums. These $n(n+1)/2$ specifically chosen values recover every entry of a real symmetric Hessian: the diagonal directly, the upper triangle by this formula, and the lower triangle by symmetry.
> <!-- bilingual-en:end -->
>
> 测量 $n$ 个坐标向量以及 $n(n-1)/2$ 个成对和向量。这 $n(n+1)/2$ 个特定值能恢复实对称 Hessian 的每个元素：对角元直接得到，上三角用此公式，下三角由对称性补齐。

<!-- bilingual-en:start -->
For example, suppose in two dimensions the measurements are $q(e_1)=2$, $q(e_2)=4$, and $q(e_1+e_2)=12$. The mixed entry is $(12-2-4)/2=3$, hence
<!-- bilingual-en:end -->

例如，在二维中测得 $q(e_1)=2$、$q(e_2)=4$、$q(e_1+e_2)=12$。混合项为 $(12-2-4)/2=3$，所以

$$H=\begin{pmatrix}2&3\\3&4\end{pmatrix}.$$

<!-- bilingual-en:start -->
The function $x^2+3xy+2y^2$ realizes this Hessian. Along $(1,1)^T$ it becomes $6t^2$, whose second derivative is $12$, not $6$. If instead the measured direction was the unit vector $(e_i+e_j)/\sqrt2$, first multiply that measured value by $2$ to obtain $q(e_i+e_j)$.
<!-- bilingual-en:end -->

函数 $x^2+3xy+2y^2$ 就具有这个 Hessian。沿 $(1,1)^T$ 代入，得到 $6t^2$，二阶导数为 $12$，不是 $6$。若实际测的是单位向量 $(e_i+e_j)/\sqrt2$，则要先把该测量值乘以 $2$，才得到 $q(e_i+e_j)$。

<!-- bilingual-en:start -->
## Enough measurements must also be different information
<!-- bilingual-en:end -->

*数量够，还需要信息独立*

<!-- bilingual-en:start -->
An arbitrary list of the same size need not suffice. In two dimensions, measuring $e_1$, $2e_1$ and $e_2$ gives three numbers, but the second is always four times the first. No mixed entry is revealed. Each measurement is a linear equation in the unknown entries of $H$; the equations must determine those entries independently. The coordinate-plus-pairwise-sum construction above does so explicitly.

Symmetry also matters: for a nonsymmetric matrix $A$, $v^TAv$ equals $v^T(A+A^T)v/2$ for every real $v$. Such measurements recover only its [[二次型对称化|symmetric part]], since the skew-symmetric part contributes zero. Our $C^2$ assumption justifies treating the actual Hessian as symmetric.
<!-- bilingual-en:end -->

任意给出相同数量的测量，仍可能不够。二维中测 $e_1$、$2e_1$ 与 $e_2$，虽然有三个数，第二个却永远是第一个的四倍，完全没有提供混合项。每个测量都是关于 $H$ 未知元素的一条线性方程；这些方程必须能独立确定相应元素。上面的“坐标向量加成对和”构造已经直接证明这一点。

对称性也很关键：对非对称矩阵 $A$，每个实向量 $v$ 都满足 $v^TAv=v^T(A+A^T)v/2$。因此，这类测量只能恢复其[[二次型对称化|对称部分]]，反对称部分的贡献恒为零。本卡的 $C^2$ 假设保证实际 Hessian 对称。

<!-- bilingual-en:start -->
This differs from a first derivative: when $f$ is [[全微分|differentiable]], $Df(a)$ is linear, so its values on coordinate vectors determine its value on every vector. The second-order response $q(v)$ is quadratic in $v$ and includes mixed terms; treating it as a linear map of direction loses precisely those terms.
<!-- bilingual-en:end -->

这一点与一阶导数不同：$f$ [[全微分|可微]]时，$Df(a)$ 是线性的，在坐标向量上的值决定它在所有向量上的值。二阶响应 $q(v)$ 关于 $v$ 是二次的，包含混合项；若错误地把它当作方向的线性函数，恰好会漏掉这些项。

## 来源与核验

<!-- bilingual-en:start -->
### Sources and verification

- [MIT 18.S096, lecture 12, PDF pp.1–2](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf#page=1): the Hessian entries, symmetric second derivative and quadratic Taylor term. The measurement identities and numerical examples above are obtained by direct expansion, with the stated $C^2$ conditions.
- [[EC400 Slides Lecture 1.pdf#page=41|LSE EC400 SOFP Lecture 1, PDF p.41 / slide39]]: the second-order Taylor expansion and full Hessian, including mixed entries.
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#4.7 再用一次链式法则，得到方向二阶导数|Permanent course note, §4.7]]: both chain-rule steps, the paired-direction recovery identity and course context; the complete calculation is retained there.
<!-- bilingual-en:end -->

- MIT 18.S096 讲义12，PDF第1–2页：Hessian 元素、对称二阶导数及 Taylor 二次项；本卡在明确 $C^2$ 条件后，直接展开验证测量恒等式与数值例子。
- LSE EC400 SOFP 第1讲，PDF第41页／编号39页：二阶 Taylor 展开和含混合项的完整 Hessian。
- 永久课程笔记 §4.7：两次链式求导、成对方向恢复公式与课程语境，完整计算保留在原笔记中。
