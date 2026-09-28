---
aliases:
  - EC400 SOFP Lecture 1
  - SOFP第一讲正式笔记
  - Tools for Optimisation
---

# SOFP Lecture 1：二次型、Taylor 展开与凹凸性

<!-- bilingual-en:start -->
*Tools for Optimisation — Quadratic Forms, Taylor Approximation, and Concavity*
<!-- bilingual-en:end -->

这节课要解决一个问题：面对几个变量一起变化的函数，怎样判断一个点附近的变化，以及一个局部结论能否推广到整个定义域？课程先准备二次型和行列式，再用 Taylor 展开把一般函数的局部变化写成二次型，最后用凹性和拟凹性描述整个函数的形状。

<!-- bilingual-en:start -->
The central question is how to study a function when several inputs change together, and when a local conclusion extends to the whole domain. Quadratic forms and determinants come first; Taylor expansion then connects them to local changes in a general function. Concavity and quasiconcavity describe its global shape.
<!-- bilingual-en:end -->

正文沿第一讲 slides 的顺序。偏导数、方向导数和链式法则放在多元 Taylor 之前补齐；课堂已讲到的局部极值判据放在 Taylor 之后，作为通向第二讲的桥。每一段先把本课的意思说完整，再连接到可跨课程复用的知识原子。

<!-- bilingual-en:start -->
The exposition follows the first lecture. Partial derivatives, directional derivatives, and the chain rule are developed before multivariable Taylor expansion. The local-extremum tests discussed in class follow Taylor as a bridge to Lecture 2. Each passage explains the course material in place and links to the corresponding reusable knowledge atom.
<!-- bilingual-en:end -->

**课程原件：** [[EC400 Slides Lecture 1.pdf|Lecture 1 slides]] · [[EC400 Lecture Notes SOFP.pdf|SOFP Lecture Notes]]

**原始记录：** [[SOFP Lecture 1 - Tools for Optimisation - 手写笔记 - 2026-09-18.pdf|8 页手写笔记]] · [[SOFP Lecture 1 - Tools for Optimisation - Claude课堂记录 - 2026-09-18|课堂记录]]

**阅读路线：** [[#1. 二次型：把所有方向上的二次变化写在一起]] → [[#2. 行列式：从有向面积到可计算的数]] → [[#3. 主子式与定号判据]] → [[#4. Taylor 展开之前：把多元微分的符号接起来]] → [[#5. Taylor 展开：把局部信息变成近似]] → [[#6. 从 Taylor 到局部极值]] → [[#7. 凸集与凹凸函数]] → [[#8. 拟凹性：关注达到同一水平的那些点]]

**主题地图：** [[对称矩阵与正定二次型.canvas]] · [[多元微分.canvas]] · [[凹凸性与拟凹性.canvas]]

**页码约定。** 下文说“slide”时指页脚编号。PDF 第 18–20 页是同一张 slide 18 的动画帧；从 slide 19 起，PDF 物理页码 = slide 编号 + 2。讲义的印刷页码与 PDF 页码在本次范围内一致。

<!-- bilingual-en:start -->
Slide numbers refer to the printed footer. PDF pages 18–20 are three animation frames of slide 18; from slide 19 onward, the physical PDF page is the slide number plus two. The lecture-note page numbers agree with physical PDF pages in the material used here.
<!-- bilingual-en:end -->

## 1. 二次型：把所有方向上的二次变化写在一起

<!-- bilingual-en:start -->
*Quadratic forms collect second-order changes across directions*
<!-- bilingual-en:end -->

课程定位：[[EC400 Slides Lecture 1.pdf#page=8|slides 8–14]]。

### 1.1 先认清对象和符号

<!-- bilingual-en:start -->
*Objects and notation*
<!-- bilingual-en:end -->

[[二次型|二次型（quadratic form）]]是只含二次齐次项的函数。这里“齐次”表示每一项的总次数相同，都是 $2$：$x_1^2$ 的次数是 $2$，$x_1x_2$ 的次数是 $1+1=2$。一次项 $3x_1$ 和常数 $5$ 都不属于二次型。

设 $x=(x_1,\ldots,x_n)^T\in\mathbb R^n$ 是列向量，$A=(A_{ij})\in\mathbb R^{n\times n}$ 是实方阵。上标 $T$ 表示转置，把列变成行。写成
$$
Q(x)=x^TAx
$$
时，维度是 $(1\times n)(n\times n)(n\times1)=1\times1$，所以结果是一个数。矩阵装着系数，$x$ 装着输入；不是对矩阵本身逐个元素求平方。

<!-- bilingual-en:start -->
A [[二次型|quadratic form]] contains only homogeneous terms of degree two: squares and products of two coordinates, with no linear or constant terms. Here $x$ is a real column vector, $A$ a real square matrix, and $T$ denotes transpose. The displayed dimensions show that $x^TAx$ is a scalar. The matrix stores coefficients and the vector supplies the inputs.
<!-- bilingual-en:end -->

课程也写 $Q(x)=\sum_{i\le j}a_{ij}x_ix_j$。符号 $\sum$ 表示把指定下标下的项加起来；$i\le j$ 让每个交叉项只出现一次。**这里多项式的系数 $a_{ij}$，不应直接当成对称矩阵的非对角元 $A_{ij}$。** 对称矩阵的两个位置会各贡献一次交叉项，因而要各放一半。

例如
$$
Q(x_1,x_2)=3x_1^2+4x_1x_2-x_2^2,
\qquad
A=\begin{pmatrix}3&2\\2&-1\end{pmatrix}.
$$
先算 $Ax$，再左乘 $x^T$：
$$
Ax=\begin{pmatrix}3x_1+2x_2\\2x_1-x_2\end{pmatrix},
$$
$$
x^TAx=x_1(3x_1+2x_2)+x_2(2x_1-x_2)
=3x_1^2+2x_1x_2+2x_2x_1-x_2^2
=3x_1^2+4x_1x_2-x_2^2.
$$
反过来，$\alpha x_1^2+\beta x_1x_2+\gamma x_2^2$ 对应的对称矩阵是
$$
\begin{pmatrix}\alpha&\beta/2\\\beta/2&\gamma\end{pmatrix}.
$$

<!-- bilingual-en:start -->
The lecture's sum over $i\le j$ lists each cross term once. Its polynomial coefficient must therefore be divided between two symmetric matrix entries. The worked multiplication shows why a cross coefficient of $4$ requires two off-diagonal entries equal to $2$. Conversely, the coefficient of $x_1x_2$ is the sum of those two entries.
<!-- bilingual-en:end -->

### 1.2 为什么可以只研究对称矩阵

<!-- bilingual-en:start -->
*Why the symmetric representative is enough*
<!-- bilingual-en:end -->

[[实对称矩阵|对称矩阵]]满足 $A^T=A$，即 $A_{ij}=A_{ji}$。一个二次型如果允许用任意矩阵表示，表示并不唯一。例如上面的交叉项 $4x_1x_2$ 也可以全放在右上角；但要求对称之后，代表矩阵唯一。

原因是[[二次型对称化|对称化恒等式]]。对任意实方阵 $B$，$x^TBx$ 是标量，转置后不变，所以
$$
x^TBx=(x^TBx)^T=x^TB^Tx.
$$
把两种写法取平均：
$$
x^TBx=\frac12(x^TBx+x^TB^Tx)
=x^T\frac{B+B^T}{2}x.
$$
因此只需保留 $B$ 的对称部分。唯一性也能直接检查：若对称矩阵 $S,T$ 对所有 $x$ 给出相同二次型，取标准基 $x=e_i$ 得到相同对角元；再取 $x=e_i+e_j$，扣掉两个平方项后就得到相同非对角元。这里 $e_i$ 是第 $i$ 个位置为 $1$、其余为 $0$ 的向量。

<!-- bilingual-en:start -->
A [[实对称矩阵|symmetric matrix]] equals its transpose. Arbitrary representing matrices are not unique, but the symmetric representative is. Since a scalar equals its transpose, averaging $x^TBx$ with $x^TB^Tx$ gives the [[二次型对称化|symmetric-part identity]]. Uniqueness follows by testing standard basis vectors to recover the diagonal entries and then sums of two basis vectors to recover every off-diagonal entry.
<!-- bilingual-en:end -->

二次型还有一个很实用的尺度性质：$Q(tx)=t^2Q(x)$。例如把向量翻到反方向，$Q(-x)=Q(x)$；把长度加倍，二次型值变成四倍。它总满足 $Q(0)=0$。稍后会算出其一阶导数在原点为零，但不能仅凭“函数值为零”推断导数为零。

<!-- bilingual-en:start -->
Homogeneity gives $Q(tx)=t^2Q(x)$. Reversing direction leaves the value unchanged, and doubling the vector multiplies the value by four. Every quadratic form vanishes at the origin. Its derivative also vanishes there, as a later calculation will show; a zero function value alone would not establish that fact.
<!-- bilingual-en:end -->

### 1.3 正定、负定、半定和不定究竟在比较什么

<!-- bilingual-en:start -->
*Definiteness concerns the sign in every nonzero direction*
<!-- bilingual-en:end -->

要判断原点附近往哪里走会升、会降，就看 $Q(x)$ 在**所有非零向量**上是什么符号。$x\ne0$ 表示至少一个分量非零，并不要求每个分量都非零。比较的是标量 $x^TAx$，不是要求矩阵每个元素同号。

| 名称 | 精确定义 | 二维例子 |
|---|---|---|
| [[正定矩阵\|正定（PD）]] | 所有 $x\ne0$ 都有 $Q(x)>0$ | $x_1^2+x_2^2$ |
| [[负定矩阵\|负定（ND）]] | 所有 $x\ne0$ 都有 $Q(x)<0$ | $-(x_1^2+x_2^2)$ |
| [[半正定矩阵\|半正定（PSD）]] | 所有 $x$ 都有 $Q(x)\ge0$ | $(x_1+x_2)^2$ |
| [[半负定矩阵\|半负定（NSD）]] | 所有 $x$ 都有 $Q(x)\le0$ | $-(x_1+x_2)^2$ |
| [[不定二次型\|不定（indefinite）]] | 存在 $u,v$ 使 $Q(u)>0$、$Q(v)<0$ | $x_1^2-x_2^2$ |

$(x_1+x_2)^2$ 沿 $(1,-1)$ 取零，因此半正定但非正定。$x_1^2-x_2^2$ 在 $(1,0)$ 取 $1$，在 $(0,1)$ 取 $-1$；只要找到这两个方向，就已经证明不定。

<!-- bilingual-en:start -->
The sign test ranges over every nonzero vector; some coordinates may still be zero. [[正定矩阵|Positive definiteness]] means strictly positive values, [[负定矩阵|negative definiteness]] strictly negative values, and [[半正定矩阵|positive]] or [[半负定矩阵|negative semidefiniteness]] allows zero. [[不定二次型|Indefiniteness]] requires actual directions of both signs. The square $(x_1+x_2)^2$ has a nonzero zero direction, whereas $x_1^2-x_2^2$ takes opposite signs on the two coordinate axes.
<!-- bilingual-en:end -->

**本文统一采用宽义半定：正定属于半正定，负定属于半负定。** 因此上表的五个名称不是互不重叠的五类。零矩阵同时半正定、半负定；需要表达平坦方向时，写“半正定但非正定”或“半负定但非负定”。尤其不能看到 $\ge0$ 就认定一定存在非零零方向。

这些定义立即给出二次型的全局结论：PD 时原点是唯一全局最小点，ND 时是唯一全局最大点；PSD 但非 PD 时，原点是全局最小点，但还有非零零方向上的最小点；不定时原点既不是最大点也不是最小点。对于一般非二次函数，这些结论不能直接照搬，必须通过 Taylor 和余项来连接。

<!-- bilingual-en:start -->
Semidefinite is used inclusively: positive definite implies positive semidefinite, and negative definite implies negative semidefinite. The zero matrix belongs to both semidefinite classes. A flat direction requires semidefiniteness without definiteness. For a pure quadratic form the sign determines global behaviour at zero; transferring this reasoning to a general nonlinear function requires Taylor expansion and control of the remainder.
<!-- bilingual-en:end -->

### 1.4 等高线把符号变成图形

<!-- bilingual-en:start -->
*Level sets make the sign visible*
<!-- bilingual-en:end -->

![[SOFP-L1-quadratic-levels.png|850]]

[[水平集|水平集（level set）]]把满足 $Q(x)=c$ 的输入点画在同一张平面上。上图分别是正定、不定、半正定但非正定的二次型：正定时 $c>0$ 得到椭圆，$c=0$ 只有原点，$c<0$ 为空；不定例 $x_1^2-x_2^2$ 在非零水平上是双曲线，在零水平上是两条相交直线；$(x_1+x_2)^2$ 在正水平上是两条平行直线，在零水平上退化成一条直线。

图中的线表示“高度相同”，不是三维函数图像本身。正定二次型的椭圆如何由特征向量和特征值决定，见[[正定二次型的椭球]]。零二次型还要单独处理：零水平集是整个平面，其他水平集为空。

<!-- bilingual-en:start -->
A [[水平集|level set]] contains the inputs giving a specified output. Positive levels of a positive-definite form are ellipses, with a single point at level zero. An indefinite form has hyperbolas at nonzero levels and intersecting lines at zero. A rank-one positive-semidefinite form has parallel lines at positive levels and one line at zero. These are sets in input space. [[正定二次型的椭球|The ellipsoid atom]] connects the positive-definite case to eigenvectors and eigenvalues. The identically zero form has the whole plane as its zero level set.
<!-- bilingual-en:end -->

## 2. 行列式：从有向面积到可计算的数

<!-- bilingual-en:start -->
*Determinants: oriented volume and calculation*
<!-- bilingual-en:end -->

课程定位：[[EC400 Slides Lecture 1.pdf#page=16|slides 16–21]]。

### 2.1 三条性质是在定义一个什么对象

<!-- bilingual-en:start -->
*What the three defining properties mean*
<!-- bilingual-en:end -->

[[行列式|行列式（determinant）]]给每个方阵一个标量，记作 $\det A$ 或 $|A|$。这里竖线放在矩阵两侧时表示行列式，不是逐项取绝对值。把矩阵写成列向量的排列 $A=(A_1,\ldots,A_n)$，实数域上的三条性质是：

- 单位矩阵 $I_n$ 满足 $\det I_n=1$。
- 固定其他列时，对某一列线性，例如 $\det(\alpha A_1+B_1,A_2,\ldots)=\alpha\det(A_1,A_2,\ldots)+\det(B_1,A_2,\ldots)$。
- 交换两列，行列式变号。

“唯一”说的是这三条性质唯一确定了行列式这个函数；不是说一个数就能唯一识别矩阵。许多不同矩阵的行列式相同。

<!-- bilingual-en:start -->
The [[行列式|determinant]] assigns a scalar to a square matrix. Vertical bars around a matrix mean its determinant here. Over the reals it is characterized by normalization at the identity, linearity in one column while the others are fixed, and a sign change under a column swap. Uniqueness concerns the determinant function, not recovery of a matrix from its determinant.
<!-- bilingual-en:end -->

逐列线性不等于对整个矩阵线性。一般 $\det(A+B)\ne\det A+\det B$；例如二阶 $A=B=I_2$，左边 $\det(2I_2)=4$，右边 $1+1=2$。整张 $n\times n$ 矩阵乘 $\alpha$，相当于每一列都乘一次，因此
$$
\det(\alpha A)=\alpha^n\det A.
$$
只缩放一列时才只出现一个 $\alpha$。这个区别由[[整体缩放的行列式]]保存；[[行列式转置不变性]]说明逐行版本与逐列版本相通。

<!-- bilingual-en:start -->
Separate column linearity is not linearity in the whole matrix. For two identity matrices of size two, the determinant of their sum is four while the sum of their determinants is two. [[整体缩放的行列式|Scaling the entire matrix]] scales all $n$ columns, producing $\alpha^n$. [[行列式转置不变性|Transpose invariance]] connects the row and column formulations.
<!-- bilingual-en:end -->

### 2.2 把手写的面积图完整算出来

<!-- bilingual-en:start -->
*Reconstructing the handwritten area argument*
<!-- bilingual-en:end -->

![[SOFP-L1-determinant-area.png|850]]

取两列 $u=(a,b)^T$、$v=(c,d)^T$，令
$$
A=\begin{pmatrix}a&c\\b&d\end{pmatrix}.
$$
它们张成图中蓝色的平行四边形。为了画出这张特定分割图，先假设 $a,b,c,d>0$ 且 $ad>bc$；图示取 $a=d=3,b=c=1$。外接长方形宽 $a+c$、高 $b+d$。外面六块分别是两块面积 $ab/2$ 的三角形、两块面积 $cd/2$ 的三角形、两块面积 $bc$ 的长方形。因此
$$
\begin{aligned}
\text{平行四边形面积}
&=(a+c)(b+d)-2\frac{ab}{2}-2\frac{cd}{2}-2bc\\
&=(ab+ad+bc+cd)-ab-cd-2bc\\
&=ad-bc.
\end{aligned}
$$
这里是对手写第 5 页面积图的重新绘制与展开；一般情形的普通面积是 $|ad-bc|$。

<!-- bilingual-en:start -->
The blue parallelogram is spanned by the columns $u=(a,b)^T$ and $v=(c,d)^T$. For the pictured arrangement, all four entries are positive and $ad>bc$. Subtracting two triangles of area $ab/2$, two of area $cd/2$, and two rectangles of area $bc$ from the bounding rectangle gives $ad-bc$. This reconstructs the diagram on handwritten page 5. Ordinary area in the general case is the absolute value.
<!-- bilingual-en:end -->

[[行列式体积与取向|行列式还保留取向信息]]：交换列以后，平行四边形的大小没有变，但列的顺序反过来，行列式就变号。$\det A=0$ 表示列向量张成的图形被压扁，面积或体积为零；对方阵，它也等价于不可逆，见[[可逆性与非零行列式]]。所以负行列式不是“负的普通面积”，而是带方向约定的体积。

<!-- bilingual-en:start -->
The [[行列式体积与取向|oriented-volume interpretation]] retains the order of the spanning vectors. Swapping columns reverses the sign without changing ordinary volume. A zero determinant means volume collapses; for a square matrix this is exactly [[可逆性与非零行列式|failure of invertibility]]. A negative determinant therefore records orientation, not negative ordinary area.
<!-- bilingual-en:end -->

### 2.3 二阶公式、余子式与三阶展开

<!-- bilingual-en:start -->
*From the two-by-two formula to cofactor expansion*
<!-- bilingual-en:end -->

对 $\begin{pmatrix}a&b\\c&d\end{pmatrix}$，$\det A=ad-bc$。别把这里“按行命名”的 $b,c$ 与上图“按列命名”的坐标混在一起；两者都是左上乘右下，减右上乘左下。

[[余子式与代数余子式|余子式]]来自删掉第 $i$ 行、第 $j$ 列后的小矩阵。课程把这个**小矩阵**记作 $M_{ij}$；其行列式 $\det M_{ij}$ 是 minor，再乘 $(-1)^{i+j}$ 才是 cofactor（代数余子式）。不同教材有时直接用 $M_{ij}$ 指 minor，应先看定义。

[[余子式展开|Laplace 展开]]沿固定一行求和：
$$
\det A=\sum_{j=1}^{n}(-1)^{i+j}A_{ij}\det M_{ij}.
$$
棋盘符号从左上角开始是 $+,-,+;\ -,+,-;\ +,-,+$。一次展开要固定同一行或同一列。

<!-- bilingual-en:start -->
For a two-by-two matrix, multiply the main diagonal and subtract the other diagonal product. A minor is the determinant left after deleting one row and one column; a cofactor additionally includes the checkerboard sign. The lecture uses $M_{ij}$ for the remaining matrix, whereas some books use it for the minor itself. [[余子式展开|Laplace expansion]] fixes one row or column and sums each entry times its cofactor.
<!-- bilingual-en:end -->

例：沿第一行计算
$$
B=\begin{pmatrix}1&2&0\\0&3&1\\2&0&4\end{pmatrix}.
$$
$$
\begin{aligned}
\det B
&=1\begin{vmatrix}3&1\\0&4\end{vmatrix}
-2\begin{vmatrix}0&1\\2&4\end{vmatrix}
+0\begin{vmatrix}0&3\\2&0\end{vmatrix}\\
&=1(3\cdot4-1\cdot0)-2(0\cdot4-1\cdot2)+0\\
&=12-2(-2)=12+4=16.
\end{aligned}
$$
第二项有两层负号：外面的棋盘负号，以及小行列式算出的 $-2$。先保留括号，再合并，比较不容易漏号。

<!-- bilingual-en:start -->
Expanding the displayed matrix along its first row gives $12-2(-2)=16$. The second term contains both the cofactor sign and the negative value of its minor. Keeping the parentheses until the last step prevents these signs from being confused.
<!-- bilingual-en:end -->

## 3. 主子式与定号判据

<!-- bilingual-en:start -->
*Principal minors and definiteness tests*
<!-- bilingual-en:end -->

课程定位：[[EC400 Slides Lecture 1.pdf#page=24|slides 22–31]]。

### 3.1 主子矩阵、主子式、顺序主子式是三个对象

<!-- bilingual-en:start -->
*Three objects that must be distinguished*
<!-- bilingual-en:end -->

[[主子式|主子矩阵]]要求保留**同一组行号和列号**。若保留下标集 $I=\{1,3\}$，就保留第 $1,3$ 行和第 $1,3$ 列，得到 $A_{I,I}$。它仍是一张矩阵；取 $\det A_{I,I}$ 后才是一个主子式。保留下标不必连续，也不只是“删某条对角线附近”。

[[顺序主子式|顺序主子式（leading principal minor）]]更特殊：第 $k$ 阶只保留最前面的 $k$ 个下标 $\{1,\ldots,k\}$，也就是左上角 $k\times k$ 块，记 $L_k$。一般 $n$ 阶矩阵有 $\binom nk$ 个 $k$ 阶主子式，而该阶顺序主子式只有一个；所有非空阶合计 $2^n-1$ 个主子式，对比 $n$ 个顺序主子式。

<!-- bilingual-en:start -->
A [[主子式|principal submatrix]] retains the same index set for rows and columns. It is a matrix; its determinant is a principal minor. A [[顺序主子式|leading principal minor]] further requires the initial index set $\{1,\ldots,k\}$. There are $\binom nk$ principal minors of order $k$, but only one leading principal minor of that order.
<!-- bilingual-en:end -->

用同一个例子把它们列全：
$$
A=\begin{pmatrix}4&1&2\\1&5&3\\2&3&6\end{pmatrix}.
$$
一阶主子式是 $4,5,6$。二阶主子式按保留的下标分为
$$
\begin{array}{c|c|c}
I&A_{I,I}&\det A_{I,I}\\\hline
\{1,2\}&\begin{pmatrix}4&1\\1&5\end{pmatrix}&4\cdot5-1\cdot1=19\\[4pt]
\{1,3\}&\begin{pmatrix}4&2\\2&6\end{pmatrix}&4\cdot6-2\cdot2=20\\[4pt]
\{2,3\}&\begin{pmatrix}5&3\\3&6\end{pmatrix}&5\cdot6-3\cdot3=21
\end{array}
$$
三阶只有整个矩阵：
$$
\det A=4(30-9)-1(6-6)+2(3-10)=84-0-14=70.
$$
所以全部主子式是 $4,5,6,19,20,21,70$，顺序主子式是 $L_1=4,L_2=19,L_3=70$。若删第 $2$ 行、第 $3$ 列，剩下 $\begin{pmatrix}4&1\\2&3\end{pmatrix}$，行指标为 $\{1,3\}$、列指标为 $\{1,2\}$，因此不是主子矩阵。

<!-- bilingual-en:start -->
The example lists every principal minor and then singles out the leading sequence $4,19,70$. Keeping rows $1,3$ but columns $1,2$ gives a valid submatrix, but not a principal one. The selection of indices and the subsequent determinant calculation are separate steps.
<!-- bilingual-en:end -->

### 3.2 四条判据与使用顺序

<!-- bilingual-en:start -->
*The four sign tests and how to apply them*
<!-- bilingual-en:end -->

以下都以 **$A$ 为实对称矩阵**为前提。

| 要证明的性质 | 应检查的数 | 符号要求 |
|---|---|---|
| [[Sylvester 正定判据\|正定]] | 全部顺序主子式 $L_k$ | 每个 $L_k>0$ |
| [[负定顺序主子式判据\|负定]] | 全部顺序主子式 $L_k$ | 奇数阶 $<0$，偶数阶 $>0$ |
| [[半正定主子式判别\|半正定]] | **全部**主子式 | 每个都 $\ge0$ |
| [[半负定主子式判据\|半负定]] | **全部**主子式 | 奇数阶 $\le0$，偶数阶 $\ge0$ |

负定的符号为什么交替？因为 $A$ 负定当且仅当 $-A$ 正定。$k$ 阶子矩阵整体乘 $-1$ 后，其行列式乘 $(-1)^k$。对 $-A$ 使用正定判据，就得到 $(-1)^kL_k>0$。半负定的证明同理，只是换成全部主子式和非严格号。

<!-- bilingual-en:start -->
These tests assume a real symmetric matrix. Positive and negative definiteness use all leading principal minors, while semidefiniteness uses all principal minors. Negative signs alternate by order because negating a matrix negates all $k$ columns of each order-$k$ submatrix, multiplying its determinant by $(-1)^k$. Thus the negative tests follow from the corresponding positive tests applied to $-A$.
<!-- bilingual-en:end -->

不要把正定判据中的 $>0$ 换成 $\ge0$ 就宣布半正定。反例
$$
A=\begin{pmatrix}0&0\\0&-1\end{pmatrix}
$$
的 $L_1=L_2=0$，却有 $Q(0,1)=-1$。它是半负定，不是半正定；遗漏的主子式正是第二个对角元 $-1$。这条边界见[[顺序主子式不推半正定]]。

实际做题时，先尝试 PD、ND 的完整判据；不满足时，再判断是否半定，或者直接找正、负两个方向。若 $\det A\ne0$，又已排除 PD 和 ND，则 $A$ 必不定：非奇异半正定会自动正定，非奇异半负定会自动负定。这个结论不要求每个较小顺序主子式非零。若 $\det A=0$，仍可能不定，例如 $\operatorname{diag}(1,-1,0)$。**inconclusive（当前方法没有结论）与 indefinite（已证明存在两种符号）完全不同。**

<!-- bilingual-en:start -->
Nonnegative leading minors do not imply positive semidefiniteness, as [[顺序主子式不推半正定|the diagonal counterexample]] shows. After ruling out both definite cases, a nonsingular symmetric matrix must be indefinite; smaller leading minors may still vanish. A singular matrix may also be indefinite. “Inconclusive” describes an unfinished test, whereas “indefinite” is a definite mathematical classification requiring both signs.
<!-- bilingual-en:end -->

### 3.3 配方为什么能推出二阶判据

<!-- bilingual-en:start -->
*How completing the square explains the two-dimensional test*
<!-- bilingual-en:end -->

[[二维二次型配方|配方]]不是只为凑出平方；它把混在一起的方向重新组合，让符号直接可见。对
$$
A=\begin{pmatrix}a&b\\b&c\end{pmatrix},\qquad Q=ax_1^2+2bx_1x_2+cx_2^2,
$$
先假设 $a\ne0$。从前两项提出 $a$，在括号里补 $b^2x_2^2/a^2$，再把多加的部分减回去：
$$
\begin{aligned}
Q
&=a\left(x_1^2+2\frac ba x_1x_2\right)+cx_2^2\\
&=a\left(x_1^2+2\frac ba x_1x_2+\frac{b^2}{a^2}x_2^2\right)
-\frac{b^2}{a}x_2^2+cx_2^2\\
&=a\left(x_1+\frac ba x_2\right)^2
+\frac{ac-b^2}{a}x_2^2.
\end{aligned}
$$
记 $\Delta=ac-b^2=\det A$，$u=x_1+(b/a)x_2$，$v=x_2$。反过来 $x_2=v,x_1=u-(b/a)v$，所以 $(u,v)=(0,0)$ 当且仅当原向量为零。

<!-- bilingual-en:start -->
[[二维二次型配方|Completing the square]] is a change of coordinates that exposes the signs of two independent squared terms. The displayed steps add and subtract exactly the same amount. The transformed coordinates are invertible, so both squares vanish simultaneously only at the original zero vector. Division by $a$ requires $a\ne0$.
<!-- bilingual-en:end -->

现在只看两个系数 $a$ 和 $\Delta/a$：二者都正，$Q$ 正定；都负，$Q$ 负定；一正一负，$Q$ 不定。因此 PD 等价于 $a>0,\Delta>0$；ND 等价于 $a<0,\Delta>0$；$\Delta<0$ 时不定。若 $\Delta=0$，只剩一个带符号平方，因而半定但非定。

若 $a=0$，不能代入含 $1/a$ 的式子。此时 $Q(1,0)=0$，所以不可能正定或负定；若 $b\ne0$，$\Delta=-b^2<0$，交叉项可以制造两个符号；若 $b=0$，就只剩 $cx_2^2$，按 $c$ 的符号判断，$c=0$ 时是零型。

<!-- bilingual-en:start -->
The two coefficients determine the sign because the new coordinates vary independently. A positive determinant gives matching signs, selected by $a$; a negative determinant gives opposite signs. A zero determinant leaves one square. If $a=0$, use the original expression: definiteness is impossible, a nonzero cross term gives indefiniteness, and otherwise the sign is determined by $c$.
<!-- bilingual-en:end -->

两个相近的例子能看出交叉项的作用：
$$
\begin{aligned}
2x_1^2+6x_1x_2+7x_2^2
&=2\left(x_1+\frac32x_2\right)^2+\frac52x_2^2,\\
2x_1^2+8x_1x_2+7x_2^2
&=2(x_1+2x_2)^2-x_2^2.
\end{aligned}
$$
第一例的矩阵是 $\begin{pmatrix}2&3\\3&7\end{pmatrix}$，行列式 $14-9=5$，正定。第二例只把非对角元从 $3$ 改为 $4$，行列式却变成 $14-16=-2$；沿 $(1,0)$ 值为 $2$，沿 $(-2,1)$ 值为 $-1$，因此不定。对角元都正并不够，见[[正对角元的正定边界]]。

<!-- bilingual-en:start -->
Increasing the cross coefficient changes the second completed-square coefficient from positive to negative. In the second example, $(1,0)$ gives a positive value and $(-2,1)$ gives a negative one. [[正对角元的正定边界|Positive diagonal entries alone]] inspect only the coordinate directions and miss this interaction.
<!-- bilingual-en:end -->

### 3.4 三维参数例：把条件一步步收拢

<!-- bilingual-en:start -->
*A three-dimensional parameter example*
<!-- bilingual-en:end -->

slides 30–31 的完整题目是
$$
Q(x)=bx_1^2+a(x_2^2+x_3^2)+2bx_2x_3,
\qquad A=\begin{pmatrix}b&0&0\\0&a&b\\0&b&a\end{pmatrix}.
$$
这里 $a,b$ 是参数，$x_1,x_2,x_3$ 是输入，角色要分开。全部主子式为：一阶 $b,a,a$；二阶 $ab,ab,a^2-b^2$；三阶 $b(a^2-b^2)$。比如下标 $\{2,3\}$ 留下 $\begin{pmatrix}a&b\\b&a\end{pmatrix}$，其行列式为 $a^2-b^2$。

PSD 要求 $a\ge0,b\ge0$，并有 $a^2-b^2\ge0$。在已知两者非负后，最后一条等价于 $a\ge b$；其余乘积条件自动满足。所以 PSD 恰好是 $a\ge b\ge0$。ND/NSD 用同样步骤，得到
$$
\begin{array}{c|c}
\mathrm{PD}&a>b>0\\
\mathrm{PSD}&a\ge b\ge0\\
\mathrm{ND}&a<b<0\\
\mathrm{NSD}&a\le b\le0.
\end{array}
$$

<!-- bilingual-en:start -->
The parameters $a,b$ are separate from the input vector. Enumerating all principal minors yields the listed expressions. For positive semidefiniteness, the first-order conditions make both parameters nonnegative, after which $a^2-b^2\ge0$ reduces to $a\ge b$. The other products then have the required signs automatically. Strict inequalities give definiteness; reversing the matrix sign gives the negative cases.
<!-- bilingual-en:end -->

![[SOFP-L1-parameter-regions.png|850]]

为什么剩下的不定区域可以写成 $ab<b^2$？分 $b$ 的符号看：$b>0$ 时，不能半正定就意味着 $a<b$，而沿 $e_1$ 已有正值，所以必不定；$b<0$ 时，不能半负定就意味着 $a>b$，沿 $e_1$ 已有负值，也必不定。两种情况都等价于 $b(a-b)<0$。

边界也要单独读。$b=0$ 时 $Q=a(x_2^2+x_3^2)$，由 $a$ 的符号决定半正定或半负定；$a=b\ne0$ 时 $Q=b[x_1^2+(x_2+x_3)^2]$，有非零零方向；$a=b=0$ 时为零型。同时还可以令 $u=(x_2+x_3)/\sqrt2,v=(x_2-x_3)/\sqrt2$，直接核对
$$
Q=bx_1^2+(a+b)u^2+(a-b)v^2.
$$
这样主子式方法与配方后的符号检查给出同一答案，背后的统一原则见[[正定与半正定的谱判据]]。

<!-- bilingual-en:start -->
Splitting by the sign of $b$ shows that the indefinite region is exactly $b(a-b)<0$. On $b=0$ and $a=b$, a nonzero zero direction remains; at the origin the form is identically zero. The displayed orthogonal change of coordinates independently checks the classification by exposing coefficients $b,a+b,a-b$, consistent with [[正定与半正定的谱判据|the spectral criterion]].
<!-- bilingual-en:end -->

## 4. Taylor 展开之前：把多元微分的符号接起来

<!-- bilingual-en:start -->
*Preparing for Taylor: the chain from partial derivatives to the Hessian*
<!-- bilingual-en:end -->

这一段补足 slides 38–39 直接使用的微分语言，也对应手写第 1–4 页。可以先把问题分成两层：一阶导数问“向这个方向走，最初升还是降”；二阶导数问“沿这个方向的斜率怎样变化”。在候选极值点，一阶项消失，二阶信息才开始起关键作用。

<!-- bilingual-en:start -->
This section supplies the differential notation used directly on slides 38–39 and develops handwritten pages 1–4. First derivatives measure the initial rate of change in a direction; second derivatives measure how that rate changes. At a stationary candidate, the first-order term vanishes and second-order information becomes central.
<!-- bilingual-en:end -->

### 4.1 偏导数、行导数和列梯度

<!-- bilingual-en:start -->
*Partial derivatives, the derivative row, and the gradient column*
<!-- bilingual-en:end -->

[[偏导数]]一次只让一个坐标变化。对第 $i$ 个标准基向量 $e_i$，
$$
f_i(a)=\frac{\partial f}{\partial x_i}(a)
=\lim_{s\to0}\frac{f(a+se_i)-f(a)}s.
$$
例如 $e_1=(1,0)^T$，所以 $a+se_1=(a_1+s,a_2)^T$。$s$ 是该坐标的微小增量；另一个坐标保持不变。符号 $\partial$ 提醒我们这是对多个输入中的一个求导。

以
$$
f(x_1,x_2)=-x_1^2-x_2^2+2x_1+4x_2
$$
为例，对 $x_1$ 的差商为
$$
\begin{aligned}
\frac{f(x_1+s,x_2)-f(x_1,x_2)}s
&=\frac{-(x_1+s)^2+x_1^2+2s}{s}\\
&=\frac{-x_1^2-2x_1s-s^2+x_1^2+2s}s\\
&=-2x_1-s+2\longrightarrow-2x_1+2.
\end{aligned}
$$
只含 $x_2$ 的项完全相消。同理 $f_2=-2x_2+4$。

<!-- bilingual-en:start -->
A [[偏导数|partial derivative]] changes only one coordinate, using a standard basis vector to specify which one. In the example, all terms depending only on the fixed coordinate cancel from the difference quotient. Expanding the remaining square, dividing by the increment, and taking its limit gives the stated derivative.
<!-- bilingual-en:end -->

课程把这些偏导排成**行向量**：
$$
Df(x)=\begin{pmatrix}f_1(x)&f_2(x)\end{pmatrix}
=\begin{pmatrix}-2x_1+2&-2x_2+4\end{pmatrix}.
$$
通常写成列向量的梯度是 $\nabla f(x)=Df(x)^T$。当 $f$ 可微时，$Df$ 是标量输出函数的 [[Jacobian矩阵|Jacobian]]，也是[[全微分]]的矩阵表示。**把偏导排成一行，不等于已经证明函数可微**；可微还要求这一行提供的线性近似能同时控制所有小扰动。

课程在光滑函数中把 $Df(x^*)=0$ 的点称为 critical point，本文称[[驻点|驻点]]。等号右边的 $0$ 是零行向量，意思是每个偏导都为零。上述例子给出 $-2x_1+2=0,-2x_2+4=0$，于是候选点是 $(1,2)$。其他教材的[[临界点]]可能还包含不可导点；阅读时要辨认这个术语差异。

<!-- bilingual-en:start -->
The course arranges partial derivatives in a row, while the gradient is its transpose. Under differentiability this row is the scalar-valued [[Jacobian矩阵|Jacobian]] representing the [[全微分|total differential]]. Mere existence of the partial derivatives does not establish differentiability. Here a [[驻点|stationary point]] has every partial derivative equal to zero, matching the lecture's usage of “critical point.” Some textbooks use [[临界点|critical point]] more broadly to include nondifferentiable points.
<!-- bilingual-en:end -->

### 4.2 为什么令偏导为零只能找候选点

<!-- bilingual-en:start -->
*Why zero partial derivatives give only candidates*
<!-- bilingual-en:end -->

[[无约束一阶条件|内点极值的一阶必要条件]]来自一元结论。若 $x^*$ 是定义域内部的局部最大点，固定其他坐标，只沿 $e_i$ 走，函数 $t\mapsto f(x^*+te_i)$ 在 $t=0$ 也必须局部最大。可导时，一元极值必要条件使其导数为零，所以每个 $f_i(x^*)=0$。

“内点”保证每条坐标线都能向前、向后走一小段。边界没有这项保证：$f(x)=x$ 在 $[0,1]$ 上最大于 $1$，但导数是 $1$。反过来，$f(x)=x^3$ 在 $0$ 导数为零却不极大也不极小。零导数是候选条件，尚未解决分类。

<!-- bilingual-en:start -->
The [[无约束一阶条件|interior first-order necessary condition]] follows by restricting the function to each coordinate line and applying the one-dimensional extremum condition. Being interior permits small moves of either sign. Boundary maxima need not have zero derivatives, and a zero derivative need not identify an extremum, as $x^3$ at zero shows.
<!-- bilingual-en:end -->

### 4.3 用一条直线，把多元函数变成一元函数

<!-- bilingual-en:start -->
*Restricting a multivariable function to a line*
<!-- bilingual-en:end -->

固定基点 $a$ 和向量 $h$，设
$$
x(t)=a+th,\qquad g(t)=f(a+th).
$$
$a$ 是经过的点；$h$ 同时指定方向和参数变化的速度；$t$ 是唯一自由变量。$t=0$ 时在 $a$，$t=1$ 时到 $a+h$。从 $a$ 到 $a+th$ 的实际距离是 $|t|\|h\|$，只有 $\|h\|=1$ 时，$|t|$ 才等于距离。欧氏长度 $\|h\|=\sqrt{\sum_i h_i^2}$，例如 $\|(3,4)\|=\sqrt{9+16}=5$。

取 $a=(1,2)^T,h=(1,1)^T$，直线上 $t=-1,0,1,2$ 分别对应 $(0,1),(1,2),(2,3),(3,4)$。代入刚才的 $f$：
$$
\begin{aligned}
g(t)&=-(1+t)^2-(2+t)^2+2(1+t)+4(2+t)\\
&=(-1-2t-t^2)+(-4-4t-t^2)+(2+2t)+(8+4t)\\
&=5-2t^2.
\end{aligned}
$$
代入完成后，$g$ 中只剩 $t$；$x_1,x_2$ 都已被具体路径替换。

<!-- bilingual-en:start -->
Fixing $a$ and $h$ leaves one variable, $t$. The base point is $a$, while $h$ specifies direction and speed in this parameterization. Physical distance is $|t|\|h\|$. Substituting the displayed line into the example and collecting every term gives $g(t)=5-2t^2$. A completed substitution leaves no free $x_1$ or $x_2$ in $g$.
<!-- bilingual-en:end -->

[[方向导数|沿 $h$ 的方向导数]]定义为
$$
D_hf(a)=\lim_{t\to0}\frac{f(a+th)-f(a)}t
=\lim_{t\to0}\frac{g(t)-g(0)}t=g'(0).
$$
它等于 $g'(0)$，因为右边恰好就是 $g$ 在 **$0$ 点的一元导数定义**；不是因为 $g'(0)$ 等于 $\lim g(t)$。后一极限在连续时给的是函数值 $g(0)$。

本课不强迫 $h$ 是单位向量。[[方向导数尺度|把 $h$ 换成 $\lambda h$]]，一阶变化率乘 $\lambda$，二阶变化率乘 $\lambda^2$。例如同一路线走得快两倍，每单位 $t$ 的一阶变化也快两倍。不要在计算中擅自归一化，因为那会改变题目指定的参数。

<!-- bilingual-en:start -->
The [[方向导数|directional derivative]] is $g'(0)$ because its quotient is exactly the one-variable derivative definition at zero. It is not the limit of $g(t)$ itself. Under the lecture's convention the direction vector need not be a unit vector. [[方向导数尺度|Rescaling it]] changes the first derivative linearly and the second derivative quadratically, so normalizing it would change the requested rate.
<!-- bilingual-en:end -->

### 4.4 链式法则：加减中间点的完整推导

<!-- bilingual-en:start -->
*The chain rule, derived by adding an intermediate point*
<!-- bilingual-en:end -->

每次先展开 $g(t)$ 再求导很费力。[[多元链式法则]]提供捷径：若 $f$ 在路径附近为 $C^1$，即一阶偏导存在且连续，那么
$$
g'(t)=Df(a+th)h=\sum_i f_i(a+th)h_i.
$$
手写第 3 页的 $\oplus$ 对应下面这段推导。为看清为什么会乘上 $h_i$，先在二维、$t=0$ 处证明。令 $\delta$ 是 $t$ 的增量，把起点和终点之间插入
$$
m=(a_1+h_1\delta,a_2).
$$
只是在一个差值里加上再减去 $f(m)$：
$$
\begin{aligned}
g(\delta)-g(0)
={}&\underbrace{f(a_1+h_1\delta,a_2+h_2\delta)-f(a_1+h_1\delta,a_2)}_{I:\ x_2\text{ changes}}\\
&+\underbrace{f(a_1+h_1\delta,a_2)-f(a_1,a_2)}_{II:\ x_1\text{ changes}}.
\end{aligned}
$$
这样，两坐标同时变化的差，被精确拆成了两个“只变一个坐标”的差。路径换了，但总函数值之差没有变。

<!-- bilingual-en:start -->
The [[多元链式法则|multivariable chain rule]] avoids expanding the entire line restriction. To prove its two-variable version under continuous partial derivatives, insert the intermediate point that changes only the first coordinate. Adding and subtracting its function value splits the original difference exactly into two single-coordinate changes. This is the derivation requested on handwritten page 3.
<!-- bilingual-en:end -->

先处理 $II$。若 $h_1\ne0$，记 $s=h_1\delta$，则
$$
\begin{aligned}
\frac{II}{\delta}
&=\frac{f(a_1+s,a_2)-f(a_1,a_2)}s\cdot\frac{s}{\delta}\\
&=\frac{f(a_1+s,a_2)-f(a_1,a_2)}s\cdot h_1
\longrightarrow f_1(a)h_1.
\end{aligned}
$$
$h_1$ 是固定非零数，所以 $\delta\to0$ 时 $s\to0$。这个乘法是在把“每单位 $x_1$ 的变化”换成“每单位 $t$ 的变化”，因为 $\Delta x_1/\Delta t=h_1$。例如每向东一米上升两米，而每秒向东三米，就每秒上升六米。若 $h_1=0$，$II$ 本来就等于零，直接处理，不作除法。

<!-- bilingual-en:start -->
For the first-coordinate change, substitute $s=h_1\delta$. The quotient becomes the coordinate derivative times the exact conversion factor $s/\delta=h_1$. This changes the rate per unit of the coordinate into the rate per unit of the path parameter. When $h_1=0$, the coordinate does not change and this contribution is zero without division.
<!-- bilingual-en:end -->

再处理 $I$。固定第一坐标 $a_1+h_1\delta$，对第二坐标用一元中值定理。存在位于 $a_2$ 与 $a_2+h_2\delta$ 之间的 $\xi_\delta$，使
$$
I=f_2(a_1+h_1\delta,\xi_\delta)\,h_2\delta.
$$
因此
$$
\frac I\delta=f_2(a_1+h_1\delta,\xi_\delta)h_2
\longrightarrow f_2(a)h_2.
$$
这里用到了 $f_2$ 的连续性：求偏导的位置也随着 $\delta$ 在移动；只知道 $f_2(a)$ 存在，不能保证附近的值趋近它。$h_2=0$ 时仍直接得到零。把两项相加，就有
$$
g'(0)=f_1(a)h_1+f_2(a)h_2=Df(a)h.
$$
把基点换成路径上的 $a+th$，就得到一般 $t$ 的公式。$C^1$ 是这份证明采用的方便充分条件；更一般地，只要 $f$ 在所求点可微，也能由线性近似推出[[方向导数梯度公式]]。

<!-- bilingual-en:start -->
The mean value theorem handles the second-coordinate change, but its evaluation point moves with the increment. Continuity is what allows the partial derivative there to converge to the value at $a$. Adding the two limits yields the formula. Repeating at another point of the path gives the general result. Continuous partial derivatives are a convenient sufficient assumption for this proof; differentiability alone at the evaluation point also yields the [[方向导数梯度公式|directional derivative formula]].
<!-- bilingual-en:end -->

这项条件不能无声省掉。令
$$
f(x_1,x_2)=\begin{cases}
\dfrac{x_1x_2}{x_1^2+x_2^2},&(x_1,x_2)\ne(0,0),\\[3pt]
0,&(x_1,x_2)=(0,0).
\end{cases}
$$
沿两条坐标轴函数恒为零，所以两个偏导在原点都是零；沿 $(t,t)$ 却有 $f(t,t)=1/2$（$t\ne0$），方向导数的差商是 $1/(2t)$，没有有限极限。这正是[[偏导不可推可微]]的边界。

<!-- bilingual-en:start -->
The displayed function has zero coordinate partial derivatives at the origin but equals one half along every nonzero point of the diagonal. Its diagonal directional difference quotient has no finite limit. [[偏导不可推可微|Existence of partial derivatives does not imply differentiability]], so the chain-rule conclusion cannot be inferred from the two coordinate values alone.
<!-- bilingual-en:end -->

### 4.5 为什么一阶只需坐标方向，二阶却不够

<!-- bilingual-en:start -->
*Why coordinate directions determine first-order but not second-order information*
<!-- bilingual-en:end -->

可微时，$Df(a)h=\sum_i f_i(a)h_i$ 对 $h$ **线性**。知道 $n$ 个系数 $f_i(a)$，就知道任意方向的一阶变化率。例如在 $a=(3,0)$，前面的函数给出 $Df(a)=(-4,4)$；沿 $(1,0)$ 的变化率为 $-4$，沿 $(1,2)$ 为 $-4+8=4$。若所有偏导为零，任意方向的一阶变化率都自动为零。

但这只消除一阶变化，不消除二阶交叉影响。取 $f(x_1,x_2)=x_1x_2$：沿两条坐标轴它恒为零；沿 $(t,t)$ 是 $t^2$，沿 $(t,-t)$ 是 $-t^2$。坐标轴没有显示出来的弯曲，藏在交叉项里。

<!-- bilingual-en:start -->
For a differentiable function, the first-order response is linear in direction, so its $n$ coordinate coefficients determine every directional response. If those coefficients vanish, all first directional derivatives vanish. Second-order behaviour also contains interactions: $x_1x_2$ vanishes on both axes but becomes $t^2$ and $-t^2$ on the two diagonals.
<!-- bilingual-en:end -->

### 4.6 Hessian：先算一般表达式，再代入具体点

<!-- bilingual-en:start -->
*Computing and evaluating the Hessian*
<!-- bilingual-en:end -->

[[Hessian矩阵]]把二阶偏导排成方阵。本课约定
$$
f_{ij}=\frac{\partial}{\partial x_j}\left(\frac{\partial f}{\partial x_i}\right),
\qquad H(x)=D^2f(x)=(f_{ij}(x)).
$$
读下标时，先对 $x_i$ 求导，再对 $x_j$ 求导。$f_{ii}$ 是纯二阶偏导，$f_{ij}$（$i\ne j$）是混合偏导。若二阶偏导在邻域内连续，[[Hessian对称条件|Clairaut–Schwarz 定理]]保证 $f_{ij}=f_{ji}$，从而 $H^T=H$。这是一条带条件的定理，不是符号自动保证的性质。

四点差 $f(x+s,y+t)-f(x+s,y)-f(x,y+t)+f(x,y)$ 可以帮助理解混合作用为何对两个增量对称；但把两个极限的次序交换，仍需要相应正则条件。

<!-- bilingual-en:start -->
The [[Hessian矩阵|Hessian]] collects second partial derivatives. In this convention, the first index identifies the first differentiation and the second index the next one. [[Hessian对称条件|The Clairaut–Schwarz condition]] guarantees symmetry when the second partials are continuous nearby. A symmetric four-point difference suggests the result intuitively, but does not by itself justify exchanging limits.
<!-- bilingual-en:end -->

例：$f=x_1^2x_2+x_2^3$。先分别求一阶偏导，再对每个结果求两次方向的偏导：
$$
f_1=2x_1x_2,\qquad f_2=x_1^2+3x_2^2,
$$
$$
f_{11}=2x_2,\quad f_{12}=2x_1,\quad f_{21}=2x_1,\quad f_{22}=6x_2.
$$
所以
$$
H(x_1,x_2)=\begin{pmatrix}2x_2&2x_1\\2x_1&6x_2\end{pmatrix},
\qquad H(1,2)=\begin{pmatrix}4&2\\2&12\end{pmatrix}.
$$
左边仍是“输入一个点，输出一张矩阵”的函数；右边才是某一点上的数值矩阵。$n\times n$ 对称 Hessian 虽有 $n^2$ 个位置，独立条目只有 $n+n(n-1)/2=n(n+1)/2$ 个：对角元 $n$ 个，上三角非对角元 $n(n-1)/2$ 个。

<!-- bilingual-en:start -->
Differentiate each first partial with respect to every coordinate, then evaluate the resulting matrix at the requested point. The general matrix-valued function and its numerical value at $(1,2)$ are different objects. A symmetric Hessian has $n(n+1)/2$ independent entries, counting its diagonal and one triangular half.
<!-- bilingual-en:end -->

### 4.7 再用一次链式法则，得到方向二阶导数

<!-- bilingual-en:start -->
*A second application of the chain rule*
<!-- bilingual-en:end -->

从已经证明的 $g'(t)=f_1(a+th)h_1+f_2(a+th)h_2$ 出发。$h_1,h_2$ 固定，变的是 $f_1,f_2$ 的输入。分别求导：
$$
\frac d{dt}f_1(a+th)=f_{11}(a+th)h_1+f_{12}(a+th)h_2,
$$
$$
\frac d{dt}f_2(a+th)=f_{21}(a+th)h_1+f_{22}(a+th)h_2.
$$
再乘回外面的 $h_1,h_2$，得到四项：
$$
\begin{aligned}
g''(t)
&=[f_{11}h_1+f_{12}h_2]h_1+[f_{21}h_1+f_{22}h_2]h_2\\
&=f_{11}h_1^2+f_{12}h_1h_2+f_{21}h_2h_1+f_{22}h_2^2.
\end{aligned}
$$
这一行中的偏导全部在 $a+th$ 评价。若 $f\in C^2$，两个混合偏导相等，便得到 $f_{11}h_1^2+2f_{12}h_1h_2+f_{22}h_2^2$。**两个交叉项都要保留；最后一个平方项对应 $f_{22}$。**

<!-- bilingual-en:start -->
Apply the chain rule separately to the two first partials, then multiply by their existing constant direction coefficients. This produces four terms before symmetry is used. The mixed terms combine only after $f_{12}=f_{21}$ has been justified; the final squared term contains $f_{22}$.
<!-- bilingual-en:end -->

矩阵写法把同一件事压缩为[[Hessian方向二阶导数|方向二阶导数恒等式]]：
$$
g''(0)=h^TH(a)h.
$$
先算 $Hh$，再左乘 $h^T$，维度仍是 $(1\times n)(n\times n)(n\times1)$。若只知道坐标方向的二阶变化率，得到的只是 $H_{ii}$。要恢复交叉项，可以再测 $e_i+e_j$ 方向，利用
$$
H_{ij}=\frac{q(e_i+e_j)-q(e_i)-q(e_j)}2,
\qquad q(h)=h^THh.
$$
所以可以用 $n(n+1)/2$ 个**恰当选择**的方向恢复对称 Hessian，不能说任意这么多条切线都足够。这个独立判断保存在[[坐标二阶导数不确定Hessian]]。

<!-- bilingual-en:start -->
The [[Hessian方向二阶导数|second directional derivative identity]] is the matrix form of the same expansion. Coordinate directions reveal only diagonal entries. Adding the directions $e_i+e_j$ recovers the mixed entries using the displayed formula. Thus a suitably chosen set of $n(n+1)/2$ measurements determines the symmetric Hessian; an arbitrary set of that size need not do so. See [[坐标二阶导数不确定Hessian|the coordinate-direction boundary]].
<!-- bilingual-en:end -->

完整核对一次：$f=x_1^2+3x_1x_2+x_2^2$，$a=(1,1)$，$h=(1,2)^T$。
$$
H=\begin{pmatrix}2&3\\3&2\end{pmatrix},\qquad
Hh=\begin{pmatrix}8\\7\end{pmatrix},\qquad h^THh=1\cdot8+2\cdot7=22.
$$
直接代入：
$$
\begin{aligned}
g(t)&=(1+t)^2+3(1+t)(1+2t)+(1+2t)^2\\
&=(1+2t+t^2)+(3+9t+6t^2)+(1+4t+4t^2)\\
&=5+15t+11t^2,
\end{aligned}
$$
所以 $g''(0)=2\cdot11=22$，两种方法一致。要核对二阶导数时，特别注意 $t^2$ 项的系数还要乘 $2$。

<!-- bilingual-en:start -->
The matrix calculation gives $22$. Expanding the line restriction gives a quadratic coefficient of $11$, whose second derivative is $22$. Agreement between these independent routes checks both the substitution and the Hessian multiplication.
<!-- bilingual-en:end -->

### 4.8 二次型的 Hessian 为什么是两倍系数矩阵

<!-- bilingual-en:start -->
*Why a quadratic form has Hessian twice its symmetric matrix*
<!-- bilingual-en:end -->

[[二次型的Hessian|若 $Q(x)=x^TAx$ 且 $A=A^T$，则 $H_Q=2A$]]。二维中
$$
Q=A_{11}x_1^2+2A_{12}x_1x_2+A_{22}x_2^2,
$$
$$
Q_1=2A_{11}x_1+2A_{12}x_2,\quad
Q_2=2A_{12}x_1+2A_{22}x_2,
$$
$$
H_Q=\begin{pmatrix}2A_{11}&2A_{12}\\2A_{12}&2A_{22}\end{pmatrix}=2A.
$$
对角元的 $2$ 来自平方求导，非对角元的 $2$ 来自原来的两个对称交叉项。一般维度同样得到 $\nabla Q=2Ax$，课程行记号是 $DQ=2x^TA$；若原始矩阵不对称，则 Hessian 为 $A+A^T$。

因此纯二次型满足 $h^TH_Qh=2h^TAh=2Q(h)$。若函数另有一次项或常数项，Hessian 仍不受这些项影响，但最后等于 $2f(h)$ 就不再成立。

<!-- bilingual-en:start -->
For a [[二次型的Hessian|quadratic form with a symmetric matrix]], the Hessian is $2A$. The factor two in diagonal entries comes from differentiating a square, while off-diagonal entries reflect the two cross terms already present. Linear and constant additions leave the Hessian unchanged, but invalidate the special equality between the directional second derivative and twice the entire function value.
<!-- bilingual-en:end -->

## 5. Taylor 展开：把局部信息变成近似

<!-- bilingual-en:start -->
*Taylor expansion turns local derivatives into an approximation*
<!-- bilingual-en:end -->

课程定位：[[EC400 Slides Lecture 1.pdf#page=36|slides 34–39]]。

### 5.1 一阶近似从导数定义直接来

<!-- bilingual-en:start -->
*First-order approximation follows from the derivative definition*
<!-- bilingual-en:end -->

一元函数在 $a$ 可导，意思是
$$
\frac{f(a+h)-f(a)}h\longrightarrow f'(a).
$$
把 $f'(a)$ 移到左边，合并成一个分式，就得到
$$
\frac{f(a+h)-f(a)-f'(a)h}{h}\longrightarrow0.
$$
于是定义[[Taylor多项式|一阶 Taylor 多项式]]与[[Taylor余项|余项]]：
$$
P_1(h\mid a)=f(a)+f'(a)h,\qquad R_1(h\mid a)=f(a+h)-P_1(h\mid a).
$$
$a$ 是展开中心，$h$ 是从中心走出的增量；$P_1(h\mid a)$ 预测的是 $f(a+h)$，不是 $f(h)$。竖线把中心这个参数单独列出，不表示条件概率。

余项满足 $R_1/h\to0$，也记为 $R_1=o(|h|)$：误差相对于步长越来越小，远强于“误差本身趋零”。例如 $R=h$ 本身趋零，却有 $R/h=1$，不满足这个要求。

<!-- bilingual-en:start -->
Rearranging the derivative definition gives the first-order [[Taylor多项式|Taylor polynomial]] and a [[Taylor余项|remainder]] negligible relative to the step size. The center is fixed and the step is the displacement, so the target is $f(a+h)$. The separator merely identifies the center. A small-o remainder is stronger than an error that simply tends to zero.
<!-- bilingual-en:end -->

### 5.2 二阶项为什么除以二，高阶项为什么除以阶乘

<!-- bilingual-en:start -->
*Why Taylor coefficients contain factorials*
<!-- bilingual-en:end -->

让 $P(h)=c_0+c_1h+c_2h^2+c_3h^3+\cdots$ 在 $h=0$ 处与 $f(a+h)$ 匹配，就要求
$$
P(0)=c_0=f(a),\quad P'(0)=c_1=f'(a),\quad P''(0)=2c_2=f''(a).
$$
所以 $c_2=f''(a)/2$。$h^k$ 连续求 $k$ 次导数会乘出 $k!=k(k-1)\cdots1$，因此
$$
P_k(h\mid a)=\sum_{j=0}^k\frac{f^{(j)}(a)}{j!}h^j.
$$
$f^{(0)}=f$，$0!=1$。在足够的光滑条件下，例如邻域内 $f\in C^k$，有 $f(a+h)=P_k(h\mid a)+o(|h|^k)$。有限阶多项式不等于无限[[Taylor级数]]，更不自动保证级数等于函数。

<!-- bilingual-en:start -->
Matching the value and successive derivatives at the center forces the factorial denominators. Under sufficient smoothness, such as $C^k$ near the center, the remainder is negligible relative to the $k$th power of the step. A finite polynomial is different from an infinite [[Taylor级数|Taylor series]] and from the question of whether that series equals the function.
<!-- bilingual-en:end -->

课件取 $f(x)=e^x,a=0$，各阶导数在零点都为 $1$：
$$
P_1(h\mid0)=1+h,\qquad P_2(h\mid0)=1+h+h^2/2.
$$
$h=0.2$ 时，$P_1=1.2$，$P_2=1+0.2+0.04/2=1.22$，真值约 $1.221402758$；一次误差约 $0.021402758$，二次误差约 $0.001402758$。

再看 $f(x)=-x^2+4x$ 在 $a=2$：$f(2)=4,f'(2)=0,f''(2)=-2$，因此
$$
f(2+h)=4+0h+\tfrac12(-2)h^2=4-h^2.
$$
余项恰好为零，因为原函数就是二次多项式。通常提高阶数改善的是 $h\to0$ 时的局部误差阶，并不保证任意远处、任意固定步长上的误差都逐阶下降。

<!-- bilingual-en:start -->
For the exponential, the second-order polynomial improves the displayed numerical approximation. For an actual quadratic polynomial the second-order formula is exact. Higher order improves the local asymptotic error scale; it does not guarantee monotonically improving accuracy at every fixed or distant input.
<!-- bilingual-en:end -->

### 5.3 多元一阶和二阶 Taylor 的每一项

<!-- bilingual-en:start -->
*Reading each multivariable Taylor term*
<!-- bilingual-en:end -->

[[多元Taylor近似]]把一元增量换成向量。若 $F$ 在 $a$ 可微，
$$
F(a+h)=F(a)+DF(a)h+R_1(h\mid a),\qquad R_1/\|h\|\to0.
$$
若 $F$ 在邻域为 $C^2$，
$$
\boxed{F(a+h)=F(a)+DF(a)h+\frac12h^TD^2F(a)h+R_2(h\mid a)},
\qquad R_2/\|h\|^2\to0.
$$
依次读成：基准值、一阶线性变化、二阶曲率修正、尚未被前两阶解释的误差。在二维、Hessian 对称时，
$$
\frac12h^THh=\frac12f_{11}h_1^2+f_{12}h_1h_2+\frac12f_{22}h_2^2.
$$
外面的 $1/2$ 使两个相同交叉项合成一个，也让平方项恢复正确系数。$h^THh$ 是沿路径的二阶导数，仍要除以 $2!$ 才是 Taylor 修正。

<!-- bilingual-en:start -->
[[多元Taylor近似|Multivariable Taylor expansion]] combines a baseline, a linear response, a quadratic correction, and a remainder. The factor one half combines the two equal cross terms and restores the squared-term coefficients. The Hessian quadratic form is the second derivative along a line, not yet its Taylor contribution.
<!-- bilingual-en:end -->

$C^2$ 一般只保证 $o(\|h\|^2)$。要直接写 $O(\|h\|^3)$，需更强条件，例如三阶导数在邻域连续，从而局部有界。$o$ 表示比例趋零，$O$ 表示比例的绝对值在附近有统一上界，不能凭印象互换。

<!-- bilingual-en:start -->
Under $C^2$ smoothness the standard remainder is second-order small-o. A cubic big-O bound needs stronger regularity, such as continuous third derivatives nearby. Small-o requires the ratio to tend to zero; big-O requires a local uniform bound on its absolute value.
<!-- bilingual-en:end -->

### 5.4 为什么沿直线可以在参数一处取值

<!-- bilingual-en:start -->
*Why the line parameter can be evaluated at one*
<!-- bilingual-en:end -->

固定一个小向量 $h$，设 $g_h(t)=F(a+th)$，则 $g_h(0)=F(a)$、$g_h(1)=F(a+h)$，而
$$
g_h'(0)=DF(a)h,\qquad g_h''(t)=h^TD^2F(a+th)h.
$$
最后取 $t=1$，并不是认为任意函数在离中心一个单位处都能近似得很好。真正趋零的是 **$h$**；$h$ 缩小时，$0\le t\le1$ 对应的整条线段都缩进 $a$ 的小邻域。

<!-- bilingual-en:start -->
A different one-variable function is defined for each displacement. Evaluating it at parameter one reaches $a+h$. Accuracy comes from shrinking the displacement, not from assuming a fixed Taylor approximation is accurate one unit away. The entire segment contracts toward the center.
<!-- bilingual-en:end -->

两次应用微积分基本定理，可把余项写得很具体：
$$
g_h(1)=g_h(0)+g_h'(0)+\int_0^1(1-t)g_h''(t)\,dt.
$$
在 Hessian 中加减 $D^2F(a)$，再用 $\int_0^1(1-t)dt=1/2$，得到
$$
R_2=\int_0^1(1-t)h^T[D^2F(a+th)-D^2F(a)]h\,dt.
$$
这里用[[谱范数|矩阵算子范数（诱导二范数）]]衡量矩阵大小：$\|B\|=\max_{\|v\|=1}\|Bv\|$，意思是矩阵能把单位向量的长度最多放大多少。因此 $\|Bh\|\le\|B\|\|h\|$，再用内积的[[Cauchy–Schwarz不等式|Cauchy–Schwarz 不等式]]，有 $|h^TBh|\le\|h\|\|Bh\|\le\|B\|\|h\|^2$。下面的 $\sup_{0\le t\le1}$ 表示沿整条线段取上确界；在这里矩阵随 $t$ 连续，也就是所能达到的最大值。

<!-- bilingual-en:start -->
The [[谱范数|induced Euclidean matrix norm]] measures the largest stretch of a unit vector. Combining its length bound with [[Cauchy–Schwarz不等式|Cauchy–Schwarz]] bounds the quadratic form by the matrix norm times the squared displacement length. The supremum takes the bound over the entire segment; continuity makes it an attained maximum here.
<!-- bilingual-en:end -->

把这个界用于积分中的每一个 $t$，再乘上 $\int_0^1(1-t)dt=1/2$，得到
$$
\frac{|R_2|}{\|h\|^2}\le\frac12\sup_{0\le t\le1}\|D^2F(a+th)-D^2F(a)\|\longrightarrow0.
$$
最后一步来自 Hessian 连续性：因为 $\|th\|\le\|h\|$，整条短线段都落在以 $a$ 为中心、半径 $\|h\|$ 的邻域里，线段上的 Hessian 因而一致地接近中心的 Hessian。这个推导补上了“把一元公式套到 $t=1$”中不能省略的误差控制。

<!-- bilingual-en:start -->
The integral remainder gives uniform control over the segment. Adding and subtracting the Hessian at the center separates out the quadratic term, with coefficient one half. Hessian continuity makes the supremum of the remaining difference tend to zero, establishing the required small-o bound.
<!-- bilingual-en:end -->

### 5.5 完整数值例：中心信息、增量和误差

<!-- bilingual-en:start -->
*A numerical example with all terms and errors*
<!-- bilingual-en:end -->

取 $F=x_1^2+x_1x_2+3x_2$，$a=(1,1)^T$，$h=(0.1,0.2)^T$。先列中心信息：
$$
F(a)=5,\quad DF(x)=(2x_1+x_2,x_1+3),\quad DF(a)=(3,4),
\quad H=\begin{pmatrix}2&1\\1&0\end{pmatrix}.
$$
一次项 $DF(a)h=3(0.1)+4(0.2)=1.1$，所以 $P_1=6.1$。再算
$$
Hh=\begin{pmatrix}2(0.1)+0.2\\0.1\end{pmatrix}=\begin{pmatrix}0.4\\0.1\end{pmatrix},
\quad h^THh=0.1(0.4)+0.2(0.1)=0.06.
$$
二次修正为 $0.06/2=0.03$，所以 $P_2=6.13$。直接计算
$$
F(1.1,1.2)=1.21+1.32+3.6=6.13.
$$
原函数是二次多项式，所以二阶展开精确。

<!-- bilingual-en:start -->
Evaluate the function, derivative row, and Hessian at the center before using the displacement. The linear term is $1.1$, the Hessian quadratic form is $0.06$, and its Taylor correction is $0.03$. The result agrees with direct evaluation because the original function is quadratic.
<!-- bilingual-en:end -->

把增量缩小十倍为 $(0.01,0.02)$，$P_1=5.11$，真值 $5.1103$，一次余项从 $0.03$ 缩为 $0.0003$。原步长长度约 $0.223607$，相对误差 $|R_1|/\|h\|\approx0.134164$；新步长长度约 $0.022361$，相对误差约 $0.013416$。这个例子中，步长缩成十分之一，误差缩成百分之一，因此相对步长的误差也缩成十分之一。

<!-- bilingual-en:start -->
Reducing the displacement by ten reduces this example's first-order remainder by one hundred. Dividing by the displacement norm then reduces the relative error by ten, illustrating the small-o statement numerically.
<!-- bilingual-en:end -->

再用课堂的 $G=x_1^2+2x_2^2$ 专门检查 $1/2$。$a=(1,1),h=(0.1,0.1)$ 时，$G(a)=3,DG(a)=(2,4),H=\operatorname{diag}(2,4)$，所以
$$
P_2=3+0.6+\tfrac12(0.06)=3.63.
$$
直线代入必须是
$$
g(t)=(1+0.1t)^2+2(1+0.1t)^2=3+0.6t+0.03t^2,
$$
取 $t=1$ 也得到 $3.63$。自查时分别确认：全部 $x_i$ 是否已换成 $a_i+th_i$？$g''(0)$ 是否除以了 $2$？

<!-- bilingual-en:start -->
This second example checks both complete substitution and the factor one half. Every coordinate becomes its path expression, and the second derivative contributes half its value to the Taylor polynomial. Both routes give $3.63$.
<!-- bilingual-en:end -->

## 6. 从 Taylor 到局部极值

<!-- bilingual-en:start -->
*From Taylor expansion to local extrema*
<!-- bilingual-en:end -->

课程定位：课堂补充；[[EC400 Lecture Notes SOFP.pdf#page=19|讲义 pp.19–22]]。这部分连接第二讲，但本次已经学习。

### 6.1 局部、全局、严格各在限制什么

<!-- bilingual-en:start -->
*Local, global, and strict impose different requirements*
<!-- bilingual-en:end -->

[[局部极值与绝对极值|局部最大点]] $x^*$ 要求存在某个 $\varepsilon>0$，只要可行点满足 $\|x-x^*\|<\varepsilon$，就有 $f(x)\le f(x^*)$。全局最大要求每个可行点都满足这个不等式，不限制距离。严格最大把除 $x=x^*$ 外的 $\le$ 改成 $<$；最小则把不等号反向。

$\varepsilon$ 必须是能同时约束附近所有方向的半径，不是每个方向各选一个半径。

<!-- bilingual-en:start -->
A [[局部极值与绝对极值|local maximum]] dominates nearby feasible points, while a global maximum dominates all feasible points. Strictness excludes equal values at distinct points. The local radius must work simultaneously for every nearby direction.
<!-- bilingual-en:end -->

### 6.2 二阶检验的充分条件和必要条件

<!-- bilingual-en:start -->
*Sufficient and necessary second-order conditions*
<!-- bilingual-en:end -->

设 $f$ 在内点 $x^*$ 的邻域为 $C^2$，且 $Df(x^*)=0$，于是
$$
f(x^*+h)-f(x^*)=\tfrac12h^TH(x^*)h+o(\|h\|^2).
$$

| 驻点处的 Hessian | 能推出的结论 |
|---|---|
| [[Hessian 局部极小判据\|正定]] | 严格局部极小 |
| [[Hessian 局部极大判据\|负定]] | 严格局部极大 |
| [[Hessian 鞍点判据\|不定]] | 鞍点，任意近处都有更高和更低值 |
| [[半定Hessian无结论\|半定但非定]] | 一般不能单靠该 Hessian 分类 |

这些定号条件是充分条件。$-x^4$ 在 $0$ 严格极大，却有 $f''(0)=0$，所以不能反过来写成“严格极大当且仅当 Hessian 负定”。必要条件只要求：内点局部最大处 Hessian 半负定；内点局部最小处半正定。

<!-- bilingual-en:start -->
At an interior $C^2$ stationary point, a definite Hessian gives a strict local extremum and an indefinite Hessian gives a saddle. These are sufficient tests. The example $-x^4$ at zero disproves necessity of negative definiteness for a strict maximum. Necessary second-order conditions require only the corresponding semidefinite sign.
<!-- bilingual-en:end -->

不能仅凭“余项小”就忽略它。若 $H$ 正定，[[正定谱下界]]给出统一的 $c>0$，使 $h^THh\ge c\|h\|^2$ 对所有 $h$ 成立。取邻域足够小，余项满足 $|R_2|\le(c/4)\|h\|^2$，于是
$$
f(x^*+h)-f(x^*)\ge\frac c2\|h\|^2-\frac c4\|h\|^2
=\frac c4\|h\|^2>0\quad(h\ne0).
$$
这才证明严格局部极小。负定时对 $-f$ 证明；不定时固定一个正方向、一个负方向，沿它们分别缩小步长，二次项压过余项，得到相反符号。

<!-- bilingual-en:start -->
A [[正定谱下界|uniform positive-definite lower bound]] dominates the remainder in every direction at once. The displayed estimate proves strict local minimality. Apply it to $-f$ for maximality; fixed positive and negative directions establish the saddle case.
<!-- bilingual-en:end -->

### 6.3 半定无结论不是“主子式算得还不够多”

<!-- bilingual-en:start -->
*Higher-order information cannot be recovered from more minors*
<!-- bilingual-en:end -->

比较 $f_1=-x_1^2+x_2^3$ 与 $f_2=-x_1^2-x_2^4$。它们在原点都驻定，Hessian 同为 $\operatorname{diag}(-2,0)$。$f_1$ 沿 $x_2$ 的正、负方向分别取正、负值；$f_2$ 在每个非零点都为负，因此原点严格全局极大。

同一个 Hessian 对应两种不同局部形状。已正确判出半定但非定后，再多算该矩阵的主子式也不会补出三次、四次项；这些信息根本没有存放在 Hessian 里。见[[半定Hessian无结论]]。

<!-- bilingual-en:start -->
The two functions have the same singular negative-semidefinite Hessian at the origin, yet one has a saddle and the other a strict maximum. Extra minors of an already classified matrix cannot recover absent higher-order terms. See [[半定Hessian无结论|the boundary of the second-order test]].
<!-- bilingual-en:end -->

[[逐条直线极小不推局部极小|逐条检查直线也可能漏掉弯曲路径]]。取
$$
f(x,y)=(y-x^2)(y-2x^2).
$$
沿固定非零方向 $(u,v)$，
$$
f(tu,tv)=t^2(v-tu^2)(v-2tu^2).
$$
若 $v\ne0$，足够小的非零 $t$ 使两个括号同号，值为正；若 $v=0$，得到 $2t^4u^4>0$。因此每条固定直线上都有严格局部最小。可是沿 $y=\tfrac32x^2$，
$$
f(x,\tfrac32x^2)=(\tfrac12x^2)(-\tfrac12x^2)=-\tfrac14x^4<0.
$$
原点不是局部最小。各条直线允许的“小”范围依赖方向，不能自动得到一个统一邻域；正定 Hessian 的统一下界恰好排除了这个问题。

<!-- bilingual-en:start -->
[[逐条直线极小不推局部极小|A strict minimum on every fixed line need not be a local minimum]]. The displayed polynomial is positive sufficiently close to zero on each fixed line but negative along the indicated parabola. Line-dependent neighbourhoods need not combine into one neighbourhood. The positive-definite lower bound provides the missing uniformity.
<!-- bilingual-en:end -->

### 6.4 两个从求导到结论的完整例子

<!-- bilingual-en:start -->
*Two complete stationary-point calculations*
<!-- bilingual-en:end -->

对 $f=2x_1^2-4x_1x_2+5x_2^2$，
$$
Df=(4x_1-4x_2,-4x_1+10x_2)=0.
$$
第一式给 $x_1=x_2$，代入第二式得 $6x_2=0$，因此唯一驻点为 $(0,0)$。
$$
H=\begin{pmatrix}4&-4\\-4&10\end{pmatrix},\quad L_1=4,
\quad L_2=40-(-4)(-4)=40-16=24>0.
$$
故严格局部极小。进一步直接配方 $f=2(x_1-x_2)^2+3x_2^2$，证明它也是唯一全局最小点。局部结论来自 Hessian 检验，全局结论来自对整个函数的直接比较。

<!-- bilingual-en:start -->
The stationary equations yield the origin. Leading Hessian minors $4$ and $24$ give a strict local minimum. Completing the entire function into two positive squares additionally proves global uniqueness.
<!-- bilingual-en:end -->

讲义的 $f=x_1^3-x_2^3+9x_1x_2$ 给出
$$
3x_1^2+9x_2=0\Rightarrow x_2=-x_1^2/3,\qquad
-3x_2^2+9x_1=0\Rightarrow x_1=x_2^2/3.
$$
代入得 $x_1=x_1^4/27$，所以 $x_1(x_1^3-27)=0$，实数解为 $0,3$，对应驻点 $(0,0),(3,-3)$。
$$
H(x)=\begin{pmatrix}6x_1&9\\9&-6x_2\end{pmatrix}.
$$
原点处 $\det H=-81<0$，直接判不定、鞍点，不必因为 $L_1=0$ 就停下。$(3,-3)$ 处 $H=\begin{pmatrix}18&9\\9&18\end{pmatrix}$，$L_1=18,L_2=324-81=243$，是严格局部极小，函数值 $27+27-81=-27$。

然而固定 $x_1=0$ 并令 $x_2\to+\infty$，有 $f=-x_2^3\to-\infty$。这个局部最小点不是全局最小点，整个函数也没有全局最小值。

<!-- bilingual-en:start -->
The cubic example has two stationary points. A negative determinant establishes a saddle at the origin despite a zero first leading minor. Positive leading minors establish a strict local minimum at $(3,-3)$ with value $-27$. Sending the second coordinate to positive infinity along the specified line disproves global minimality and global attainment of a minimum.
<!-- bilingual-en:end -->

## 7. 凸集与凹凸函数

<!-- bilingual-en:start -->
*Convex sets and concave or convex functions*
<!-- bilingual-en:end -->

课程定位：[[EC400 Slides Lecture 1.pdf#page=44|slides 42–50]]。

### 7.1 凸集说的是线段可不可行

<!-- bilingual-en:start -->
*Convexity of a set keeps entire segments feasible*
<!-- bilingual-en:end -->

[[凸集]] $U$ 的定义是：任意 $x,y\in U$、任意 $t\in[0,1]$，都有
$$
tx+(1-t)y\in U.
$$
$x,y$ 是两个输入点，$t$ 是混合权重；两个权重非负且和为 $1$。$t=0$ 得 $y$，$t=1$ 得 $x$，$t=1/2$ 得中点；也可写成 $y+t(x-y)$，正是一条以 $y$ 为起点、$x-y$ 为方向的线段。

例如 $x=(4,0),y=(0,2)$，混合点是 $(4t,2-2t)$；当 $t=0,1/4,1/2,3/4,1$ 时，依次为 $(0,2),(1,1.5),(2,1),(3,0.5),(4,0)$。不是任意正系数加权都叫凸组合，权重之和必须是 $1$。

<!-- bilingual-en:start -->
A [[凸集|convex set]] contains the entire segment between every pair of its points. The weights are nonnegative and sum to one. The parameterized example explicitly moves from one endpoint to the other. Arbitrary positive weights without normalization do not define a convex combination.
<!-- bilingual-en:end -->

半平面 $U=\{x:x_1\ge0\}$ 是凸的：若 $x_1,y_1\ge0$，则混合点第一坐标 $tx_1+(1-t)y_1\ge0$。圆盘也凸，但圆周不凸：圆周上 $(1,0)$ 与 $(-1,0)$ 的中点是 $(0,0)$，不在圆周上。判断的是完整集合，不是只看轮廓“圆不圆”。

预算集合 $B=\{x\ge0:p^Tx\le m\}$ 也是凸的。两个可行组合的混合仍逐坐标非负，而成本为 $p^T(tx+(1-t)y)=tp^Tx+(1-t)p^Ty\le tm+(1-t)m=m$。这解释了为什么经济学中混合两个可行选择往往仍可行。

<!-- bilingual-en:start -->
Half-spaces are convex by preservation of their defining linear inequality. A disk is convex, but its circumference is not: the midpoint of opposite points leaves the circumference. A nonnegative budget set is convex because mixing preserves both nonnegativity and the linear budget bound.
<!-- bilingual-en:end -->

### 7.2 凹凸性比较的是哪两个高度

<!-- bilingual-en:start -->
*Concavity compares function values with chord heights*
<!-- bilingual-en:end -->

在凸定义域 $U$ 上，[[凹凸性|凹函数]]要求
$$
f(tx+(1-t)y)\ge tf(x)+(1-t)f(y).
$$
左边是先混合**输入**，再求函数值；右边是先算两个**输出**，再取加权平均。凹函数在弦上方，凸函数把不等号反过来，在弦下方。这里“上方、下方”都允许重合。

以 $f(z)=-z^2$、$x=2,y=0,t=1/2$ 为例，左边 $f(1)=-1$，右边 $[f(2)+f(0)]/2=-2$，所以左边较高。但检查一组点只是在读定义，不能证明整个函数凹。一般地
$$
f(tx+(1-t)y)-[tf(x)+(1-t)f(y)]
=t(1-t)(x-y)^2\ge0,
$$
这个对任意 $x,y,t$ 成立的式子才给出完整证明。

<!-- bilingual-en:start -->
For a [[凹凸性|concave function]], evaluating the mixed input gives at least the weighted average of the endpoint outputs. Convexity reverses the inequality. One numerical check illustrates the definition; the general nonnegative chord gap displayed for $-z^2$ proves it for all inputs and weights.
<!-- bilingual-en:end -->

取负号会交换凹与凸；仿射函数 $f(x)=c^Tx+d$ 在弦不等式中恒取等号，所以既凹又凸。仿射允许常数项，严格线性还要求 $d=0$。严格凹则要求 $x\ne y$、$0<t<1$ 时不等号严格；端点 $t=0,1$ 本来必为等号，不应要求严格。

凸集讨论输入能否混合，凹凸函数讨论混合后的输出高度，是两个层次。定义本身不要求可导，折线也可以是凹函数。

<!-- bilingual-en:start -->
Negation exchanges concavity and convexity. Affine functions, including a constant term, satisfy both with equality. Strict concavity requires strict inequality only for distinct endpoints and interior weights. Convexity of the domain and curvature of the function are separate conditions, and neither definition requires differentiability.
<!-- bilingual-en:end -->

### 7.3 一阶判据：凹函数在每个切平面下方

<!-- bilingual-en:start -->
*The first-order tangent inequality*
<!-- bilingual-en:end -->

若 $f$ 在开凸域 $U$ 上可微，[[凹性一阶判据]]给出
$$
f\text{ 凹}\quad\Longleftrightarrow\quad
f(y)\le f(x)+Df(x)(y-x)\quad(\forall x,y\in U).
$$
右边恰好是在 $x$ 处的一阶 Taylor 近似。普通函数只在附近近似贴着切平面；凹函数则在整个定义域都不高于这个切平面。于是“图像在弦上方”与“图像在切平面下方”并不冲突：弦连接两个图像上的点，切平面由一点的导数确定。

<!-- bilingual-en:start -->
The [[凹性一阶判据|first-order criterion]] states that on an open convex domain, a differentiable concave function lies below every tangent affine function throughout the domain. This is compatible with lying above its chords: chords join two graph points, whereas tangent planes use derivative information at one point.
<!-- bilingual-en:end -->

先证“凹 $\Rightarrow$ 切平面上界”。对 $0<t\le1$，凹性给出
$$
f(x+t(y-x))\ge(1-t)f(x)+tf(y).
$$
减 $f(x)$，再除以正数 $t$：
$$
\frac{f(x+t(y-x))-f(x)}t\ge f(y)-f(x).
$$
$t\downarrow0$ 后，左边由方向导数公式趋于 $Df(x)(y-x)$，所以上界成立。整个过程中没有除以向量 $y-x$；多元推导直接对标量参数 $t$ 取极限。

<!-- bilingual-en:start -->
Concavity along the segment gives the displayed inequality. Subtracting the baseline and dividing by the positive scalar parameter produces a directional difference quotient. Its limit yields the tangent bound without ever dividing by a vector.
<!-- bilingual-en:end -->

反过来，假设每一点的切平面都是上界。记 $z=tx+(1-t)y$，分别以 $z$ 为展开点：
$$
f(x)-f(z)\le Df(z)(x-z),\qquad
f(y)-f(z)\le Df(z)(y-z).
$$
第一式乘 $t$，第二式乘 $1-t$ 再相加。右边为
$$
Df(z)[t(x-z)+(1-t)(y-z)]
=Df(z)[tx+(1-t)y-z]=0.
$$
因此 $tf(x)+(1-t)f(y)\le f(z)$，正是凹性。两个方向都证明了，才能写“当且仅当”。

<!-- bilingual-en:start -->
For the converse, apply the tangent bound at the mixed point to both endpoints. Weighting and adding cancels the derivative term exactly because the weighted displacement is zero. The remaining inequality is concavity, establishing both directions of the equivalence.
<!-- bilingual-en:end -->

### 7.4 二阶判据：一点的曲率与处处的曲率

<!-- bilingual-en:start -->
*The second-order criterion requires curvature everywhere*
<!-- bilingual-en:end -->

在开凸域 $U$ 上，若 $f\in C^2$，[[Hessian凹凸性判据]]是
$$
f\text{ 凹}\iff H(x)\preceq0\quad\forall x\in U;
\qquad
f\text{ 凸}\iff H(x)\succeq0\quad\forall x\in U.
$$
$H\preceq0$ 表示矩阵半负定，即 $h^THh\le0$ 对所有 $h$ 成立，不是说所有矩阵元素都非正。**“每一个点”和“每一个方向”两个量词缺一不可。**

证明把多元问题限制到连接任意两点的线段：$g(t)=f(x+t(y-x))$，于是 $g''(t)=(y-x)^TH(x+t(y-x))(y-x)$。Hessian 处处半负定，便使每条线段上的 $g''\le0$，由一元[[导数判凹凸]]得到凹性。反向则从凹函数的每条局部直线限制为凹出发，得到任意点、任意方向的二阶导数非正。

<!-- bilingual-en:start -->
The [[Hessian凹凸性判据|Hessian criterion]] requires the semidefinite sign at every point of an open convex domain. Matrix order refers to all directional quadratic forms, not to entrywise signs. Restricting to a segment converts the Hessian condition into the [[导数判凹凸|one-dimensional second-derivative criterion]]; local restrictions in every direction prove the converse.
<!-- bilingual-en:end -->

开凸域是一个清楚、够用的适用条件。有边界时，常可在内部应用判据，再用连续性把定义不等式延伸到边界；若定义域只是平面中的一条线，则应检查那条线内的可行方向，不能把整个环境空间的 Hessian 半负定硬当作必要条件。

若 $H(x)$ **处处负定**，则函数严格凹；反向不成立，$-x^4$ 严格凹而零点二阶导数为零。此边界由[[严格凹与负定Hessian]]保存。只在某一个驻点负定，是局部极大检验；要得到全域凹性，需要在整片定义域检查。

<!-- bilingual-en:start -->
An open convex domain is a clean sufficient setting for the criterion. Continuity can extend inequalities to included boundary points; a lower-dimensional domain requires testing feasible directions. [[严格凹与负定Hessian|An everywhere negative-definite Hessian implies strict concavity, but the converse fails]]. A sign check at one stationary point supplies only a local extremum test.
<!-- bilingual-en:end -->

例如 $f=-x_1^2-2x_2^2+x_1x_2$，
$$
H=\begin{pmatrix}-2&1\\1&-4\end{pmatrix},\quad L_1=-2,\quad\det H=8-1=7.
$$
这个 Hessian 不随点变化，且负定，所以 $f$ 在 $\mathbb R^2$ 严格凹。相反，$f=x_1^3$ 的 Hessian 为 $\operatorname{diag}(6x_1,0)$：在 $x_1>0$ 的半平面半正定，在 $x_1<0$ 半平面半负定，在整个平面既不凸也不凹。同一个代数式，换定义域后判断可能改变。

<!-- bilingual-en:start -->
The constant negative-definite Hessian proves strict concavity of the quadratic on the entire plane. For the cubic, the Hessian changes sign across the vertical axis: it is convex on the positive half-plane, concave on the negative half-plane, and neither on the whole plane. The domain is part of the claim.
<!-- bilingual-en:end -->

### 7.5 凹性如何把一阶条件变成全局结论

<!-- bilingual-en:start -->
*How concavity turns stationarity into global optimality*
<!-- bilingual-en:end -->

若 $f$ 凹、可微，且 $Df(x^*)=0$，对任意可行 $y$，一阶判据给出
$$
f(y)\le f(x^*)+Df(x^*)(y-x^*)=f(x^*).
$$
因此 $x^*$ 是全局最大点。这项结论独立保存在[[凹函数驻点全局最优]]。更一般地，只要 $Df(x^*)(y-x^*)\le0$ 对所有可行 $y$ 成立，就已经足够；这也能容纳某些边界最优点。

即使不可导，[[凸优化全局性|凹函数在凸集上的局部最大点仍是全局最大点]]：若远处存在更好的点 $y$，往它走一点点的凸组合仍可行，凹性会让这些任意接近 $x^*$ 的点也更好，与局部最大矛盾。

<!-- bilingual-en:start -->
At a stationary point, the global tangent upper bound reduces directly to the function value there, proving [[凹函数驻点全局最优|global maximality]]. A nonpositive directional derivative toward every feasible point is also sufficient. Without differentiability, [[凸优化全局性|local maximality of a concave function on a convex set is still global]]: moving slightly toward a better point would contradict local maximality.
<!-- bilingual-en:end -->

严格凹使最大点**至多一个**，但不负责保证最大点存在。若有两个不同最大点，严格凹会让它们中间的值更大，矛盾；而 $\log x$ 在 $(0,\infty)$ 严格凹，却向上无界。见[[严格凹不保证解存在]]。存在性、候选点、局部分类、全局性是四个不同问题。

<!-- bilingual-en:start -->
Strict concavity gives at most one maximizer, since mixing two distinct maximizers would improve their value. It does not guarantee attainment: the logarithm on the positive half-line is strictly concave but unbounded above. [[严格凹不保证解存在|Uniqueness and existence must be separated]].
<!-- bilingual-en:end -->

### 7.6 非负加权与利润函数

<!-- bilingual-en:start -->
*Nonnegative sums and the profit function*
<!-- bilingual-en:end -->

[[凹函数非负加权和|同一凸域上的凹函数作非负加权，仍凹]]。若 $a_i\ge0$，$F=\sum_i a_if_i$，对每个 $f_i$ 用凹性后乘 $a_i$，不等号方向不变，求和得
$$
F(tx+(1-t)y)\ge\sum_i a_i[tf_i(x)+(1-t)f_i(y)]
=tF(x)+(1-t)F(y).
$$
负权重不能照搬；例如 $-x^2$ 凹，乘 $-1$ 后成为凸函数 $x^2$。课件要求正权重，这里也允许零权重，因为零项不影响不等式。

<!-- bilingual-en:start -->
[[凹函数非负加权和|Nonnegative weighted sums preserve concavity]] on a common convex domain. Multiply each defining inequality by its nonnegative weight and add. Negative weights may reverse the curvature; zero weights cause no problem.
<!-- bilingual-en:end -->

slides 50 的利润函数为
$$
\Pi(x)=p\,g(x)-\sum_{i=1}^n w_ix_i.
$$
$x$ 是投入组合，$g(x)$ 是产量，$p\ge0$ 是产出价格，$w_i$ 是单位投入成本。若 $g$ 在凸投入域上凹，$pg$ 凹，负成本是仿射函数、也凹，所以 $\Pi$ 凹。若内点最优且可微，一阶条件为
$$
p\frac{\partial g}{\partial x_i}=w_i.
$$
左边是多用一单位投入带来的边际收入，右边是边际成本。凹性使符合条件的一阶解具有全局含义。若有非负投入等边界约束，仍须检查可行性与边界条件，不能强迫每个最优解都满足无约束的零梯度。

这种“先写目标与约束，再用边际条件比较”的结构也出现在 [[02_DPDE Lecture 1 - 消费储蓄与欧拉方程|DPDE 第一讲]]；静态与跨期问题会通过[[无约束一阶条件]]、[[凹凸性]]和[[多元链式法则]]这些共同对象连接起来。

<!-- bilingual-en:start -->
With a nonnegative output price, concavity of production and affine input costs imply concavity of profits. At a differentiable interior optimum, marginal revenue from each input equals its marginal cost. Boundary constraints still require feasible optimality conditions. The same objective-and-constraint reasoning connects to [[02_DPDE Lecture 1 - 消费储蓄与欧拉方程|DPDE Lecture 1]] through shared atoms for [[无约束一阶条件|first-order conditions]], [[凹凸性|curvature]], and the [[多元链式法则|chain rule]].
<!-- bilingual-en:end -->

## 8. 拟凹性：关注达到同一水平的那些点

<!-- bilingual-en:start -->
*Quasiconcavity studies sets of inputs meeting a threshold*
<!-- bilingual-en:end -->

课程定位：[[EC400 Slides Lecture 1.pdf#page=54|slides 52–65]]。

### 8.1 水平集、上轮廓集、下轮廓集

<!-- bilingual-en:start -->
*Level, upper contour, and lower contour sets*
<!-- bilingual-en:end -->

先固定实数门槛 $k$，再问哪些输入满足要求。对 $f:U\to\mathbb R$：
$$
\begin{aligned}
C_k&=\{x\in U:f(x)=k\}&&\text{水平集},\\
C_k^+&=\{x\in U:f(x)\ge k\}&&\text{上轮廓集},\\
C_k^-&=\{x\in U:f(x)\le k\}&&\text{下轮廓集}.
\end{aligned}
$$
[[水平集]]回答“恰好等于”；[[上轮廓集]]回答“至少达到”；[[下轮廓集]]回答“至多达到”。花括号里装的是输入 $x$，不是函数值，也不是三维曲面上的点。上标 $+$、$-$ 标记不等号方向，不表示集合里的坐标必须正或负。

下面补全手写第 8 页要求加入的定义和图例。二维图中可以把水平集画成线，再给满足不等式的区域上色；但一般水平集也可能是点、区域或更复杂的集合，不能总叫“曲线”。

<!-- bilingual-en:start -->
A [[水平集|level set]] collects inputs exactly at a threshold, an [[上轮廓集|upper contour set]] those at least at that threshold, and a [[下轮廓集|lower contour set]] those at most at it. These are sets of inputs. The superscripts describe the inequality, not the signs of coordinates. In two-dimensional examples they can often be drawn as a boundary and a shaded region, but level sets need not always be curves.
<!-- bilingual-en:end -->

![[SOFP-L1-contour-sets.png|850]]

图示用 $f(x_1,x_2)=x_1+x_2$、$k=2$，只画第一象限的窗口。先代入数值，再判断：$(0,0)$ 的值为 $0$，不在 $C_2^+$；$(1,1)$ 的值为 $2$，在水平集也在上、下轮廓集中；$(3,0)$ 的值为 $3$，在上轮廓集、不在水平集；$(0.5,0.5)$ 的值为 $1$，在下轮廓集、不在上轮廓集。

$C_2$ 是直线 $x_1+x_2=2$；$C_2^+$ 是这条线及较大和数的一侧；$C_2^-$ 是线及另一侧。读“上轮廓”时，先看函数值大的一侧，不要机械理解为纸面上方。比如函数 $-x_1-x_2$ 的较大值恰好在相反一侧。

<!-- bilingual-en:start -->
For the sum function at threshold two, evaluate each input before deciding membership. Equality puts a point in the level set and in both contour sets. The upper set is the side with larger function values, which need not mean visually upward on the page. Negating the function reverses the relevant side.
<!-- bilingual-en:end -->

### 8.2 拟凹与拟凸：定义中的每个门槛都要检查

<!-- bilingual-en:start -->
*Quasiconcavity and quasiconvexity quantify over all thresholds*
<!-- bilingual-en:end -->

在凸定义域 $U$ 上，[[拟凹函数|拟凹（quasiconcave）]]要求每个 $k\in\mathbb R$ 的 $C_k^+$ 都是凸集；[[拟凸函数|拟凸（quasiconvex）]]要求每个 $C_k^-$ 都凸。拟凹把达到某个标准的点放在一起，要求两点达标时，它们之间整条线段也达标。

“每个”很重要：找到一个凸上轮廓集，不能证明拟凹；找到一个不凸上轮廓集，就足以否定拟凹。门槛超出值域时，上轮廓集可能是空集或整个定义域，仍要按定义理解；空集是凸的，要求中的“任意两个集合内的点”在这里没有反例。

<!-- bilingual-en:start -->
On a convex domain, a [[拟凹函数|quasiconcave function]] has convex upper contour sets at every threshold, while a [[拟凸函数|quasiconvex function]] has convex lower contour sets. One convex contour set is insufficient for a positive conclusion, but one nonconvex contour set disproves it. Thresholds outside the range produce empty or whole-domain sets, which cause no exception to the definition.
<!-- bilingual-en:end -->

取 $f(x_1,x_2)=x_1x_2$，明确限制在 $U=(0,\infty)^2$。门槛为 $1$ 时，
$$
C_1^+=\{x_1>0,x_2>0:x_2\ge1/x_1\}.
$$
$(1,1)$ 在边界；$(2,2)$ 与 $(0.5,3)$ 达标；$(0.5,1)$ 与 $(3,0.2)$ 不达标。对于一般 $k>0$，边界 $x_2=k/x_1$ 的二阶导数为 $2k/x_1^3>0$，是凸曲线；其上方区域为凸集。也可以直接验证：若两端在曲线上方，$k/x_1$ 的凸性使混合点仍在曲线上方。$k\le0$ 时，上轮廓集是整个正象限。因此 $f$ 拟凹。

但 $H=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ 不定，所以 $f$ 不凹。换到整个 $\mathbb R^2$ 后它甚至不再拟凹：$(1,1)$ 与 $(-1,-1)$ 都在 $C_1^+$，中点 $(0,0)$ 却不在。**函数和定义域必须一起说。**

<!-- bilingual-en:start -->
The product function is quasiconcave on the strictly positive quadrant. Positive upper thresholds give the region above the convex curve $k/x_1$, while nonpositive thresholds give the whole domain. Its indefinite Hessian shows that it is not concave. On the entire plane it is not even quasiconcave, because two opposite points meeting threshold one have a midpoint below that threshold. The domain is essential.
<!-- bilingual-en:end -->

![[SOFP-L1-quasi-examples.png|850]]

### 8.3 为什么凹性一定推出拟凹性

<!-- bilingual-en:start -->
*Why concavity implies quasiconcavity*
<!-- bilingual-en:end -->

[[凹性推出拟凹性]]可以直接从定义证明。取 $x,y\in C_k^+$，所以 $f(x)\ge k,f(y)\ge k$。凹性给出
$$
f(tx+(1-t)y)\ge tf(x)+(1-t)f(y)\ge tk+(1-t)k=k.
$$
因此混合点仍在 $C_k^+$，这个集合凸。$k$ 任意，故函数拟凹。凸推出拟凸的证明把不等号反向即可。

凹性要求混合后的函数值至少达到两端的**加权平均**；拟凹性只要求至少达到两端中**较低的值**。前者更强，反向不能保证，正象限上的乘积函数已经给出反例。

<!-- bilingual-en:start -->
[[凹性推出拟凹性|Concavity implies quasiconcavity]] because mixing two inputs above a common threshold keeps their weighted output average above it, and concavity places the actual output no lower. The analogous reversed inequalities give convexity implying quasiconvexity. Concavity controls the weighted average; quasiconcavity only controls the lower endpoint value.
<!-- bilingual-en:end -->

### 8.4 经济含义：喜欢混合究竟有多强

<!-- bilingual-en:start -->
*The precise meaning of preferring mixtures*
<!-- bilingual-en:end -->

如果两个消费组合 $x,y$ 的效用相同，$u(x)=u(y)=k$，拟凹性保证任意混合的效用不低于 $k$，所以混合不差于两端。若两端效用不同，就只能保证不差于较差的那个，不能说总比两个都好。这一关系保存在[[拟凹效用与混合偏好]]。

例如 $u(z)=z$，它甚至同时凹、凸、拟凹、拟凸；取 $x=0,y=10$，中点效用 $5$ 低于较好的端点 $10$。所以 slide 56 中“任意两个组合的混合优于任一个”的文字必须结合图中的同效用前提来读。“不差”允许无差异，并不表示严格偏好。

<!-- bilingual-en:start -->
With equal endpoint utility, quasiconcavity makes every mixture weakly at least as good as both endpoints. With unequal endpoint utility, it guarantees only the lower of the two values. [[拟凹效用与混合偏好|This distinction]] matters: a linear utility between zero and ten gives five at the midpoint, below the better endpoint. The lecture's mixture illustration uses an equal-utility premise; weak preference also allows indifference.
<!-- bilingual-en:end -->

平滑无差异曲线还能把这种形状联系到边际替代率。若两种商品的边际效用 $u_1,u_2$ 为正，在曲线 $u(x_1,x_2)=k$ 上求导：
$$
u_1\,dx_1+u_2\,dx_2=0,
\qquad\frac{dx_2}{dx_1}=-\frac{u_1}{u_2}.
$$
[[边际替代率|$MRS_{12}=u_1/u_2$]]表示效用保持不变时，增加一单位商品 $1$ 愿意放弃多少商品 $2$。在能写成 $x_2=\phi(x_1)$ 的平滑曲线段上，凸上轮廓区域使边界函数 $\phi$ 凸，故斜率 $\phi'$ 非减；斜率为负，因此 $MRS_{12}=-\phi'$ 非增。这就是[[拟凹效用的递减MRS|递减 MRS]]的精确方向。

这些话需要平滑性、正边际效用与明确的比率约定；不能把“拟凹”无条件等同于任意函数都存在递减 MRS。比如 Leontief 效用有折点，拟凹定义仍适用，但折点处这个导数比值未必存在。

<!-- bilingual-en:start -->
For a smooth indifference curve with positive marginal utilities, implicit differentiation gives the slope and the [[边际替代率|marginal rate of substitution]]. On a regular decreasing graph segment, a convex upper contour region makes the boundary convex; its slope increases while its negative, the MRS, decreases. [[拟凹效用的递减MRS|This implication]] needs the stated regularity and sign assumptions. Quasiconcavity itself still applies at nondifferentiable corners where the derivative ratio is unavailable.
<!-- bilingual-en:end -->

### 8.5 一维结论为什么不能照搬到二维

<!-- bilingual-en:start -->
*Why one-dimensional shape rules fail in higher dimensions*
<!-- bilingual-en:end -->

[[一维单调函数的拟凹拟凸性|区间上的单调函数既拟凹又拟凸]]。因为两个输入之间的函数值夹在两端函数值之间，上、下轮廓集都是区间或半区间，不会中间断开。[[一维单峰函数拟凹性|一维单峰]]在这里指先非减、后非增，允许平顶；某个水平以上的部分仍是一段区间，因此拟凹。“只有一个全局最大点”本身不是这一定义，低处若有起伏，仍可能让某个上轮廓集断开。

<!-- bilingual-en:start -->
[[一维单调函数的拟凹拟凸性|A monotone function on an interval is both quasiconcave and quasiconvex]], because intermediate values remain between endpoint values. [[一维单峰函数拟凹性|A unimodal function]] means nondecreasing up to a peak and nonincreasing afterward, allowing a plateau. Merely having a unique global maximizer does not impose this shape at lower levels.
<!-- bilingual-en:end -->

[[多元单调不推拟凹|二维逐坐标递增却可能不拟凹]]。课件取 $f(x_1,x_2)=x_1^2+x_2^2$，定义域为非负象限。$(1,0)$、$(0,1)$ 的函数值都是 $1$，中点 $(1/2,1/2)$ 的值却是 $1/2$，所以 $C_1^+$ 不凸。即使把定义域换成严格正象限，也可取 $(2,1),(1,2)$：两端值为 $5$，中点 $(1.5,1.5)$ 的值为 $4.5$，仍不拟凹。

原因是沿一条连接两个点的线段，一个坐标可能增加、另一个减少。“每个坐标单独增加都会使函数上升”，并没有控制这种互相替换的路径。

<!-- bilingual-en:start -->
[[多元单调不推拟凹|Coordinatewise monotonicity does not imply multivariable quasiconcavity]]. The sum of squares gives endpoint values above their midpoint value, both on the nonnegative and strictly positive quadrants. A segment can increase one coordinate while decreasing another, so separate monotonicity does not control the tradeoff.
<!-- bilingual-en:end -->

[[唯一峰值不推拟凹|二维有唯一峰值也不够]]。令
$$
f(x_1,x_2)=-\sqrt{|x_1-1|}-\sqrt{|x_2-1|}.
$$
每项非正，只有 $(1,1)$ 取 $0$，所以这是唯一全局最大点。可是 $f(1,0)=f(0,1)=-1$，而
$$
f(1/2,1/2)=-2\sqrt{1/2}=-\sqrt2<-1.
$$
因此 $C_{-1}^+$ 不凸，函数不拟凹。峰在哪里，与整个山体各个高度的截面是否凸，是不同问题。

<!-- bilingual-en:start -->
[[唯一峰值不推拟凹|A unique peak does not ensure quasiconcavity]]. The function has a unique maximum of zero at $(1,1)$, but two points at level minus one have a midpoint below that level. Locating the peak does not determine the geometry of every upper contour set.
<!-- bilingual-en:end -->

### 8.6 递增变换保留拟凹性

<!-- bilingual-en:start -->
*Increasing transformations preserve quasiconcavity*
<!-- bilingual-en:end -->

[[单调变换保持拟凹性|若 $f$ 拟凹，$g$ 在 $f$ 的值域上非减，则 $g\circ f$ 仍拟凹]]。最稳妥的证明使用下一小节的线段判据：
$$
f(tx+(1-t)y)\ge\min\{f(x),f(y)\}.
$$
两边应用非减的 $g$：
$$
g(f(tx+(1-t)y))\ge g(\min\{f(x),f(y)\})
=\min\{g(f(x)),g(f(y))\}.
$$
所以复合函数也符合拟凹判据。不需要 $g$ 可导。若还要求效用所代表的排序完全不变，则应取严格递增 $g$；仅非减的变换可能把原先不同的效用压成相同数值。

<!-- bilingual-en:start -->
[[单调变换保持拟凹性|A nondecreasing transformation preserves quasiconcavity]], as the segment inequality shows directly, without differentiability of the transformation. To preserve the full preference ordering rather than merely quasiconcavity, the transformation should be strictly increasing; a weakly increasing one can collapse distinct values into a tie.
<!-- bilingual-en:end -->

例如正象限上的 $v=\log x_1+\log x_2$ 凹，因此拟凹；应用严格递增的 $g(z)=e^z$，得到 $e^v=x_1x_2$，仍拟凹，却不凹。由此也看见：凹性一般不受任意递增变换保持。

课件用 $C_k^+(f)=C_{g(k)}^+(g\circ f)$ 解释这个性质；这个集合等式需要严格递增，不能原样套给有平台的 $g$。线段证明同时覆盖非减变换，并避免遗漏不在 $g$ 值域内的门槛。

<!-- bilingual-en:start -->
Exponentiating the concave sum of logarithms gives the product function, which remains quasiconcave but is not concave. The lecture's matching-threshold set equality requires strict increase; it is not generally valid for transformations with flat portions. The segment proof covers nondecreasing transformations and all thresholds without that gap.
<!-- bilingual-en:end -->

### 8.7 Leontief 例子：折点不妨碍拟凹

<!-- bilingual-en:start -->
*The Leontief example works without differentiability*
<!-- bilingual-en:end -->

slides 63 取 $f(x_1,x_2)=\min\{x_1,x_2\}$。要让较小的那个至少等于 $k$，两个分量都必须至少为 $k$：
$$
C_k^+=\{x_1\ge k,\ x_2\ge k\}.
$$
它是两个半平面的交集，因此凸。直接验证也很简单：两端的每个坐标都至少为 $k$，混合后的每个坐标也至少为 $k$。函数在 $x_1=x_2$ 处有折点，仍然拟凹；事实上它还凹，可由“两个仿射函数的逐点最小值为凹函数”或直接代定义证明。这里用轮廓集就已经够用。

<!-- bilingual-en:start -->
For the Leontief function, the minimum of two coordinates exceeds a threshold precisely when both coordinates do. The resulting upper set is an intersection of two half-spaces and is convex. Nondifferentiability on the diagonal does not affect the definition. The function is in fact also concave, though the contour-set argument already proves the required quasiconcavity.
<!-- bilingual-en:end -->

### 8.8 线段判据：最后把几何定义变成可算的不等式

<!-- bilingual-en:start -->
*The segment criterion makes the set definition calculable*
<!-- bilingual-en:end -->

[[拟凹性线段判据]]说：在凸定义域上，
$$
\boxed{f\text{ 拟凹}\iff f(tx+(1-t)y)\ge\min\{f(x),f(y)\}}
$$
对所有 $x,y$ 和 $t\in[0,1]$ 成立。**从集合到不等式：** 令 $k=\min\{f(x),f(y)\}$，两端都在 $C_k^+$；该集合凸，所以整个线段都在里面，混合点函数值至少为 $k$。

**从不等式到集合：** 固定任意门槛 $k$，取任意 $x,y\in C_k^+$。两端值都至少为 $k$，所以
$$
f(tx+(1-t)y)\ge\min\{f(x),f(y)\}\ge k.
$$
于是混合点也在 $C_k^+$。门槛任意，拟凹性成立。拟凸的对应判据为 $f(tx+(1-t)y)\le\max\{f(x),f(y)\}$，见[[拟凸性线段判据]]。

<!-- bilingual-en:start -->
The [[拟凹性线段判据|segment criterion for quasiconcavity]] is equivalent to convexity of every upper contour set. In one direction choose the threshold to be the smaller endpoint value. In the other, choose any threshold and endpoints in its upper set. The [[拟凸性线段判据|quasiconvex counterpart]] bounds the mixture by the larger endpoint value.
<!-- bilingual-en:end -->

拟凹性比凹性弱，也不能直接继承“驻点一定全局最优”。[[拟凹驻点不保证最优|反例是 $f(x)=x^3$]]：它单调，因而拟凹又拟凸；$f'(0)=0$，零点却既不最大也不最小。第二讲使用拟凹性处理约束优化时，还需要相应定理的其他条件。

做一道形状判断题，可以依次问：定义域是什么？要检查上轮廓还是下轮廓？要证明所有门槛都成立，还是找一个反例？若使用导数，所需可微性和全域条件是否满足？这些问题比把所有函数都往 Hessian 判据里塞更可靠。

<!-- bilingual-en:start -->
[[拟凹驻点不保证最优|Quasiconcavity does not make every stationary point optimal]]. The monotone function $x^3$ has both quasi-properties and a zero derivative at zero, but no extremum there. Applying quasiconcavity to constrained optimization in the next lecture therefore needs additional theorem assumptions. Start each shape question with its domain and the appropriate contour set or inequality.
<!-- bilingual-en:end -->

## 9. 留给复习的五个检查点

<!-- bilingual-en:start -->
*Five focused checks for later review*
<!-- bilingual-en:end -->

这些题对应本讲几处关键连接；先独立写出理由，再展开答案。它们不要求重复抄整份讲义。

1. 把 $3x_1^2-4x_1x_2+2x_2^2$ 写成对称矩阵形式，用配方和顺序主子式各判一次。
2. 对 $f=x_1^2x_2+x_2^3$，取 $a=(1,2),h=(1,-1)$，分别用直接展开与 Hessian 算 $g''(0)$。
3. 对 $G=x_1^2+2x_2^2$，取 $a=(1,1),h=(0.1,0.1)$，写完整二阶近似并解释为什么误差为零。
4. $f=x_1x_2$ 在正象限上为什么拟凹却不凹？域改成整个平面后会怎样？
5. “Hessian 在驻点半负定，所以该点局部最大”缺了什么？给两个同 Hessian、不同结论的函数。

<!-- bilingual-en:start -->
Try these checks independently before opening the answers: recover and classify a quadratic form by two methods; compute a directional second derivative by substitution and by its Hessian; carry out an exact quadratic Taylor expansion; explain why the product's quasi-property depends on its domain; and distinguish a singular semidefinite Hessian from a sufficient maximum test.
<!-- bilingual-en:end -->

> [!example]- 参考解与自查
> 1. $A=\begin{pmatrix}3&-2\\-2&2\end{pmatrix}$，$L_1=3,L_2=6-4=2>0$；$Q=3(x_1-2x_2/3)^2+(2/3)x_2^2$，正定。
> 2. $H(a)=\begin{pmatrix}4&2\\2&12\end{pmatrix}$，$Hh=(2,-10)^T$，$h^THh=12$。直接展开 $g(t)=(1+t)^2(2-t)+(2-t)^3=10-9t+6t^2-2t^3$，所以 $g''(0)=12$。
> 3. $P_2=3+0.6+\tfrac12(0.06)=3.63$；原函数为二次多项式，二阶展开已包含全部项。
> 4. 正象限上正水平的上轮廓集是 $x_2\ge k/x_1$；非正水平给整个域。Hessian 不定，故不凹。整个平面上用 $(1,1),(-1,-1)$ 及其中点即可否定拟凹。
> 5. 半负定但非负定保留零方向，高阶项未受该点 Hessian 控制。$-x_1^2+x_2^3$ 与 $-x_1^2-x_2^4$ 在原点 Hessian 相同，前者鞍点，后者严格最大。
>
> <!-- bilingual-en:start -->
> &nbsp;
> **1.** The leading minors are $3$ and $2$, and completing the square gives two positive weights.<br>
> **2.** Both the Hessian calculation and the full cubic expansion give a second derivative of $12$.<br>
> **3.** The correct value is $3.63$; a quadratic polynomial has no remainder beyond second order.<br>
> **4.** The positive quadrant gives convex upper sets, whereas the full plane admits opposite endpoints whose midpoint fails the threshold.<br>
> **5.** The same singular negative-semidefinite Hessian can coexist with a saddle or a strict maximum because higher-order terms differ.
> <!-- bilingual-en:end -->

## 10. 来源定位与已核对的表述

<!-- bilingual-en:start -->
*Source locations and checked wording*
<!-- bilingual-en:end -->

| 正文范围 | 课程位置 | 原子入口 |
|---|---|---|
| 二次型与定号 | [[EC400 Slides Lecture 1.pdf#page=8\|slides 8–14]]；讲义 pp.2–3 | [[二次型]]、[[二次型对称化]]、[[正定矩阵]] |
| 行列式、主子式和配方 | [[EC400 Slides Lecture 1.pdf#page=16\|slides 16–31]]；讲义 pp.4–9 | [[行列式]]、[[主子式]]、[[二维二次型配方]] |
| 微分补充与 Taylor | [[EC400 Slides Lecture 1.pdf#page=36\|slides 34–39]]；课堂记录补课线，手写 pp.1–4、6–7 | [[多元链式法则]]、[[Hessian方向二阶导数]]、[[多元Taylor近似]] |
| 局部极值 | [[EC400 Lecture Notes SOFP.pdf#page=19\|讲义 pp.19–22]]；课堂补充 | [[无约束一阶条件]]、[[半定Hessian无结论]] |
| 凸集与凹性 | [[EC400 Slides Lecture 1.pdf#page=44\|slides 42–50]]；讲义 pp.12–14 | [[凸集]]、[[凹性一阶判据]]、[[Hessian凹凸性判据]] |
| 拟凹性 | [[EC400 Slides Lecture 1.pdf#page=54\|slides 52–65]]；讲义 pp.15–18 | [[上轮廓集]]、[[拟凹函数]]、[[拟凹性线段判据]] |

课程原件决定顺序、符号与题目；课堂记录和手写用于保留问题、解释路径及例子。微分推导、配方运算、数值核对和图示在正文中重新展开。补充核验用了 [MIT 18.S096 的 Hessian 讲义](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec12.pdf)检查导数行列约定及二阶 Taylor 项；[Boyd–Vandenberghe §§3.1.3–3.1.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=83)用于核对开凸域条件和严格凹的二阶条件边界。

<!-- bilingual-en:start -->
The course materials establish sequence, notation, and exercises; the classroom record and handwriting preserve the explanatory route. Calculations, derivations, and figures are reconstructed in the text. The linked MIT notes cross-check Hessian notation and the quadratic Taylor term; the cited textbook sections check the open-domain assumptions and the boundary of the strict second-order curvature implication.
<!-- bilingual-en:end -->

以下差异已在正文采用正确表述，原始材料保留原样：

- 讲义 p.3 的负定例子漏了平方；应为 $-(x_1^2+x_2^2)$，与 slide 14 一致。
- 讲义 p.2 的等高线清单省略了单点；还须考虑单直线和零二次型等退化情形。
- 半定统一采用含定的宽义定义；零矩阵同时半正定与半负定。
- 手写 p.2 的 $g'(0)$ 应由导数差商解释；p.3 的 $Df(a)$ 要乘方向 $h$ 才得到 $g'(0)$；p.4 的二阶展开要保留两个混合项及最后的 $f_{22}h_2^2$。
- 课堂记录把某处“严格局部极大”与“Hessian 负定”写成等价；正确的是后者充分而非必要。另一处把 $g(t)=6t^2$ 与 $g''(0)=6$ 同时记为正确，两者不相容：前者会给出 $g''(0)=12$；该题原始函数未完整记录，不能据此判断究竟哪一项抄错。
- 手写 p.7 的“阶数越高，近似区间越宽”不是一般定理；Taylor 的基本保证是局部误差阶。
- slide 56 的任意混合表述需要区分同效用和不同效用两种情况；讲义 p.18 唯一峰值例的符号应为其他点 $f(x)<0=f(1,1)$，不是反过来。
- slide 62 的轮廓集等式按严格递增解释；非减变换保持拟凹性的更一般结论用线段判据证明。

<!-- bilingual-en:start -->
The original records remain intact while the exposition corrects the missing squares, degenerate level-set cases, inclusive semidefinite terminology, handwritten derivative notation, and the distinction between sufficient and necessary Hessian tests. One recorded exercise inconsistently pairs $g(t)=6t^2$ with a second derivative of $6$; without the full original problem, which entry was transcribed incorrectly cannot be determined. The text also corrects the unrestricted claim about higher-order approximation ranges, the mixture premise, the peak example's reversed sign, and the distinction between strict and weakly increasing transformations.
<!-- bilingual-en:end -->
