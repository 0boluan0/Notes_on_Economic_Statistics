---
aliases:
  - MIT 18.06SC Unit III
  - 正定矩阵及其应用
tags:
  - 线性代数
  - mit-ocw
  - course-note
---

# Positive Definite Matrices and Applications

> [!info] 课程来源与范围
> 本笔记依照 MIT OCW 18.06SC *Linear Algebra*（Fall 2011）Unit III 的官方顺序整理：Session 3.1–3.8 → Session 3.9 Exam 3 Review → Unit 3 Exam。Lecture 33 的 left/right inverse 与 pseudoinverse 已按课程知识主线放入 Session 3.8；Lecture 32 是 Exam 3 review，而不是伪逆课。

## 0. 本单元要解决什么

前两单元已经回答了“怎样解 $Ax=b$”和“怎样利用正交、行列式与特征值理解方阵”。Unit III 进一步追问：

1. 哪些矩阵拥有最稳定、最清晰的谱结构？答案是对称矩阵（symmetric matrix）与正定矩阵（positive definite matrix）。
2. 一个方阵不能对角化时，怎样准确描述缺失的特征向量？答案是相似变换与 Jordan 形。
3. 一个矩阵非方阵、秩亏、甚至没有特征分解时，怎样找到最合适的输入—输出坐标？答案是奇异值分解（singular value decomposition, SVD）。
4. 为什么同一个线性映射会有许多不同矩阵？答案是基、坐标与换基。
5. 当真正的逆不存在时，怎样保留“能逆的那一部分”？答案是左右逆与 Moore–Penrose 伪逆。

本单元的统一图景是

$$
\text{对称/正定}\longrightarrow\text{正交谱分解},\qquad
\text{一般方阵}\longrightarrow\text{相似/Jordan},
$$

$$
\text{任意 }m\times n\text{ 矩阵}\longrightarrow\text{SVD}\longrightarrow\text{换基、压缩、最小二乘与伪逆}.
$$

## 1. 全局约定与阅读导航

### 1.1 域、尺寸和共轭

- 未特别说明时，矩阵在实数域 $\mathbb R$ 上；涉及复特征值、Fourier 矩阵和 Hermitian 矩阵时，底层域改为 $\mathbb C$。
- $A\in\mathbb F^{m\times n}$ 表示 $A$ 有 $m$ 行、$n$ 列，并定义线性映射 $x\in\mathbb F^n\mapsto Ax\in\mathbb F^m$。
- 实矩阵的转置是 $A^T$；复矩阵的共轭转置是
  $$
  A^*=\overline{A}^{\,T}.
  $$
- 复向量内积约定为 $\langle x,y\rangle=x^*y$，因此 $\|x\|^2=x^*x>0$（只要 $x\ne0$）。不能把它误写成 $x^Tx$。
- “正定”在本课程中默认指实对称正定，或复 Hermitian 正定。若矩阵不对称，二次型只看它的对称部分：
  $$
  x^TAx=x^T\frac{A+A^T}{2}x.
  $$

### 1.2 Session 导航

- [[#Session 3.1 Symmetric matrices and positive definiteness|3.1 对称矩阵与正定性]]
- [[#Session 3.2 Complex matrices and fast Fourier transform|3.2 复矩阵与 FFT]]
- [[#Session 3.3 Positive definite matrices and minima|3.3 二次型、正定矩阵与极小值]]
- [[#Session 3.4 Similar matrices and Jordan form|3.4 相似矩阵与 Jordan 形]]
- [[#Session 3.5 Singular value decomposition|3.5 奇异值分解]]
- [[#Session 3.6 Linear transformations and their matrices|3.6 线性变换及其矩阵]]
- [[#Session 3.7 Change of basis and image compression|3.7 换基与图像压缩]]
- [[#Session 3.8 Left and right inverses and pseudoinverse|3.8 左右逆与伪逆]]
- [[#Session 3.9 Exam 3 review|3.9 Exam 3 Review]]
- [[#Unit 3 Exam 完整题解|Unit 3 Exam 完整题解]]

---

## Session 3.1 Symmetric matrices and positive definiteness

### 本节问题、前置知识与尺寸

本节要回答：为什么实对称矩阵是特征值理论中最“友好”的矩阵？怎样从特征值、主元和顺序主子式判断正定？

前置知识：特征值与特征向量、对角化、正交矩阵、行列式和无行交换的消元。设 $A\in\mathbb R^{n\times n}$；证明实特征值时允许特征向量 $x\in\mathbb C^n$，因此必须使用共轭转置。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.1sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S01_Lecture_Lecture_25_Symmetric_Matrices_and_Positive_Definiteness.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S01_Recitation_Problem_Solving_Symmetric_Matrices_and_Positive_Definiteness.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.1prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.1sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Symmetric Matrix|对称矩阵]]、[[Positive Definite Matrix|正定矩阵]]、[[Spectral Decomposition|谱分解]]、[[Testing Positive Definiteness|正定性判别]]。

### 3.1.1 实对称矩阵的谱定理

若 $A=A^T$，称 $A$ 为实[[Symmetric Matrix|对称矩阵]]。课程采用的实谱定理（spectral theorem）是：

> [!theorem] [[Spectral Theorem Proof|实谱定理]]
> 对每个 $A\in\mathbb R^{n\times n}$，若 $A=A^T$，则：
> 1. $A$ 的所有特征值都是实数；
> 2. 不同特征值的特征向量彼此正交；
> 3. $\mathbb R^n$ 存在一组由 $A$ 的特征向量组成的标准正交基。
>
> 因而存在正交矩阵 $Q=[q_1\ \cdots\ q_n]$ 与实对角矩阵 $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$，使
> $$
> A=Q\Lambda Q^T,\qquad Q^TQ=QQ^T=I.
> $$

本课程完整证明前两项，并把“重特征值的每个特征子空间内可选标准正交基”与对称算子的归纳分解作为第三项的课程级论证。严格的完整存在性证明需要先保证特征多项式在 $\mathbb C$ 分裂；这是代数学基本定理提供的背景。

#### 证明一：特征值为何一定是实数

**目标。** 已知 $Ax=\lambda x$、$x\ne0$，证明 $\lambda=\overline\lambda$。

**构造。** 虽然 $A$ 是实矩阵，$x$ 和 $\lambda$ 事先可能是复数。计算同一标量 $x^*Ax$ 的两种表达。

由 $Ax=\lambda x$，左乘 $x^*$：

$$
x^*Ax=x^*(\lambda x)=\lambda x^*x. \tag{1}
$$

另一方面，对 $Ax=\lambda x$ 取共轭转置：

$$
x^*A^*=\overline\lambda x^*.
$$

因为 $A$ 实对称，所以 $A^*=A^T=A$。右乘 $x$ 得

$$
x^*Ax=\overline\lambda x^*x. \tag{2}
$$

比较 (1)、(2)：

$$
(\lambda-\overline\lambda)x^*x=0.
$$

而

$$
x^*x=\sum_{j=1}^n|x_j|^2>0
$$

是严格正实数，所以只能有 $\lambda=\overline\lambda$，即 $\lambda\in\mathbb R$。

> [!warning] 本单元必须修正的旧稿错误
> 对复特征向量不能写“$x^Tx>0$”。例如 $x=(i,1)^T\ne0$，但 $x^Tx=i^2+1=0$。正确的正量是 $x^*x=|i|^2+|1|^2=2$。

#### 证明二：不同特征值的特征向量正交

设 $Ax=\lambda x$、$Ay=\mu y$，其中 $\lambda\ne\mu$。因为 $A=A^*$，

$$
\langle Ax,y\rangle=(Ax)^*y=x^*A^*y=x^*Ay=\langle x,Ay\rangle.
$$

代入特征方程：

$$
\overline\lambda\langle x,y\rangle=\mu\langle x,y\rangle.
$$

上一证明已知 $\lambda$ 为实数，所以

$$
(\lambda-\mu)\langle x,y\rangle=0.
$$

因 $\lambda\ne\mu$，得到 $\langle x,y\rangle=0$。

#### 证明三：为什么能得到一整组标准正交特征向量

前两步已经说明：在 $\mathbb C$ 中出现的特征值实际都为实数。取一个实特征值 $\lambda_1$，因为 $A-\lambda_1I$ 是实奇异矩阵，它有非零实零空间；从中选单位特征向量 $q_1$。

关键是证明正交补 $q_1^\perp$ 在 $A$ 下保持不变。若 $y\in q_1^\perp$，则

$$
q_1^TAy=(A^Tq_1)^Ty=(Aq_1)^Ty=(\lambda_1q_1)^Ty=0.
$$

所以 $Ay\in q_1^\perp$。把 $A$ 限制到这个 $(n-1)$ 维子空间，它仍然是对称算子；在其中重复同一过程。对维数作归纳，得到 $q_1,\ldots,q_n$ 的标准正交特征基。把这些向量作列即得 $Q$，并由 $AQ=Q\Lambda$ 推出

$$
A=Q\Lambda Q^{-1}=Q\Lambda Q^T.
$$

这一步使用的背景事实只有：特征多项式在 $\mathbb C$ 上至少有一个根；这是代数学基本定理在本课程中的使用位置。

#### 从谱分解到投影分解

把 $Q\Lambda Q^T$ 按列展开：

$$
A=\sum_{j=1}^n\lambda_jq_jq_j^T.
$$

$q_jq_j^T$ 是投影到一维子空间 $\operatorname{span}(q_j)$ 的正交投影。对任意 $x$，

$$
Ax=\sum_{j=1}^n\lambda_jq_j(q_j^Tx).
$$

因此 $A$ 的作用可以逐方向理解：先把 $x$ 分解到互相垂直的特征方向，再把第 $j$ 个分量乘以 $\lambda_j$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-spectral-positive-definite.png|820]]

### 3.1.2 正定的定义与等价判据

> [!definition] 正定矩阵
> 实矩阵 $A\in\mathbb R^{n\times n}$ 若满足 $A=A^T$，且对每个 $x\ne0$ 都有
> $$
> x^TAx>0,
> $$
> 则称 $A$ 为正定矩阵。若改为 $x^TAx\ge0$，则称[[Positive Semidefinite Matrix|正半定（positive semidefinite）]]。复数情形把条件改成 $A=A^*$ 与 $x^*Ax>0$。

对实对称 $A$，下列条件等价：

1. $x^TAx>0$ 对所有 $x\ne0$ 成立；
2. 所有特征值 $\lambda_i>0$；
3. 在不需要换行的对称消元中，所有主元 $d_i>0$；
4. 所有顺序主子式
   $$
   \Delta_k=\det A_{1:k,1:k}>0,\qquad k=1,\ldots,n
   $$
   都为正（Sylvester criterion）。

这里“主元为正”必须解释为保持对称结构的 $LDL^T$ 分解中的对角元，或无行交换消元得到的主元；不能对任意行交换后的消元结果直接套用。

#### 二次型判据与特征值判据等价

由谱分解 $A=Q\Lambda Q^T$，令 $y=Q^Tx$。因为 $Q$ 可逆，$x\ne0\iff y\ne0$。于是

$$
x^TAx=x^TQ\Lambda Q^Tx=y^T\Lambda y=\sum_{i=1}^n\lambda_i y_i^2.
$$

- 若全部 $\lambda_i>0$，则至少一个 $y_i\ne0$，所以和严格为正。
- 反之，若某个 $\lambda_k\le0$，取 $x=q_k$，便有 $x^TAx=\lambda_k\le0$，违反正定。

因此二者等价。

#### 主元判据为何成立

若对称消元无换行得到

$$
A=LDL^T,
$$

其中 $L$ 为单位下三角矩阵，$D=\operatorname{diag}(d_1,\ldots,d_n)$ 为主元对角阵。令 $y=L^Tx$，则 $L^T$ 可逆，所以 $x\ne0\iff y\ne0$，并且

$$
x^TAx=x^TLDL^Tx=y^TDy=\sum_{i=1}^n d_i y_i^2.
$$

故 $A$ 正定当且仅当所有 $d_i>0$。此外，前 $k$ 个顺序主子式满足

$$
\Delta_k=d_1d_2\cdots d_k,
$$

所以

$$
d_k=\frac{\Delta_k}{\Delta_{k-1}},\qquad \Delta_0=1.
$$

这又说明“所有主元正”与“所有顺序主子式正”等价。

> [!example] 课件例题：三条路线判断正定
> 设
> $$
> A=\begin{bmatrix}5&2\\2&3\end{bmatrix}.
> $$
> 由两个非对角元都等于 $2$，可见 $A=A^T$。
>
> **主元法：** 第一主元 $d_1=5$；消去第二行第一项时，
> $$
> d_2=3-\frac25\cdot2=\frac{11}{5}>0.
> $$
>
> **顺序主子式法：** $\Delta_1=5>0$，$\Delta_2=15-4=11>0$。
>
> **特征值法：**
> $$
> \det(A-\lambda I)=(5-\lambda)(3-\lambda)-4
> =\lambda^2-8\lambda+11,
> $$
> 所以 $\lambda=4\pm\sqrt5>0$。
>
> 三种方法都证明 $A$ 正定。

### 3.1.3 Recitation：四个性质

1. **正定矩阵必可逆。** 所有特征值正，因此 $\det A=\prod_i\lambda_i>0$。
2. **唯一正定投影矩阵是 $I$。** 投影矩阵满足 $P^2=P$，故特征值只可能是 $0,1$；正定排除 $0$，谱分解于是给出 $P=QIQ^T=I$。
3. **正对角矩阵正定。** 若 $D=\operatorname{diag}(d_i)$ 且 $d_i>0$，则 $x^TDx=\sum d_ix_i^2>0$。
4. **$\det S>0$ 不足以推出正定。** 例如
   $$
   S=\begin{bmatrix}-3&1\\1&-2\end{bmatrix},\qquad \det S=5>0,
   $$
   但取 $x=e_1$，有 $x^TSx=-3<0$。

### 3.1.4 Homework：完整题解

> [!question]- Problem 25.1：找出“所有实矩阵的特征值都为实数”这一伪证明的漏洞
> 题目声称：由 $Ax=\lambda x$ 得 $x^TAx=\lambda x^Tx$，所以 $\lambda=(x^TAx)/(x^Tx)$ 为实数。用
> $$
> A=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad \lambda=i,\quad x=\begin{bmatrix}i\\1\end{bmatrix}
> $$
> 检查。

> [!success]- 解答
> **已知。** $A$ 是逆时针 $90^\circ$ 旋转矩阵。
>
> **验证特征方程：**
> $$
> Ax=\begin{bmatrix}-1\\i\end{bmatrix}
> =i\begin{bmatrix}i\\1\end{bmatrix}=ix.
> $$
>
> **检查伪证明前一步：**
> $$
> x^TAx=\begin{bmatrix}i&1\end{bmatrix}\begin{bmatrix}-1\\i\end{bmatrix}=0,
> $$
> 且
> $$
> \lambda x^Tx=i(i^2+1)=0.
> $$
> 所以前一步没有错；真正的问题是
> $$
> x^Tx=i^2+1=0,
> $$
> 最后一步除以 $x^Tx$ 是除以零。复向量应使用 $x^*x=2$。此外，即便改用 $x^*x$，一般实矩阵的 $x^*Ax$ 也未必为实；只有 $A=A^*$ 时才能保证。

> [!question]- Problem 25.2：哪些矩阵集合构成群？
> 以矩阵乘法为运算，判断：(a) 对称正定矩阵；(b) 正交矩阵；(c) 固定矩阵 $A$ 的所有 $e^{tA}$；(d) 行列式为 $1$ 的矩阵。

> [!success]- 解答
> **(a) 不构成群。** 逆仍然正定，但乘积未必对称。取
> $$
> A=\begin{bmatrix}2&1\\1&1\end{bmatrix},\qquad
> B=\begin{bmatrix}1&1/2\\1/2&1\end{bmatrix}.
> $$
> 两者顺序主子式均正，故正定；然而
> $$
> AB=\begin{bmatrix}5/2&2\\3/2&3/2\end{bmatrix}
> $$
> 不对称，因此不在集合中。
>
> **(b) 构成群。** 若 $Q^TQ=I$，则 $Q^{-1}=Q^T$ 仍正交；若 $A,B$ 正交，
> $$
> (AB)^T(AB)=B^TA^TAB=B^TB=I.
> $$
>
> **(c) 构成群。** 因为都是同一个 $A$ 的函数，$pA$ 与 $qA$ 可交换，
> $$
> e^{pA}e^{qA}=e^{(p+q)A},\qquad (e^{pA})^{-1}=e^{-pA}.
> $$
> 单位元为 $e^{0A}=I$。
>
> **(d) 构成群。** 若 $\det A=\det B=1$，则
> $$
> \det(AB)=1,
> $$
> 且 $\det(A^{-1})=1/\det A=1$。单位矩阵的行列式也是 $1$。

### 3.1.5 边界、反例与易错点

- 正对角元只是正定的必要条件，不是充分条件；$\begin{bmatrix}1&2\\2&1\end{bmatrix}$ 对角元正，却有特征值 $3,-1$。
- 在本节的实对称前提下，$\det A>0$ **迫使**负特征值的个数为偶数，但不能排除存在两个、四个等负特征值；因此它远不足以推出正定。
- “主元符号与特征值符号数相同”是对实对称矩阵的惯性定律语境；不能任意推广到非对称矩阵。
- 半正定情形不能把“所有顺序主子式非负”当作充分条件；完整半正定判据要求所有主子式非负，而不只顺序主子式。

### 3.1.6 三道自检题

> [!question]- 自检 1（计算）
> 判断 $A=\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$ 是否正定，并给出两种依据。

> [!success]- 答案
> $\Delta_1=2>0$、$\Delta_2=3>0$，故正定；或特征值为 $1,3$，均正。

> [!question]- 自检 2（证明）
> 若 $A$ 对称正定，证明 $A^{-1}$ 也对称正定。

> [!success]- 答案
> $A=Q\Lambda Q^T$，故 $A^{-1}=Q\Lambda^{-1}Q^T$；$\Lambda^{-1}$ 的对角元 $1/\lambda_i$ 全正。因此 $A^{-1}$ 对称正定。

> [!question]- 自检 3（条件诊断）
> “$A$ 的主元全正，所以 $A$ 正定”缺少什么前提？

> [!success]- 答案
> 至少要有 $A$ 实对称，并且主元来自不换行、保持对称结构的消元或 $LDL^T$ 分解。

### 知识链小结

$$
A=A^T\Longrightarrow\text{实谱与正交特征基}
\Longrightarrow A=Q\Lambda Q^T
\Longrightarrow x^TAx=\sum\lambda_i y_i^2.
$$

正定性把[[Eigenvalues|特征值]]、主元、顺序主子式、二次型与可逆性连接在一起；下一节把相同思想推广到复数域。

---

## Session 3.2 Complex matrices and fast Fourier transform

### 本节问题、前置知识与尺寸

本节要回答：进入 $\mathbb C^n$ 后，长度、正交、对称和正交矩阵应怎样改写？离散 Fourier 矩阵为何能被递归分解，从 $O(n^2)$ 降为 $O(n\log n)$？

前置知识：复数及其共轭、内积、正交矩阵、矩阵分块。设 $z\in\mathbb C^n$；Fourier 矩阵 $F_n\in\mathbb C^{n\times n}$，下标从 $0$ 到 $n-1$。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S02_Lecture_Lecture_26_Complex_Matrices_Fast_Fourier_Transform_FFT.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S02_Recitation_Problem_Solving_Complex_Matrices.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.2prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.2sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Orthogonality|正交性]]、[[Orthogonal Matrix|正交矩阵]]、[[Fourier Series|Fourier 展开]]。

### 3.2.1 复向量的长度与 Hermitian 内积

旧公式 $z^Tz$ 不再代表长度平方。例如 $z=(1,i)^T$ 时 $z^Tz=1+i^2=0$。正确公式是

$$
\langle z,w\rangle=z^*w=\sum_{k=1}^n\overline z_k w_k,
\qquad
\|z\|^2=z^*z=\sum_{k=1}^n|z_k|^2.
$$

它满足：

1. $\langle z,z\rangle\ge0$，且等号仅在 $z=0$ 时成立；
2. $\langle z,w\rangle=\overline{\langle w,z\rangle}$；
3. 对第二个变量线性，对第一个变量共轭线性。

> [!definition] Hermitian 与 unitary
> - $A\in\mathbb C^{n\times n}$ 若 $A=A^*$，称为[[Hermitian Matrix|Hermitian 矩阵]]；它是实对称矩阵的复数推广。
> - $Q\in\mathbb C^{n\times n}$ 若 $Q^*Q=QQ^*=I$，称为[[Unitary Matrix|unitary 矩阵]]；它是实正交矩阵的复数推广。

Hermitian 谱定理把 3.1 的所有 $T$ 换为 $*$：特征值为实数，可以选 unitary 特征向量矩阵 $Q$，并写成

$$
A=Q\Lambda Q^*.
$$

### 3.2.2 Recitation：完整对角化一个 Hermitian 矩阵

取

$$
A=\begin{bmatrix}2&1-i\\1+i&3\end{bmatrix}=A^*.
$$

特征多项式为

$$
\det(A-\lambda I)=(2-\lambda)(3-\lambda)-(1-i)(1+i)
=\lambda^2-5\lambda+4,
$$

故 $\lambda_1=1$、$\lambda_2=4$，确实都是实数。

当 $\lambda_1=1$ 时，可取

$$
v_1=\begin{bmatrix}1-i\\-1\end{bmatrix};
$$

当 $\lambda_2=4$ 时，可取

$$
v_2=\begin{bmatrix}1\\1+i\end{bmatrix}.
$$

检查 Hermitian 正交性：

$$
v_1^*v_2=\begin{bmatrix}1+i&-1\end{bmatrix}
\begin{bmatrix}1\\1+i\end{bmatrix}=0.
$$

两向量的长度平方都为 $3$，因此

$$
Q=\frac1{\sqrt3}\begin{bmatrix}1-i&1\\-1&1+i\end{bmatrix},\qquad
Q^*Q=I,
$$

并有

$$
A=Q\begin{bmatrix}1&0\\0&4\end{bmatrix}Q^*.
$$

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-complex-action.png|780]]

### 3.2.3 离散 Fourier 矩阵

令

$$
\omega_n=e^{2\pi i/n},\qquad \omega_n^n=1.
$$

课程使用未归一化 Fourier 矩阵

$$
(F_n)_{jk}=\omega_n^{jk},\qquad j,k=0,1,\ldots,n-1.
$$

第 $k$ 列是离散频率 $k$ 在 $n$ 个采样点上的取值。

#### 证明：Fourier 列彼此正交

第 $k$ 与第 $\ell$ 列的内积为

$$
\sum_{j=0}^{n-1}\overline{\omega_n^{jk}}\omega_n^{j\ell}
=\sum_{j=0}^{n-1}\omega_n^{j(\ell-k)}.
$$

- 若 $k=\ell$，每项为 $1$，和为 $n$。
- 若 $k\ne\ell$，令 $r=\omega_n^{\ell-k}\ne1$，但 $r^n=1$。由有限几何级数
  $$
  \sum_{j=0}^{n-1}r^j=\frac{1-r^n}{1-r}=0.
  $$

因此

$$
F_n^*F_n=nI,\qquad \frac1{\sqrt n}F_n\text{ 是 unitary 矩阵},\qquad
F_n^{-1}=\frac1nF_n^*.
$$

对 $n=4$，$\omega_4=i$：

$$
F_4=
\begin{bmatrix}
1&1&1&1\\
1&i&-1&-i\\
1&-1&1&-1\\
1&-i&-1&i
\end{bmatrix}.
$$

若 $e_0=(1,0,0,0)^T$，则 $F_4e_0=(1,1,1,1)^T$：时间域的单个脉冲包含所有频率，且幅度相同。

### 3.2.4 [[Radix-2 FFT|FFT 的偶奇递归]]

普通矩阵—向量乘法 $F_nx$ 需要约 $n^2$ 次标量运算。对 $n=2m$，先用置换矩阵 $P$ 把输入排成偶数下标、奇数下标：

$$
Px=(x_0,x_2,\ldots,x_{2m-2},x_1,x_3,\ldots,x_{2m-1})^T.
$$

令 $D=\operatorname{diag}(1,\omega_{2m},\ldots,\omega_{2m}^{m-1})$，则

$$
F_{2m}=
\begin{bmatrix}I&D\\I&-D\end{bmatrix}
\begin{bmatrix}F_m&0\\0&F_m\end{bmatrix}P.
$$

原因是 $\omega_{2m}^{2jk}=\omega_m^{jk}$：偶数样本与奇数样本各自形成一个长度 $m$ 的 DFT，$D$ 提供奇数部分的相位修正（twiddle factors），最后用加减法合并。

若 $T(n)$ 是运算量，则

$$
T(n)=2T(n/2)+O(n),
$$

递归 $\log_2n$ 层，得到 $T(n)=O(n\log n)$。这不是近似算法；它与直接乘 $F_n$ 得到完全相同的结果，只是利用结构避免重复计算。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-fft-butterfly.png|820]]

### 3.2.5 Homework：完整题解

> [!question]- Problem 26.1：计算 $F_2$

> [!success]- 解答
> $\omega_2=e^{2\pi i/2}=-1$，所以
> $$
> F_2=\begin{bmatrix}1&1\\1&\omega_2\end{bmatrix}
> =\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
> $$
> 检查：$F_2^*F_2=2I$，所以 $(1/\sqrt2)F_2$ 是 unitary。

> [!question]- Problem 26.2：求 $F_4$ 分解中的 $D$ 与 $P$，并验算
> 求
> $$
> F_4=\begin{bmatrix}I&D\\I&-D\end{bmatrix}
> \begin{bmatrix}F_2&0\\0&F_2\end{bmatrix}P.
> $$

> [!success]- 解答
> 四次单位根依次为 $1,i,-1,-i$，故
> $$
> D=\begin{bmatrix}1&0\\0&i\end{bmatrix}.
> $$
> $P$ 把 $(x_0,x_1,x_2,x_3)^T$ 排成 $(x_0,x_2,x_1,x_3)^T$：
> $$
> P=\begin{bmatrix}
> 1&0&0&0\\
> 0&0&1&0\\
> 0&1&0&0\\
> 0&0&0&1
> \end{bmatrix}.
> $$
> 先算中间两块：
> $$
> \begin{bmatrix}F_2&0\\0&F_2\end{bmatrix}P
> =\begin{bmatrix}
> 1&0&1&0\\1&0&-1&0\\0&1&0&1\\0&1&0&-1
> \end{bmatrix}.
> $$
> 再左乘 butterfly 矩阵，得到
> $$
> \begin{bmatrix}
> 1&1&1&1\\1&i&-1&-i\\1&-1&1&-1\\1&-i&-1&i
> \end{bmatrix}=F_4.
> $$

### 3.2.6 边界、反例与易错点

- $A^T=A$ 与 $A^*=A$ 在复数域不同；复数对称矩阵未必有实特征值。
- $F_n$ 本身不是 unitary；$(1/\sqrt n)F_n$ 才是。若采用其他 DFT 符号约定，指数可能写成 $-2\pi i/n$，逆变换的符号会相应交换。
- “实矩阵的复特征值成共轭对”只依赖特征多项式系数为实，不要求矩阵对称。
- FFT 的 $O(n\log n)$ 结论通常假设 $n$ 可递归分解；其他长度也有相应算法，但本课只需掌握二进制分解。

### 3.2.7 三道自检题

> [!question]- 自检 1（概念）
> 为什么 $z^Tz$ 不能作为复向量的长度平方？

> [!success]- 答案
> 它可能为零或非实数，即使 $z\ne0$。例如 $(1,i)^T(1,i)=0$；$z^*z=\sum|z_i|^2$ 才满足正定性。

> [!question]- 自检 2（计算）
> 求 $F_4^{-1}$。

> [!success]- 答案
> 因 $F_4^*F_4=4I$，故 $F_4^{-1}=\frac14F_4^*$。

> [!question]- 自检 3（证明）
> 证明 unitary 矩阵保持长度。

> [!success]- 答案
> 若 $Q^*Q=I$，则
> $$
> \|Qx\|^2=(Qx)^*(Qx)=x^*Q^*Qx=x^*x=\|x\|^2.
> $$

### 知识链小结

$$
T\mapsto *\Longrightarrow\text{Hermitian/unitary}
\Longrightarrow\text{复正交谱结构}
\Longrightarrow F_n^*F_n=nI
\Longrightarrow\text{FFT 的偶奇分解}.
$$

---

## Session 3.3 Positive definite matrices and minima

### 本节问题、前置知识与尺寸

本节要回答：正定性为什么等价于一个二次函数沿每个方向都向上？消元、配方、Hessian 与极小值为什么会出现同一组数？

前置知识：3.1 的正定判据、多元微积分中的临界点、$LDL^T$ 分解。设 $A\in\mathbb R^{n\times n}$ 对称，$x\in\mathbb R^n$；二次型（quadratic form）是 $q(x)=x^TAx$。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S03_Lecture_Lecture_27_Positive_Definite_Matrices_and_Minima.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S03_Recitation_Problem_Solving_Positive_Definite_Matrices_and_Minima.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.3prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.3sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Positive Definite Matrix|正定矩阵]]、[[Quadratic Form|二次型]]、[[Testing Positive Definiteness|正定性判别]]、[[Least Squares|最小二乘]]。

### 3.3.1 二阶二次型的四个判据

对称矩阵

$$
A=\begin{bmatrix}a&b\\b&c\end{bmatrix}
$$

对应

$$
q(x,y)=\begin{bmatrix}x&y\end{bmatrix}A\begin{bmatrix}x\\y\end{bmatrix}
=ax^2+2bxy+cy^2.
$$

$A$ 正定等价于以下任一条件：

1. 两个特征值都正；
2. $a>0$ 且 $ac-b^2>0$；
3. 两个对称消元主元 $a$ 与 $(ac-b^2)/a$ 都正；
4. $q(x,y)>0$ 对所有 $(x,y)\ne(0,0)$ 成立。

以

$$
A_y=\begin{bmatrix}2&6\\6&y\end{bmatrix}
$$

为例，$\Delta_1=2>0$，$\Delta_2=2y-36$，故

$$
A_y\succ0\iff y>18.
$$

$y=18$ 是边界：

$$
x^TA_{18}x=2x_1^2+12x_1x_2+18x_2^2
=2(x_1+3x_2)^2\ge0,
$$

但在 $(x_1,x_2)=(3,-1)$ 处为零，因此只是正半定。

### 3.3.2 配方就是 $LDL^T$

课件比较

$$
q_+(x,y)=2x^2+12xy+20y^2
$$

与

$$
q_-(x,y)=2x^2+12xy+7y^2.
$$

逐步配方：

$$
2x^2+12xy+20y^2
=2(x+3y)^2+2y^2>0
$$

对非零 $(x,y)$ 成立；而

$$
2x^2+12xy+7y^2
=2(x+3y)^2-11y^2
$$

在 $(x,y)=(-3,1)$ 处为负。配方中的系数 $2,2$ 和 $2,-11$，恰好是对应矩阵消元后的主元：

$$
\begin{bmatrix}2&6\\6&20\end{bmatrix}
\longrightarrow
\begin{bmatrix}2&6\\0&2\end{bmatrix},
$$

$$
\begin{bmatrix}2&6\\6&7\end{bmatrix}
\longrightarrow
\begin{bmatrix}2&6\\0&-11\end{bmatrix}.
$$

一般地，若 $A=LDL^T$，则

$$
x^TAx=(L^Tx)^TD(L^Tx),
$$

这就是高维“配方”：$L^Tx$ 给出新的线性表达式，$D$ 的主元给出每个平方项前的系数。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-quadratic-bowl.png|820]]

### 3.3.3 [[Hessian Matrix|Hessian]] 与极小值

对二次函数

$$
f(x)=\frac12x^TAx-b^Tx+c,\qquad A=A^T,
$$

梯度与 Hessian 为

$$
\nabla f(x)=Ax-b,\qquad \nabla^2f(x)=A.
$$

若 $A\succ0$，则临界点唯一：

$$
x_*=A^{-1}b.
$$

要证明它是唯一全局极小点，不只说“二阶导数为正”，而是令 $h=x-x_*$ 并完整展开：

$$
\begin{aligned}
f(x_*+h)
&=\frac12(x_*+h)^TA(x_*+h)-b^T(x_*+h)+c\\
&=f(x_*)+h^T(Ax_*-b)+\frac12h^TAh\\
&=f(x_*)+\frac12h^TAh.
\end{aligned}
$$

因为 $A\succ0$，$h\ne0$ 时 $h^TAh>0$，所以 $f(x)>f(x_*)$。这同时证明了局部极小与全局唯一性。

边界情况：

- $A\succeq0$ 时函数是凸的，但可能沿零空间方向平坦，极小点可能不唯一；若 $b$ 在 $A$ 的列空间之外，甚至可能无下界。
- $A$ 有负特征值时，沿对应特征向量 $q$，$f(tq)$ 的二次项为 $(\lambda/2)t^2$，会向 $-\infty$ 下降。
- $A$ 同时有正、负特征值时：若 $b=0$，纯二次函数在原点呈马鞍形；更一般地，若 $A$ 可逆，则唯一临界点 $x_*=A^{-1}b$ 呈马鞍形。不能在 $b\ne0$ 时无条件把原点称为临界点。

#### 三维课件例子

$$
A=\begin{bmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{bmatrix}.
$$

顺序主子式为

$$
\Delta_1=2,\qquad \Delta_2=3,\qquad \Delta_3=4,
$$

因此 $A\succ0$。相应主元为

$$
d_1=2,\qquad d_2=\frac32,\qquad d_3=\frac43.
$$

二次型

$$
x^TAx=2x_1^2+2x_2^2+2x_3^2-2x_1x_2-2x_2x_3
$$

的等值面 $x^TAx=1$ 是椭球；特征向量给出主轴方向，半轴长度与 $1/\sqrt{\lambda_i}$ 成正比。

### 3.3.4 Recitation：带参数的三种判别

设

$$
B(c)=\begin{bmatrix}
2&-1&-1\\
-1&2&-1\\
-1&-1&2+c
\end{bmatrix}.
$$

顺序主子式：

$$
\Delta_1=2,\qquad \Delta_2=3,\qquad \Delta_3=3c.
$$

所以 $c>0$ 时正定。对称消元得到主元

$$
2,\quad \frac32,\quad c.
$$

同时，完整配方是

$$
\begin{aligned}
\begin{bmatrix}x&y&z\end{bmatrix}B(c)
\begin{bmatrix}x\\y\\z\end{bmatrix}
&=2\left(x-\frac y2-\frac z2\right)^2
+\frac32(y-z)^2+cz^2.
\end{aligned}
$$

因此：

- $c>0$：三个平方项系数都正，且对应三角变换可逆，所以正定；
- $c=0$：表达式非负，但 $(x,y,z)=(1,1,1)$ 使它为零，所以正半定但不正定；
- $c<0$：取 $x=y=z=1$，前两个平方项为零，剩下 $c<0$，故不半正定。

### 3.3.5 Homework：完整题解

> [!question]- Problem 27.1：证明两个正定矩阵乘积的特征值仍为正
> $A,B$ 均为实对称正定，$AB$ 未必对称。由 $ABx=\lambda x$ 出发，证明 $\lambda>0$。

> [!success]- 解答
> 因 $B$ 可逆，$x\ne0$ 时 $Bx\ne0$。特征向量事先可能是复向量，所以使用 Hermitian 内积。由 $ABx=\lambda x$，左乘 $(Bx)^*$：
> $$
> (Bx)^*ABx=\lambda (Bx)^*x=\lambda x^*Bx.
> $$
> 实对称正定矩阵对非零复向量也满足 $z^*Az>0$：把 $z=p+iq$ 展开可得 $z^*Az=p^TAp+q^TAq$。因此左边 $(Bx)^*A(Bx)>0$，分母 $x^*Bx>0$。于是
> $$
> \lambda=\frac{(Bx)^*A(Bx)}{x^*Bx}>0.
> $$
> 这个商本身是实数，所以也证明了 $AB$ 的特征值为实且正。另一种结构证明是 $AB$ 与 $B^{1/2}AB^{1/2}$ 相似，而后者对称正定。

> [!question]- Problem 27.2：求矩阵的二次型并判断符号
> 对
> $$
> A=\begin{bmatrix}1&5\\7&9\end{bmatrix},
> $$
> 求 $[x\ y]A[x\ y]^T$，判断它恒正、恒负还是可正可负。

> [!success]- 解答
> 逐步相乘：
> $$
> A\begin{bmatrix}x\\y\end{bmatrix}
> =\begin{bmatrix}x+5y\\7x+9y\end{bmatrix},
> $$
> 因而
> $$
> q(x,y)=x(x+5y)+y(7x+9y)=x^2+12xy+9y^2.
> $$
> 取 $(x,y)=(1,0)$，$q=1>0$；取 $(2,-2)$，
> $$
> q=4-48+36=-8<0.
> $$
> 所以可正可负。
>
> 注意 $A$ 不对称，二次型实际只由
> $$
> \frac{A+A^T}{2}=\begin{bmatrix}1&6\\6&9\end{bmatrix}
> $$
> 决定；其行列式 $9-36=-27<0$，直接显示一正一负两个方向。不能把非对称 $A$ 的“正定判据”不加说明地直接套用。

### 3.3.6 边界、反例与易错点

- Hessian 正定是临界点为严格局部极小的充分条件；若 Hessian 仅半正定，二阶检验可能无结论，必须看高阶项。
- 对一般 $C^2$ 函数，$\nabla^2f(x_*)\succ0$ 只先给出局部极小；二次函数或全局 Hessian 半正定时才能直接推全局凸性。
- “所有顺序主子式非负”不足以判正半定；$c=0$ 的例子可由结构直接验证，但一般半正定要检查所有主子式或特征值。

### 3.3.7 三道自检题

> [!question]- 自检 1（配方）
> 把 $3x^2+6xy+5y^2$ 配成平方和并判断正定性。

> [!success]- 答案
> $$
> 3x^2+6xy+5y^2=3(x+y)^2+2y^2>0
> $$
> 对非零 $(x,y)$ 成立，因此对应矩阵正定。

> [!question]- 自检 2（半正定边界）
> 判断 $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$ 是正定、半正定还是不定，并说明“顺序主子式非负”为何在这里不能被误读成正定。

> [!success]- 答案
> $x^TAx=(x_1+2x_2)^2\ge0$，故 $A\succeq0$；但 $A( -2,1)^T=0$，存在非零零空间方向，故不正定。其顺序主子式为 $1,0$；“非负”在这个例子中可验证半正定，却不能作为一般矩阵只检查顺序主子式的充分判据。

> [!question]- 自检 3（极小值）
> 若 $A\succ0$，求 $f(x)=\tfrac12x^TAx-b^Tx$ 的最小点与最小值表达式。

> [!success]- 答案
> $x_*=A^{-1}b$；代入得
> $$
> f(x_*)=-\frac12b^TA^{-1}b.
> $$

### 知识链小结

$$
A=LDL^T\Longleftrightarrow\text{完成平方}
\Longleftrightarrow x^TAx\text{ 的曲率}
\Longleftrightarrow\text{Hessian 与极小值}.
$$

---

## Session 3.4 Similar matrices and Jordan form

### 本节问题、前置知识与尺寸

本节先补齐 Lecture 28 开头关于正定矩阵封闭性质与 $A^TA$ 的结论，再回答：相似矩阵为何表示“同一个线性变换的不同坐标”？当特征向量不足时，Jordan 链怎样补足基？Jordan 块怎样影响矩阵幂和微分方程长期行为？

前置知识：对角化 $A=S\Lambda S^{-1}$、特征空间、矩阵幂。设 $A,B\in\mathbb F^{n\times n}$。Jordan 形要求特征多项式在底层域 $\mathbb F$ 上分裂；在 $\mathbb C$ 上总能分裂，在 $\mathbb R$ 上则未必，例如非平凡二维旋转没有实 Jordan 对角形。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S04_Recitation_Problem_Solving_Similar_Matrices.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.4prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.4sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Positive Definite Matrix|正定矩阵]]、[[Similar Matrix|相似矩阵]]、[[Diagonalization|对角化]]、[[Jordan Form|Jordan 形]]、[[Matrix Exponential|矩阵指数]]。

### 3.4.1 Lecture 28 开头：正定矩阵的封闭性质与 $A^TA$

这一段属于官方 Session 3.4，而不是上一节的二次型例题。以下每个结论都保留假设；尤其“正定”默认矩阵为实对称方阵。

#### 正定矩阵之和仍正定

若 $A\succ0$、$B\succ0$ 且尺寸相同，则 $A+B$ 对称，并且对任意 $x\ne0$，

$$
x^T(A+B)x=x^TAx+x^TBx>0+0=0.
$$

故 $A+B\succ0$。这里不能把“和”替换为“乘积”：$AB$ 未必对称。

#### 正定矩阵之逆仍正定

由谱定理，$A=Q\Lambda Q^T$，其中 $\lambda_i>0$。于是

$$
A^{-1}=Q\Lambda^{-1}Q^T,
\qquad
\Lambda^{-1}=\operatorname{diag}(1/\lambda_1,\ldots,1/\lambda_n).
$$

所有 $1/\lambda_i$ 仍为正数，所以 $A^{-1}\succ0$。

#### $C^TC$ 总半正定，满列秩时才正定

对任意 $C\in\mathbb R^{m\times n}$ 与 $x\in\mathbb R^n$，

$$
x^TC^TCx=(Cx)^T(Cx)=\|Cx\|^2\ge0,
$$

故 $C^TC\succeq0$。进一步，

$$
\begin{aligned}
C^TC\succ0
&\iff \|Cx\|^2>0\quad\text{对每个 }x\ne0\\
&\iff Cx=0\text{ 只有零解}\\
&\iff N(C)=\{0\}\\
&\iff \operatorname{rank}(C)=n.
\end{aligned}
$$

因此 $C^TC$ 可逆当且仅当 $C$ 满列秩。此时正规方程 $C^TC\hat x=C^Tb$ 有唯一解；列相关时仍可用 SVD 或伪逆描述全部最小二乘解。

### 3.4.2 相似关系保存什么

> [!definition] 相似矩阵
> 若存在可逆 $M\in\mathbb F^{n\times n}$ 使
> $$
> B=M^{-1}AM,
> $$
> 则称 $A$ 与 $B$ 相似。$M$ 的列通常是新基在旧坐标中的表示。

相似变换保存：

- 特征多项式、特征值及其代数重数；
- 每个特征值的几何重数；
- 秩、行列式、迹；
- 最小多项式与 Jordan 块尺寸；
- 对任意多项式 $p$，$p(B)=M^{-1}p(A)M$。

#### 证明特征值保存

若 $Ax=\lambda x$，令 $y=M^{-1}x\ne0$。则

$$
By=M^{-1}AM(M^{-1}x)=M^{-1}Ax=\lambda M^{-1}x=\lambda y.
$$

所以 $\lambda$ 也是 $B$ 的特征值。反向使用 $A=MBM^{-1}$ 得到完全相同的结论。

#### 证明多项式保持相似

先看幂：

$$
B^2=(M^{-1}AM)(M^{-1}AM)=M^{-1}A^2M,
$$

中间的 $MM^{-1}$ 消去。归纳得 $B^k=M^{-1}A^kM$，于是

$$
p(B)=\sum_{k=0}^rc_kB^k
=M^{-1}\left(\sum_{k=0}^rc_kA^k\right)M
=M^{-1}p(A)M.
$$

相同论证还能通过收敛幂级数给出 $e^B=M^{-1}e^AM$。

### 3.4.3 对角化与重复特征值

若 $A$ 有 $n$ 个线性无关特征向量，取 $S=[x_1\ \cdots\ x_n]$，则

$$
AS=S\Lambda,\qquad S^{-1}AS=\Lambda.
$$

有 $n$ 个互不相同特征值时必有 $n$ 个独立特征向量，所以必可对角化。但特征值重复时，代数重数不保证几何重数足够。例如

$$
\begin{bmatrix}4&0\\0&4\end{bmatrix}
$$

有二维特征空间，而

$$
\begin{bmatrix}4&1\\0&4\end{bmatrix}
$$

只有一维特征空间；二者不能相似。

### 3.4.4 Jordan 链与 Jordan 标准形

对特征值 $\lambda$，大小为 $k$ 的 Jordan 块是

$$
J_k(\lambda)=
\begin{bmatrix}
\lambda&1&0&\cdots&0\\
0&\lambda&1&\ddots&\vdots\\
\vdots&\ddots&\ddots&\ddots&0\\
0&\cdots&0&\lambda&1\\
0&\cdots&\cdots&0&\lambda
\end{bmatrix}
=\lambda I+N,
$$

其中 $N^k=0$。对应基向量形成广义特征向量链：

$$
(A-\lambda I)v_1=0,\qquad
(A-\lambda I)v_2=v_1,\qquad\ldots,\qquad
(A-\lambda I)v_k=v_{k-1}.
$$

在特征多项式分裂的条件下，存在可逆 $S$ 使 $S^{-1}AS$ 为 Jordan 块的直和。课程重点是理解块结构与计算，不从头证明整个存在性定理。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-jordan-chain.png|820]]

#### Jordan 块的幂和指数

因为 $\lambda I$ 与 $N$ 可交换，

$$
J_k(\lambda)^m
=\sum_{j=0}^{\min(m,k-1)}\binom mj\lambda^{m-j}N^j.
$$

矩阵指数为

$$
e^{tJ_k(\lambda)}
=e^{\lambda t}e^{tN}
=e^{\lambda t}\sum_{j=0}^{k-1}\frac{t^j}{j!}N^j.
$$

因此长期行为不只由 $e^{\lambda t}$ 决定：Jordan 块大小 $k$ 还带来最高 $t^{k-1}$ 的多项式因子。

> [!warning] 稳定性的重要边界
> 若 $\operatorname{Re}\lambda<0$，指数衰减最终压过任意固定次数多项式；若 $\operatorname{Re}\lambda>0$，解指数增长。若 $\operatorname{Re}\lambda=0$，非平凡 Jordan 块会产生多项式增长。尤其“所有特征值纯虚，所以解有界”只在这些纯虚特征值对应 Jordan 块全为 $1\times1$（即相应部分可对角化）时成立。

### 3.4.5 课件与 Recitation 例子

课件用

$$
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}
$$

说明同一个相似类有许多矩阵。$A$ 的特征值是 $3,1$，故相似于 $\operatorname{diag}(3,1)$。任取可逆 $M$，$M^{-1}AM$ 仍有相同谱结构，但条目可能完全不同。

Recitation 的核心判断：

1. 若 $A\sim B$，则 $2A^3+A-3I\sim2B^3+B-3I$，因为多项式保持相似。
2. 两个 $3\times3$ 矩阵若都有互异特征值 $1,0,-1$，则都相似于同一对角矩阵，故彼此相似。
3. 两个 Jordan 形即使特征值相同，若某个 $\dim N(J-\lambda I)$ 不同，就不相似。更一般地，$\dim N((A-\lambda I)^j)$ 的序列能恢复 Jordan 块尺寸。

### 3.4.6 Homework：完整题解

> [!question]- Problem 28.1：证明两个零特征值 Jordan 矩阵不相似
> 设
> $$
> J=\begin{bmatrix}0&1&0&0\\0&0&0&0\\0&0&0&1\\0&0&0&0\end{bmatrix},\quad
> K=\begin{bmatrix}0&1&0&0\\0&0&1&0\\0&0&0&0\\0&0&0&0\end{bmatrix}.
> $$
> 对一般 $M=(m_{ij})$，证明 $JM=MK$ 会迫使 $M$ 不可逆。

> [!success]- 解答
> 左乘 $J$ 会把 $M$ 的第 2 行放到第 1 行、第 4 行放到第 3 行，其余行变零：
> $$
> JM=\begin{bmatrix}
> m_{21}&m_{22}&m_{23}&m_{24}\\
> 0&0&0&0\\
> m_{41}&m_{42}&m_{43}&m_{44}\\
> 0&0&0&0
> \end{bmatrix}.
> $$
> 右乘 $K$ 后，各列为 $0$、$M$ 的第 1 列、第 2 列、$0$：
> $$
> MK=\begin{bmatrix}
> 0&m_{11}&m_{12}&0\\
> 0&m_{21}&m_{22}&0\\
> 0&m_{31}&m_{32}&0\\
> 0&m_{41}&m_{42}&0
> \end{bmatrix}.
> $$
> 比较元素得到 $m_{21}=m_{41}=0$，又由第二行得 $m_{21}=m_{22}=0$，第四行得 $m_{41}=m_{42}=0$，第一、三行进一步给出 $m_{11}=m_{31}=0$。因此 $M$ 的第一列全为零，$M$ 不可逆。
>
> 若 $J\sim K$，应存在可逆 $M$ 使 $K=M^{-1}JM$，等价于 $JM=MK$；与上述结论矛盾。故不相似。结构上，$J$ 的块尺寸为 $2+2$，$K$ 为 $3+1$。

> [!question]- Problem 28.2：解释五个相似性判断

> [!success]- 解答
> **(a)** 若 $A=M^{-1}BM$，则
> $$
> A^2=M^{-1}BM M^{-1}BM=M^{-1}B^2M,
> $$
> 所以 $A^2\sim B^2$。
>
> **(b)** 逆命题不成立。取
> $$
> A=\begin{bmatrix}0&0\\0&0\end{bmatrix},\qquad
> B=\begin{bmatrix}0&1\\0&0\end{bmatrix}.
> $$
> 则 $A^2=B^2=0$，但 $A$ 的秩为 $0$、$B$ 的秩为 $1$，故 $A\not\sim B$。
>
> **(c)** $\operatorname{diag}(3,4)$ 与 $\begin{bmatrix}3&1\\0&4\end{bmatrix}$ 都有两个互异特征值 $3,4$，因此都可对角化且彼此相似。可显式取
> $$
> M=\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad
> M^{-1}\begin{bmatrix}3&1\\0&4\end{bmatrix}M
> =\begin{bmatrix}3&0\\0&4\end{bmatrix}.
> $$
>
> **(d)** $3I$ 与 $\begin{bmatrix}3&1\\0&3\end{bmatrix}$ 不相似。前者特征空间维数为 $2$，后者只有 $1$；并且 $M^{-1}(3I)M=3I$ 对任何可逆 $M$ 都不变。
>
> **(e)** 设 $P$ 交换前两坐标，则 $P^{-1}=P$。先交换 $A$ 的前两行是 $PA$，再交换前两列是 $PAP=P^{-1}AP$，所以新矩阵与 $A$ 相似。

### 3.4.7 边界、反例与易错点

- “特征值相同”通常不推出相似；还需相同 Jordan 块结构。只有特征值互异从而都可对角化时，才可直接推出。
- 相似与合同（congruence）不同：$M^{-1}AM$ 描述同一线性算子换基，$M^TAM$ 描述同一二次型换坐标。
- Jordan 形在浮点计算中极不稳定；它主要提供理论分类，不是大规模数值算法首选。
- 实矩阵若有非实特征值，必须到 $\mathbb C$ 上使用普通 Jordan 形，或在 $\mathbb R$ 上使用 $2\times2$ 实块。

### 3.4.8 三道自检题

> [!question]- 自检 1（结构）
> $J=\begin{bmatrix}2&1\\0&2\end{bmatrix}$ 的 $J^m$ 是什么？

> [!success]- 答案
> 写成 $J=2I+N$、$N^2=0$：
> $$
> J^m=2^mI+m2^{m-1}N
> =\begin{bmatrix}2^m&m2^{m-1}\\0&2^m\end{bmatrix}.
> $$

> [!question]- 自检 2（稳定性）
> $A=\begin{bmatrix}i&1\\0&i\end{bmatrix}$ 的 $e^{tA}$ 是否有界？

> [!success]- 答案
> $$
> e^{tA}=e^{it}\begin{bmatrix}1&t\\0&1\end{bmatrix},
> $$
> 模长含线性因子 $t$，故不有界；纯虚谱不足以保证有界。

> [!question]- 自检 3（判断）
> 迹和行列式都相同的两个 $2\times2$ 矩阵一定相似吗？

> [!success]- 答案
> 不一定。$3I$ 与 $\begin{bmatrix}3&1\\0&3\end{bmatrix}$ 迹、行列式相同，却因几何重数不同而不相似。

### 知识链小结

$$
\text{换基}\Longrightarrow B=M^{-1}AM
\Longrightarrow\text{谱与多项式不变量}
\Longrightarrow\begin{cases}
\Lambda,&\text{特征向量够}\,;\\
J,&\text{用广义特征向量补足}.
\end{cases}
$$

---

## Session 3.5 Singular value decomposition

### 本节问题、前置知识与尺寸

本节要回答：一个任意矩阵 $A\in\mathbb R^{m\times n}$ 如何在输入空间与输出空间分别选择最佳的正交坐标，使它只剩沿互相垂直方向的伸缩？

前置知识：谱定理、四个基本子空间、$A^TA$ 的半正定性。尺寸必须始终检查：

$$
A_{m\times n}=U_{m\times m}\Sigma_{m\times n}V^T_{n\times n}.
$$

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S05_Lecture_Lecture_29_Singular_Value_Decomposition.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S05_Recitation_Problem_Solving_Computing_the_Singular_Value_Decomposition.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.5prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.5sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Singular Value Decomposition|奇异值分解]]、[[Spectral Decomposition|谱分解]]、[[Orthogonality|正交性]]、[[Matrix Rank|矩阵秩]]。

### 3.5.1 SVD 定理与几何意义

> [!theorem] 奇异值分解（SVD）
> 对每个 $A\in\mathbb R^{m\times n}$，存在正交矩阵 $U\in\mathbb R^{m\times m}$、$V\in\mathbb R^{n\times n}$，以及仅在主对角线上可能非零的 $\Sigma\in\mathbb R^{m\times n}$，使
> $$
> A=U\Sigma V^T.
> $$
> 非零对角元按
> $$
> \sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0
> $$
> 排列，称为[[Singular Value|奇异值（singular value）]]；$r=\operatorname{rank}(A)$。

等价地，若 $v_i$ 是 $V$ 的第 $i$ 列、$u_i$ 是 $U$ 的第 $i$ 列，则

$$
Av_i=\sigma_i u_i\quad(1\le i\le r),
$$

而 $Av_i=0$（$i>r$）。因此 $V^T$ 先把输入转到右奇异向量坐标，$\Sigma$ 沿正交坐标轴伸缩或压扁，$U$ 再把结果旋转/反射到输出空间。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-svd-geometry.png|860]]

单位球经 $A$ 映成椭球：轴方向为 $u_i$，半轴长度为 $\sigma_i$；零奇异值把相应输入方向压到零。

### 3.5.2 [[SVD Existence Proof|SVD 的课程级存在性证明]]

**目标。** 从已经证明的实对称谱定理构造 $U,\Sigma,V$。

**第一步：对角化 $A^TA$。**

$A^TA\in\mathbb R^{n\times n}$ 对称半正定，因为

$$
x^TA^TAx=\|Ax\|^2\ge0.
$$

谱定理给出标准正交特征基 $v_1,\ldots,v_n$：

$$
A^TAv_i=\lambda_i v_i,\qquad \lambda_i\ge0.
$$

按 $\lambda_1\ge\cdots\ge\lambda_n$ 排列，定义

$$
\sigma_i=\sqrt{\lambda_i}\ge0.
$$

**第二步：由右奇异向量构造左奇异向量。**

对 $\sigma_i>0$，定义

$$
u_i=\frac{Av_i}{\sigma_i}.
$$

验证长度：

$$
\|u_i\|^2
=\frac{v_i^TA^TAv_i}{\sigma_i^2}
=\frac{\lambda_i v_i^Tv_i}{\lambda_i}=1.
$$

验证不同 $u_i$ 正交：若 $i\ne j$，

$$
u_i^Tu_j
=\frac{v_i^TA^TAv_j}{\sigma_i\sigma_j}
=\frac{\lambda_jv_i^Tv_j}{\sigma_i\sigma_j}=0.
$$

因此非零奇异值对应的 $u_1,\ldots,u_r$ 是标准正交组，且按定义 $Av_i=\sigma_i u_i$。

**第三步：处理零空间与补齐基。**

$\lambda_i=0$ 当且仅当

$$
0=v_i^TA^TAv_i=\|Av_i\|^2,
$$

即 $Av_i=0$。所以 $v_{r+1},\ldots,v_n$ 构成 $N(A)$ 的标准正交基。再把 $u_1,\ldots,u_r$ 补成 $\mathbb R^m$ 的标准正交基；补入的 $u_{r+1},\ldots,u_m$ 可选为 $N(A^T)$ 的标准正交基。

令

$$
V=[v_1\ \cdots\ v_n],\qquad U=[u_1\ \cdots\ u_m],
$$

并把 $\sigma_i$ 放进 $\Sigma$，便有 $AV=U\Sigma$；右乘 $V^T=V^{-1}$ 得

$$
A=U\Sigma V^T.
$$

> [!note] 证明层级说明
> 这个证明把 SVD 的存在性完全归约到实对称谱定理。若要从实数公理开始完整证明谱定理，还需补入代数学基本定理或等价的紧致性论证；MIT 18.06SC 把谱定理作为本课程已建立的核心结果。

### 3.5.3 四个基本子空间在 SVD 中的归位

若 $\operatorname{rank}(A)=r$，则

$$
\begin{aligned}
C(A^T)&=\operatorname{span}(v_1,\ldots,v_r),\\
N(A)&=\operatorname{span}(v_{r+1},\ldots,v_n),\\
C(A)&=\operatorname{span}(u_1,\ldots,u_r),\\
N(A^T)&=\operatorname{span}(u_{r+1},\ldots,u_m).
\end{aligned}
$$

其中 $A$ 在行空间与列空间之间给出一一对应：$v_i\mapsto\sigma_i u_i$；在零空间上则全部映到零。这正是伪逆能够“只逆可逆部分”的原因。

SVD 还给出秩一展开：

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^T.
$$

每一项 $u_iv_i^T$ 先取输入沿 $v_i$ 的分量，再沿 $u_i$ 输出。

### 3.5.4 课件例题：按[[Computing an SVD|标准流程计算 SVD]]

设

$$
A=\begin{bmatrix}4&4\\-3&3\end{bmatrix}.
$$

**Step 1：计算 $A^TA$。**

$$
A^TA=
\begin{bmatrix}4&-3\\4&3\end{bmatrix}
\begin{bmatrix}4&4\\-3&3\end{bmatrix}
=\begin{bmatrix}25&7\\7&25\end{bmatrix}.
$$

其标准正交特征向量与特征值为

$$
v_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\quad \lambda_1=32;
\qquad
v_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix},\quad \lambda_2=18.
$$

所以

$$
\sigma_1=4\sqrt2,\qquad \sigma_2=3\sqrt2.
$$

**Step 2：用 $u_i=Av_i/\sigma_i$ 固定符号。**

$$
Av_1=\frac1{\sqrt2}\begin{bmatrix}8\\0\end{bmatrix}
=4\sqrt2\begin{bmatrix}1\\0\end{bmatrix},
$$

所以 $u_1=(1,0)^T$。又

$$
Av_2=\frac1{\sqrt2}\begin{bmatrix}0\\-6\end{bmatrix}
=3\sqrt2\begin{bmatrix}0\\-1\end{bmatrix},
$$

所以 $u_2=(0,-1)^T$。最终

$$
U=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
\Sigma=\begin{bmatrix}4\sqrt2&0\\0&3\sqrt2\end{bmatrix},\quad
V=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}.
$$

检查 $U^TU=V^TV=I$，并直接相乘验证 $U\Sigma V^T=A$。

> [!warning] SVD 的符号配对
> $A^TA$ 的特征向量可乘 $-1$，但 $u_i,v_i$ 的符号必须成对协调，使 $Av_i=\sigma_i u_i$。先随意求 $U$、$V$ 再拼起来，最容易出现一列符号错误。

### 3.5.5 Recitation 例题：$C=\begin{bmatrix}5&5\\-1&7\end{bmatrix}$

$$
C^TC=\begin{bmatrix}26&18\\18&74\end{bmatrix}.
$$

特征多项式

$$
\lambda^2-100\lambda+1600=(\lambda-20)(\lambda-80),
$$

所以奇异值为 $2\sqrt5,4\sqrt5$。按从大到小排列可取

$$
v_1=\frac1{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix},\quad \sigma_1=4\sqrt5;
\qquad
v_2=\frac1{\sqrt{10}}\begin{bmatrix}-3\\1\end{bmatrix},\quad \sigma_2=2\sqrt5.
$$

由 $u_i=Cv_i/\sigma_i$：

$$
u_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\qquad
u_2=\frac1{\sqrt2}\begin{bmatrix}-1\\1\end{bmatrix}.
$$

因此

$$
C=
\frac1{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}
\begin{bmatrix}4\sqrt5&0\\0&2\sqrt5\end{bmatrix}
\frac1{\sqrt{10}}\begin{bmatrix}1&3\\-3&1\end{bmatrix}.
$$

最后一个矩阵是 $V^T$，不是 $V$；这是尺寸和转置检查的重点。

### 3.5.6 Homework：完整题解

> [!question]- Problem 29.1：验证 Fibonacci 矩阵的奇异值
> 对
> $$
> A=\begin{bmatrix}1&1\\1&0\end{bmatrix},
> $$
> 验证
> $$
> \Sigma=\begin{bmatrix}\frac{1+\sqrt5}{2}&0\\0&\frac{\sqrt5-1}{2}\end{bmatrix}.
> $$

> [!success]- 解答
> 因 $A=A^T$，
> $$
> A^TA=A^2=\begin{bmatrix}2&1\\1&1\end{bmatrix}.
> $$
> 特征多项式为
> $$
> \lambda^2-3\lambda+1,
> $$
> 故
> $$
> \lambda_{1,2}=\frac{3\pm\sqrt5}{2}.
> $$
> 奇异值是正平方根。逐项验证：
> $$
> \left(\frac{1+\sqrt5}{2}\right)^2
> =\frac{3+\sqrt5}{2},
> $$
> $$
> \left(\frac{\sqrt5-1}{2}\right)^2
> =\frac{3-\sqrt5}{2}.
> $$
> 二者均非负且按降序排列，因此给出的 $\Sigma$ 正确。

> [!question]- Problem 29.2：已知 $A$ 的列正交，直接写出 SVD
> 设 $A=[w_1\ \cdots\ w_n]$ 的列彼此正交，且 $\|w_i\|=\sigma_i>0$。计算 $A^TA$，并给出 $U,\Sigma,V$。

> [!success]- 解答
> $A^TA$ 的第 $(i,j)$ 项是 $w_i^Tw_j$，故
> $$
> A^TA=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2).
> $$
> 因它已经在标准基下对角化，可取
> $$
> \widetilde U_0=\begin{bmatrix}w_1/\sigma_1&\cdots&w_n/\sigma_n\end{bmatrix},
> \qquad
> \Sigma_0=\operatorname{diag}(\sigma_1,\ldots,\sigma_n),
> \qquad V_0=I
> $$
> 给出薄（thin）、未排序的分解 $A=\widetilde U_0\Sigma_0V_0^T$。这里非零奇异方向满足
> $$
> u_i=\frac{w_i}{\sigma_i}.
> $$
> 这些列已经标准正交。若坚持全文“奇异值按降序”约定，令置换矩阵 $P\in\mathbb R^{n\times n}$ 把 $\sigma_i$ 排成降序，则薄 SVD 可写为
> $$
> \widetilde U=\widetilde U_0P,\qquad
> \Sigma=P^T\Sigma_0P,\qquad
> V=P,
> $$
> 从而 $\widetilde U\Sigma V^T=\widetilde U_0\Sigma_0=A$。若需要完整 SVD 且 $m>n$，最后再把 $\widetilde U$ 补成 $m\times m$ 正交矩阵，并把 $\Sigma$ 补成 $m\times n$。若某列长度为零，则它属于零空间，需要把零奇异值单独处理，不能除以 $\sigma_i=0$。

### 3.5.7 边界、反例与易错点

- 奇异值永远非负；负号应吸收到 $u_i$ 或 $v_i$，不能写进 $\Sigma$。
- $A^TA$ 与 $AA^T$ 的非零特征值相同，都是 $\sigma_i^2$；零特征值个数可能因 $m,n$ 不同而不同。
- 特征值只适用于方阵；SVD 适用于任意矩阵。
- 对称矩阵的奇异值是特征值的绝对值。只有对称正半定时才可直接令 $U=V=Q$、$\Sigma=\Lambda$。
- 重复奇异值对应的奇异向量不唯一，但相应子空间唯一。

### 3.5.8 三道自检题

> [!question]- 自检 1（尺寸）
> $A\in\mathbb R^{4\times2}$ 的完整 SVD 中，$U,\Sigma,V$ 的尺寸分别是什么？

> [!success]- 答案
> $U:4\times4$，$\Sigma:4\times2$，$V:2\times2$。

> [!question]- 自检 2（子空间）
> 哪些奇异向量张成 $N(A)$？

> [!success]- 答案
> 与零奇异值对应的右奇异向量，即 $v_{r+1},\ldots,v_n$。

> [!question]- 自检 3（证明）
> 证明 $\|A\|_2=\max_{\|x\|=1}\|Ax\|=\sigma_1$。

> [!success]- 答案
> 写 $x=Vy$，则 $\|y\|=1$，且
> $$
> \|Ax\|^2=\|U\Sigma y\|^2=\sum_i\sigma_i^2y_i^2\le\sigma_1^2.
> $$
> 取 $x=v_1$ 时等号成立。

### 知识链小结

$$
A^TA=V\Sigma^T\Sigma V^T
\Longrightarrow Av_i=\sigma_i u_i
\Longrightarrow A=U\Sigma V^T
\Longrightarrow\text{四子空间与秩一展开}.
$$

---

## Session 3.6 Linear transformations and their matrices

### 本节问题、前置知识与尺寸

本节要回答：矩阵为什么不是线性变换本身，而是线性变换在选定输入基和输出基下的坐标表示？怎样从基向量的像逐列构造矩阵？

前置知识：向量空间、基、坐标向量、矩阵乘法。设 $T:V\to W$，$\dim V=n$、$\dim W=m$；选定基后，$[T]$ 的尺寸是 $m\times n$。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.6sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S06_Lecture_Lecture_30_Linear_Transformations_and_their_Matrices.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S06_Recitation_Problem_Solving_Linear_Transformations.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.6prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.6sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Linear Transformation|线性变换]]、[[Basis|基]]、[[Change of Basis|换基]]。

### 3.6.1 定义与立刻可推出的性质

> [!definition] 线性变换
> $T:V\to W$ 称为线性变换（linear transformation），若对所有 $v,w\in V$ 与标量 $c,d\in\mathbb F$，
> $$
> T(cv+dw)=cT(v)+dT(w).
> $$

把 $c=d=1$ 得加法保持。要得到齐次性，在定义中令 $d=0$：

$$
T(cv)=cT(v)+0T(w)=cT(v).
$$

再令 $c=0$，得到

$$
T(0)=T(0\cdot v)=0T(v)=0.
$$

所以平移 $T(v)=v+v_0$（$v_0\ne0$）不线性；长度函数 $T(v)=\|v\|$ 也不线性，因为负标量破坏齐次性。

典型线性变换包括：投影、过原点的旋转/反射、微分、积分（在合适函数空间上）、转置，以及任何固定矩阵乘法 $T(x)=Ax$。

### 3.6.2 为什么基向量的像决定整个变换

设 $\mathcal B=(v_1,\ldots,v_n)$ 是 $V$ 的基。任意 $v\in V$ 有唯一展开

$$
v=c_1v_1+\cdots+c_nv_n.
$$

线性性给出

$$
T(v)=c_1T(v_1)+\cdots+c_nT(v_n).
$$

因此只需知道 $n$ 个基向量的像，就知道 $T$ 在所有向量上的作用。

再选 $W$ 的基 $\mathcal C=(w_1,\ldots,w_m)$，把每个像展开：

$$
T(v_j)=a_{1j}w_1+\cdots+a_{mj}w_m.
$$

定义

$$
[T]_{\mathcal C\leftarrow\mathcal B}
=\begin{bmatrix}
a_{11}&\cdots&a_{1n}\\
\vdots&&\vdots\\
a_{m1}&\cdots&a_{mn}
\end{bmatrix}.
$$

第 $j$ 列正是 $[T(v_j)]_{\mathcal C}$。于是对任意 $v$，

$$
[T(v)]_{\mathcal C}
=[T]_{\mathcal C\leftarrow\mathcal B}[v]_{\mathcal B}.
$$

这个箭头记号明确区分输入基与输出基，可以避免“到底乘哪个换基矩阵”的混乱。

### 3.6.3 课件例子

#### 例 1：逆时针旋转 $45^\circ$

Lecture 30 从旋转说明“矩阵的列是基向量的像”。标准基下

$$
R=\frac1{\sqrt2}
\begin{bmatrix}
1&-1\\
1&1
\end{bmatrix}.
$$

第一列是 $R e_1=(1/\sqrt2,1/\sqrt2)^T$，第二列是 $R e_2=(-1/\sqrt2,1/\sqrt2)^T$；它们正是两个标准基向量旋转 $45^\circ$ 后的坐标。

#### 补充例：反射

标准基下

$$
A=\begin{bmatrix}1&0\\0&-1\end{bmatrix}
$$

把 $(x,y)$ 映到 $(x,-y)$，即关于 $x$ 轴反射。两个标准基向量分别映到 $e_1$ 与 $-e_2$，恰好形成矩阵两列。

#### 例 2：投影到 $45^\circ$ 直线

令 $a=(1,1)^T$。标准基下正交投影矩阵是

$$
P=\frac{aa^T}{a^Ta}
=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

若改用标准正交基

$$
q_1=\frac1{\sqrt2}(1,1)^T,\qquad q_2=\frac1{\sqrt2}(1,-1)^T,
$$

同一个变换的矩阵变成

$$
\begin{bmatrix}1&0\\0&0\end{bmatrix}.
$$

变换没变，只有坐标变简单了。

#### 例 3：微分算子

令 $V=P_2$ 为次数不超过 $2$ 的多项式空间，输入基 $(1,x,x^2)$；输出落在 $P_1$，取基 $(1,x)$。微分算子 $D(p)=p'$ 满足

$$
D(1)=0,\qquad D(x)=1,\qquad D(x^2)=2x.
$$

故

$$
[D]_{(1,x)\leftarrow(1,x,x^2)}
=\begin{bmatrix}0&1&0\\0&0&2\end{bmatrix}.
$$

尺寸 $2\times3$ 与“输入维数 3、输出维数 2”完全一致。

### 3.6.4 复合、逆、核与像

若 $T_1:V\to W$、$T_2:W\to Z$，并且中间空间使用同一基，则

$$
[T_2\circ T_1]=[T_2][T_1].
$$

顺序与函数复合相反：先做 $T_1$，矩阵写在右边。

若 $T:V\to W$ 双射，则 $\dim V=\dim W$，矩阵方阵可逆，而且

$$
[T^{-1}]=[T]^{-1}.
$$

在固定基下，

$$
N([T])\leftrightarrow\ker T,\qquad C([T])\leftrightarrow\operatorname{im}T.
$$

秩—零度定理因此可写成

$$
\dim\ker T+\dim\operatorname{im}T=\dim V.
$$

### 3.6.5 Recitation：转置算子在两组基下

在 $M_{2\times2}(\mathbb R)$ 上定义 $T(A)=A^T$。因为

$$
(A+B)^T=A^T+B^T,\qquad (cA)^T=cA^T,
$$

所以 $T$ 线性；又 $(A^T)^T=A$，故 $T^{-1}=T$。

标准基

$$
E_{11},E_{12},E_{21},E_{22}
$$

下，$T(E_{11})=E_{11}$、$T(E_{12})=E_{21}$、$T(E_{21})=E_{12}$、$T(E_{22})=E_{22}$，所以

$$
[T]_{\text{std}}=
\begin{bmatrix}
1&0&0&0\\
0&0&1&0\\
0&1&0&0\\
0&0&0&1
\end{bmatrix}.
$$

改用三个对称矩阵加一个斜对称矩阵组成的基

$$
w_1=E_{11},\quad w_2=E_{22},\quad
w_3=E_{12}+E_{21},\quad w_4=E_{12}-E_{21},
$$

则前三个满足 $T(w_i)=w_i$，最后一个满足 $T(w_4)=-w_4$，故

$$
[T]_{\mathcal W}=\operatorname{diag}(1,1,1,-1).
$$

这也揭示了 $M_{2\times2}$ 分解为对称子空间与斜对称子空间。

### 3.6.6 Homework：完整题解

> [!question]- Problem 30.1：极坐标中的径向放大是否线性？
> $T(r,\theta)=(2r,\theta)$：判断线性，改写为直角坐标，求矩阵。

> [!success]- 解答
> 保持方向并把到原点距离乘 $2$，在向量语言中就是 $T(v)=2v$。因此
> $$
> T(v+w)=2(v+w)=T(v)+T(w),\qquad T(cv)=2cv=cT(v),
> $$
> 所以线性。
>
> 直角坐标下 $T(x,y)=(2x,2y)$。长度检查：
> $$
> \|(2x,2y)\|=\sqrt{4x^2+4y^2}=2\sqrt{x^2+y^2}.
> $$
> 标准基下矩阵为
> $$
> [T]=\begin{bmatrix}2&0\\0&2\end{bmatrix}=2I.
> $$
> 极坐标表达看似非线性，是因为 $(r,\theta)$ 不是向量空间的线性坐标；判断线性必须回到向量加法和数乘。

> [!question]- Problem 30.2：给出固定零向量但不线性的变换

> [!success]- 解答
> 例如
> $$
> T(x,y)=(x,y^2).
> $$
> 它满足 $T(0,0)=(0,0)$，但通常
> $$
> T(c(x,y))=(cx,c^2y^2)\ne(cx,cy^2)=cT(x,y).
> $$
> 取 $c=2,y=1$ 即可看出失败。另一个例子是 $T(x,y)=(x,|y|)$：它保持零，却不满足 $T(-v)=-T(v)$。

### 3.6.7 边界、反例与易错点

- $T(0)=0$ 是线性的必要条件，不是充分条件。
- “矩阵的列是基向量的像”只有在输入基、输出基都已说明时才完整。
- 同一个线性变换在不同基下矩阵不同；反过来，同一矩阵若作用在不同坐标约定下，也可能代表不同几何变换。
- 仿射映射 $x\mapsto Ax+b$ 在 $b\ne0$ 时不是线性映射，但可在增广坐标中表示。

### 3.6.8 三道自检题

> [!question]- 自检 1（判断）
> $T(f)=f(0)$ 从 $P_2$ 到 $\mathbb R$ 是否线性？标准基下矩阵是什么？

> [!success]- 答案
> 线性；$T(1)=1,T(x)=T(x^2)=0$，故矩阵是 $[1\ 0\ 0]$。

> [!question]- 自检 2（列构造）
> 若 $T(e_1)=(1,2)^T$、$T(e_2)=(-1,3)^T$，标准基矩阵是什么？

> [!success]- 答案
> $$
> [T]=\begin{bmatrix}1&-1\\2&3\end{bmatrix}.
> $$

> [!question]- 自检 3（复合）
> $T_1$ 的矩阵为 $A$，$T_2$ 的矩阵为 $B$，先做 $T_1$ 再做 $T_2$ 的矩阵是什么？

> [!success]- 答案
> $BA$，前提是 $A$ 的输出坐标基与 $B$ 的输入坐标基一致。

### 知识链小结

$$
\text{线性}\Longrightarrow\text{基向量的像决定全部}
\Longrightarrow [T]_{\mathcal C\leftarrow\mathcal B}
\Longrightarrow\text{矩阵乘法表示复合}.
$$

---

## Session 3.7 Change of basis and image compression

### 本节问题、前置知识与尺寸

本节要回答：同一个向量如何在两组基之间换坐标？同一个线性算子的矩阵为何通过相似变换改变？为什么换到 Fourier、wavelet 或 SVD 基后，图像可以只保留少量系数？

前置知识：基与坐标、线性变换、相似矩阵、正交矩阵、SVD。以下先在 $n$ 维空间讨论方阵换基，再说明压缩中的高维向量和低秩矩阵。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.7sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S07_Lecture_Lecture_31_Change_of_Basis_Image_Compression.pdf#page=1|Lecture transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S07_Recitation_Problem_Solving_Change_of_Basis.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.7prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.7sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Change of Basis|换基]]、[[Similar Matrix|相似矩阵]]、[[Singular Value Decomposition|奇异值分解]]、[[Fourier Series|Fourier 展开]]。

### 3.7.1 向量换基：先说明矩阵的方向

设旧基为 $\mathcal E=(e_1,\ldots,e_n)$，新基为 $\mathcal W=(w_1,\ldots,w_n)$。把新基向量用旧基坐标作列，得到

$$
W=\begin{bmatrix}[w_1]_{\mathcal E}&\cdots&[w_n]_{\mathcal E}\end{bmatrix}.
$$

若 $c=[x]_{\mathcal W}$，则

$$
x=c_1w_1+\cdots+c_nw_n,
$$

所以旧基坐标满足

$$
[x]_{\mathcal E}=W[x]_{\mathcal W}. \tag{1}
$$

反向为

$$
[x]_{\mathcal W}=W^{-1}[x]_{\mathcal E}. \tag{2}
$$

式 (1)–(2) 对任意两组基都成立；但仅知道几何基 $\mathcal W$ 标准正交，并不能在任意旧基 $\mathcal E$ 的坐标中推出 $W^{-1}=W^T$。若 $\mathcal E$ 与 $\mathcal W$ 都是在同一个内积下的标准正交基，则坐标矩阵 $W$ 为 orthogonal（复数域为 unitary），此时

$$
W^{-1}=W^T\quad\text{（复数域为 }W^{-1}=W^*\text{）}.
$$

不依赖旧坐标选择的内在说法是：只要 $\mathcal W$ 本身为标准正交基，向量的 $\mathcal W$-坐标就由

$$
c_i=\langle w_i,x\rangle
$$

给出；在标准实坐标中是 $c_i=w_i^Tx$，在标准复坐标中是 $c_i=w_i^*x$。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-change-of-basis.png|820]]

> [!warning] 主动变换与被动换基
> $Wx$ 若把 $W$ 看作一个线性算子，是主动地移动向量；式 (1) 中 $W$ 是坐标翻译器，几何向量并未改变。两者数值公式可能相同，含义不同。

### 3.7.2 线性算子的换基公式

设同一个 $T:V\to V$ 在旧基 $\mathcal E$ 下矩阵为 $A$，在新基 $\mathcal W$ 下矩阵为 $B$。从新坐标 $c=[x]_{\mathcal W}$ 出发：

1. 用 $Wc$ 转为旧坐标；
2. 用 $A$ 作用，得到 $AWc$；
3. 用 $W^{-1}$ 转回新坐标。

因此

$$
Bc=W^{-1}AWc
$$

对所有 $c$ 成立，即

$$
B=W^{-1}AW.
$$

这就是相似变换的坐标意义。若 $W$ 的列是 $A$ 的特征向量，则 $B=\Lambda$；若列是广义特征向量，则 $B=J$。

### 3.7.3 图像为什么是向量，压缩为什么是换基

一幅 $512\times512$ 灰度图可以看成 $\mathbb R^{512^2}$ 中的向量：每个标准基向量只改变一个像素。标准基适合记录，却不适合压缩，因为自然图像的相邻像素往往高度相关。

换到 Fourier、离散余弦或 wavelet 基后，平滑区域、边缘和局部细节会集中到少数系数：

$$
x=Wc,\qquad c=W^{-1}x.
$$

压缩流程是

$$
x\xrightarrow{W^{-1}}c
\xrightarrow{\text{小系数置零/量化}}\widehat c
\xrightarrow{W}\widehat x.
$$

好的压缩基同时满足：

1. $W$ 与 $W^{-1}$ 乘法快；
2. 典型信号的系数稀疏或快速衰减；
3. 丢弃小系数造成的视觉误差可控。

Fourier 基擅长全局频率；Haar wavelet 同时具有尺度和位置局部性。JPEG 实际主要使用分块离散余弦变换，而不是直接使用复 Fourier 矩阵；课程用 Fourier 思想解释其结构。

### 3.7.4 课程外补充：SVD 的[[Low-Rank Approximation|低秩近似与压缩]]

> [!note] 与 Lecture 31 的边界
> 官方 Lecture 31 的图像压缩主线是 Fourier/Haar 等固定基中的坐标稀疏化。下面的 Eckart–Young 低秩近似是基于 SVD 的自然延伸，用来补充“矩阵图像”的另一种压缩视角，不把它误记为本讲采用的证明或算法。

若把灰度图本身看成矩阵 $A\in\mathbb R^{m\times n}$，SVD 给出

$$
A=\sum_{i=1}^r\sigma_i u_iv_i^T.
$$

对 $0\le k<r$，保留前 $k$ 项得到秩至多 $k$ 的近似

$$
A_k=\sum_{i=1}^k\sigma_i u_iv_i^T.
$$

Eckart–Young 定理说明，在所有秩至多 $k$ 的矩阵中，$A_k$ 同时最小化谱范数误差与 Frobenius 范数误差：

$$
\|A-A_k\|_2=\sigma_{k+1},
$$

$$
\|A-A_k\|_F^2=\sum_{i=k+1}^r\sigma_i^2.
$$

若 $k\ge r$，则 $A_k=A$，上述两种误差都为零。Eckart–Young 的完整极小化证明通常放在后续数值线性代数课程；这里把结论作为拓展而不冒充 Lecture 31 的内容。存储 $A_k$ 约需 $k(m+n+1)$ 个数；只有当它显著小于 $mn$ 时才真正节省空间。

### 3.7.5 Recitation：插值基、换基与微分矩阵

在 $P_2$ 中，设 $w_1,w_2,w_3$ 满足

$$
w_j(-1)=\delta_{1j},\qquad w_j(0)=\delta_{2j},\qquad w_j(1)=\delta_{3j}.
$$

也就是说，$w_1,w_2,w_3$ 是节点 $-1,0,1$ 的 Lagrange 基。

#### (a) 不显式求基也能求坐标

若 $y(x)=-x+5$，则

$$
y(-1)=6,\qquad y(0)=5,\qquad y(1)=4.
$$

对 $y=\alpha w_1+\beta w_2+\gamma w_3$ 在三个节点取值，立刻得

$$
[y]_{\mathcal W}=\begin{bmatrix}6\\5\\4\end{bmatrix},
\qquad y=6w_1+5w_2+4w_3.
$$

#### (b) 两个换基矩阵

对 $p(x)=a+bx+cx^2$，其新基坐标就是三点函数值：

$$
[p]_{\mathcal W}
=\begin{bmatrix}p(-1)\\p(0)\\p(1)\end{bmatrix}
=E\begin{bmatrix}a\\b\\c\end{bmatrix},
$$

其中

$$
E=\begin{bmatrix}
1&-1&1\\
1&0&0\\
1&1&1
\end{bmatrix}.
$$

反向矩阵为

$$
E^{-1}=\begin{bmatrix}
0&1&0\\
-\frac12&0&\frac12\\
\frac12&-1&\frac12
\end{bmatrix}.
$$

它的列给出

$$
w_1=-\frac12x+\frac12x^2,\quad
w_2=1-x^2,\quad
w_3=\frac12x+\frac12x^2.
$$

#### (c) 微分算子的换基

在标准基 $(1,x,x^2)$ 下，若输出仍放在 $P_2$ 中，

$$
D_{\mathcal E}=\begin{bmatrix}0&1&0\\0&0&2\\0&0&0\end{bmatrix}.
$$

在 $\mathcal W$ 下：

$$
D_{\mathcal W}=ED_{\mathcal E}E^{-1}
=\begin{bmatrix}
-\frac32&2&-\frac12\\
-\frac12&0&\frac12\\
\frac12&-2&\frac32
\end{bmatrix}.
$$

这个三矩阵乘法恰好对应“新坐标 → 旧坐标 → 微分 → 新坐标”。

### 3.7.6 Homework：完整题解

> [!question]- Problem 31.1：验证 Haar wavelet 基正交并归一化

> [!success]- 解答
> 课程给出的 $\mathbb R^8$ Haar 向量可写为
> $$
> \begin{aligned}
> h_0&=(1,1,1,1,1,1,1,1),\\
> h_1&=(1,1,1,1,-1,-1,-1,-1),\\
> h_2&=(1,1,-1,-1,0,0,0,0),\\
> h_3&=(0,0,0,0,1,1,-1,-1),\\
> h_4&=(1,-1,0,0,0,0,0,0),\\
> h_5&=(0,0,1,-1,0,0,0,0),\\
> h_6&=(0,0,0,0,1,-1,0,0),\\
> h_7&=(0,0,0,0,0,0,1,-1).
> \end{aligned}
> $$
> 不同尺度的向量要么支撑不相交，内积为零；要么一个向量在另一个向量的支撑上拥有相同数量的 $+1$、$-1$，相消为零。因此两两正交。
>
> 长度为
> $$
> \|h_0\|=\|h_1\|=\sqrt8,\quad
> \|h_2\|=\|h_3\|=2,\quad
> \|h_4\|=\cdots=\|h_7\|=\sqrt2.
> $$
> 所以标准正交基是
> $$
> \frac{h_0}{\sqrt8},\ \frac{h_1}{\sqrt8},
> \frac{h_2}{2},\ \frac{h_3}{2},\
> \frac{h_4}{\sqrt2},\ldots,\frac{h_7}{\sqrt2}.
> $$
> 八个非零正交向量自动线性无关，在八维空间中因此构成基。

> [!question]- Problem 31.2：为 $M_{2\times2}(\mathbb R)$ 给出两组基，并比较用途

> [!success]- 解答
> 最直接的标准基是
> $$
> E_{11}=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad
> E_{12}=\begin{bmatrix}0&1\\0&0\end{bmatrix},\quad
> E_{21}=\begin{bmatrix}0&0\\1&0\end{bmatrix},\quad
> E_{22}=\begin{bmatrix}0&0\\0&1\end{bmatrix}.
> $$
> 第二组可选
> $$
> B_1=I,\quad
> B_2=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad
> B_3=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
> B_4=\begin{bmatrix}0&1\\-1&0\end{bmatrix}.
> $$
> 把矩阵按条目摊平成 $\mathbb R^4$ 向量，可见两组都线性无关，故都是基。
>
> - 描述对角矩阵时，标准基只需 $E_{11},E_{22}$；第二组只需 $B_1,B_3$，两者都方便。
> - 描述上/下三角矩阵时，标准基最直接，只需三个相应基向量。
> - 描述对称矩阵时，第二组更好：对称矩阵恰好是 $\operatorname{span}(B_1,B_2,B_3)$，斜对称分量 $B_4$ 的系数为零。
>
> “哪组基更好”没有绝对答案；标准是目标对象能否用少数、易计算的坐标表示。

### 3.7.7 边界、反例与易错点

- 写换基公式前先写清“矩阵的列是哪组基用哪组坐标表示”，不要只背 $W$ 或 $W^{-1}$。
- 若 $W$ 的列是新基在旧基中的坐标，则 $W^{-1}=W^T$ 要求这个坐标矩阵本身正交；例如新旧两组基都标准正交。只说“新基在几何上标准正交”而旧基任意，还不够。
- 阈值置零是有损压缩；可逆换基本身不损失信息。
- SVD 低秩近似对给定矩阵最优，但计算完整 SVD 可能昂贵；JPEG 使用固定快速基而不是每张小块都求全局 SVD。

### 3.7.8 三道自检题

> [!question]- 自检 1（方向）
> $W$ 的列是新基向量的旧坐标。已知旧坐标 $x$，怎样求新坐标？

> [!success]- 答案
> $[x]_{\mathcal W}=W^{-1}[x]_{\mathcal E}$。

> [!question]- 自检 2（算子换基）
> 旧基矩阵是 $A$，新基矩阵列组成 $W$。新基下算子矩阵是什么？

> [!success]- 答案
> $W^{-1}AW$。

> [!question]- 自检 3（误差）
> 截断 SVD 到秩 $k$ 后，谱范数误差是多少？

> [!success]- 答案
> $\sigma_{k+1}$；若 $k\ge r$，误差为零。

### 知识链小结

$$
x=Wc\Longrightarrow c=W^{-1}x
\Longrightarrow B=W^{-1}AW
\Longrightarrow\text{选择稀疏/低秩坐标进行压缩}.
$$

---

## Session 3.8 Left and right inverses and pseudoinverse

### 本节问题、前置知识与尺寸

本节要回答：长方矩阵何时有左逆或右逆？秩亏时怎样利用 SVD 定义在行空间—列空间之间真正可逆的部分？伪逆为何同时给出最小二乘解和最小范数解？

前置知识：秩、四个基本子空间、最小二乘、SVD。始终设 $A\in\mathbb R^{m\times n}$、$\operatorname{rank}(A)=r$。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf#page=1|Summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S08_Lecture_Lecture_33_Left_and_Right_Inverses_Pseudoinverse.pdf#page=1|Lecture 33 transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S08_Recitation_Problem_Solving_Pseudoinverses.pdf#page=1|Recitation p.1]] · [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses3.8prob.pdf#page=1|Homework p.1]] · [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses3.8sol.pdf#page=1|Official solution p.1]]

关联卡片：[[Left Inverse|左逆]]、[[Right Inverse|右逆]]、[[Pseudoinverse|伪逆]]、[[Least Squares|最小二乘]]、[[Orthogonal Projection|正交投影]]。

### 3.8.1 左逆与满列秩

若存在 $L\in\mathbb R^{n\times m}$ 使

$$
LA=I_n,
$$

则 $L$ 是 $A$ 的左逆。若 $Ax=0$，左乘 $L$ 得 $x=0$，所以 $N(A)=\{0\}$，即 $r=n$；因此必须 $m\ge n$。

反之，若 $r=n$，则 $A^TA$ 正定可逆，并且

$$
L=(A^TA)^{-1}A^T
$$

满足

$$
LA=(A^TA)^{-1}A^TA=I_n.
$$

所以

$$
A\text{ 有左逆}\iff\operatorname{rank}(A)=n.
$$

此时 $Ax=b$ 至多一个解，但 $b\notin C(A)$ 时无解。$AL=A(A^TA)^{-1}A^T$ 通常不是 $I_m$，而是投影到 $C(A)$。

### 3.8.2 右逆与满行秩

若存在 $R\in\mathbb R^{n\times m}$ 使

$$
AR=I_m,
$$

则 $R$ 是 $A$ 的右逆。这说明每个 $b\in\mathbb R^m$ 都等于 $A(Rb)$，故 $C(A)=\mathbb R^m$，即 $r=m$；因此必须 $n\ge m$。

反之，若 $r=m$，则 $AA^T$ 正定可逆，且

$$
R=A^T(AA^T)^{-1}
$$

满足 $AR=I_m$。所以

$$
A\text{ 有右逆}\iff\operatorname{rank}(A)=m.
$$

此时 $Ax=b$ 对每个 $b$ 都可解，但若 $n>m$，因为 $\dim N(A)=n-m>0$，解不唯一。

> [!note] 双边逆为何只属于满秩方阵
> 若同一个 $A$ 同时有左逆和右逆，则 $n\le m$ 且 $m\le n$，所以 $m=n=r$；此时左右逆相等并等于通常的 $A^{-1}$。

### 3.8.3 SVD 定义伪逆

设完整 SVD 为

$$
A=U\Sigma V^T,\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_r,0,\ldots)
$$

（按 $m\times n$ 尺寸理解）。定义 $\Sigma^+\in\mathbb R^{n\times m}$：把每个非零 $\sigma_i$ 换成 $1/\sigma_i$，转置矩形形状，零仍保留为零。Moore–Penrose 伪逆为

$$
A^+=V\Sigma^+U^T.
$$

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit3-pseudoinverse.png|840]]

它的作用是

$$
u_i\mapsto \frac1{\sigma_i}v_i\quad(i\le r),
$$

并把 $N(A^T)$ 中的分量映为零。于是

$$
A^+A=V
\begin{bmatrix}I_r&0\\0&0\end{bmatrix}V^T
$$

是投影到 $C(A^T)$（行空间）的正交投影，而

$$
AA^+=U
\begin{bmatrix}I_r&0\\0&0\end{bmatrix}U^T
$$

是投影到 $C(A)$ 的正交投影。

#### 为什么行空间到列空间的映射可逆

限制映射

$$
A:C(A^T)\to C(A)
$$

是双射。先证明满射：任取 $y\in C(A)$，由列空间定义存在 $z\in\mathbb R^n$ 使 $y=Az$。利用基本子空间的正交直和

$$
\mathbb R^n=C(A^T)\oplus N(A),
$$

唯一分解 $z=z_r+z_0$，其中 $z_r\in C(A^T)$、$z_0\in N(A)$。于是

$$
y=Az=A z_r+A z_0=A z_r,
$$

所以 $y$ 确实有一个来自定义域 $C(A^T)$ 的原像；这才完成满射证明。

再证明单射：若 $x,y\in C(A^T)$ 且 $Ax=Ay$，则

$$
A(x-y)=0,
$$

所以 $x-y\in N(A)$；同时行空间对子法封闭，故 $x-y\in C(A^T)$。而

$$
C(A^T)\perp N(A),
$$

两者交集只有零，因此 $x-y=0$。伪逆就是这个限制映射的逆，并在左零空间上定义为零。

### 3.8.4 Moore–Penrose 四条件

$A^+$ 是唯一满足下列四式的矩阵：

$$
AA^+A=A,\qquad A^+AA^+=A^+,
$$

$$
(AA^+)^T=AA^+,\qquad (A^+A)^T=A^+A.
$$

前两式说明在可逆部分上来回不会改变；后两式保证两个乘积是正交投影，而不是任意斜投影。用 SVD 代入后，四式都归结为对角矩阵 $\Sigma,\Sigma^+$ 的逐项关系。

### 3.8.5 伪逆同时解决两类“最佳解”

对任意 $b\in\mathbb R^m$，令

$$
\hat x=A^+b.
$$

则

$$
A\hat x=AA^+b=P_{C(A)}b,
$$

所以残差 $b-A\hat x$ 与 $C(A)$ 正交，$\hat x$ 是最小二乘解。若最小二乘解不唯一，所有解相差一个 $z\in N(A)$；而 $\hat x\in C(A^T)$ 且 $C(A^T)\perp N(A)$，故

$$
\|\hat x+z\|^2=\|\hat x\|^2+\|z\|^2\ge\|\hat x\|^2.
$$

所以 $A^+b$ 还是所有最小二乘解中范数最小者。

特殊情况：

$$
A^+=(A^TA)^{-1}A^T\quad\text{若 }r=n,
$$

$$
A^+=A^T(AA^T)^{-1}\quad\text{若 }r=m,
$$

$$
A^+=A^{-1}\quad\text{若 }r=m=n.
$$

### 3.8.6 Recitation：$A=[1\ 2]$

$A\in\mathbb R^{1\times2}$，$AA^T=[5]$，所以它满行秩并且

$$
A^+=A^T(AA^T)^{-1}
=\frac15\begin{bmatrix}1\\2\end{bmatrix}.
$$

两个乘积为

$$
AA^+=1,
$$

$$
A^+A=\frac15\begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

后者投影到 $C(A^T)=\operatorname{span}((1,2)^T)$：

- 若 $x\in N(A)=\operatorname{span}((-2,1)^T)$，则 $A^+Ax=0$；
- 若 $x\in C(A^T)$，则 $A^+Ax=x$。

这里右下角必须是 $4/5$；任何写成 $1/5$ 的版本都会使投影矩阵不幂等，是可通过 $P^2=P$ 发现的算术错误。

### 3.8.7 Homework：完整题解

> [!question]- Problem 32.1：求一个右逆
> 对
> $$
> A=\begin{bmatrix}1&0&1\\0&1&0\end{bmatrix},
> $$
> 求右逆。

> [!success]- 解答
> $A$ 的两行独立，满行秩 $r=m=2$，可用
> $$
> R=A^T(AA^T)^{-1}.
> $$
> 逐步计算
> $$
> A^T=\begin{bmatrix}1&0\\0&1\\1&0\end{bmatrix},\qquad
> AA^T=\begin{bmatrix}2&0\\0&1\end{bmatrix},
> $$
> $$
> (AA^T)^{-1}=\begin{bmatrix}1/2&0\\0&1\end{bmatrix}.
> $$
> 因而
> $$
> R=\begin{bmatrix}1/2&0\\0&1\\1/2&0\end{bmatrix}.
> $$
> 验算：
> $$
> AR=\begin{bmatrix}1&0\\0&1\end{bmatrix}=I_2.
> $$
> 右逆不唯一；例如改变第一列为 $(t,0,1-t)^T$ 仍可得到第一标准基向量。

> [!question]- Problem 32.2：秩一方阵的左右逆与伪逆
> 对
> $$
> A=\begin{bmatrix}4&3\\8&6\end{bmatrix},
> $$
> 判断并求适当的逆。

> [!success]- 解答
> 第二行是第一行的两倍，故 $\operatorname{rank}(A)=1<2$、$\det A=0$。它既非满列秩，也非满行秩，所以没有左逆或右逆。若存在 $AB=I$ 或 $BA=I$，取行列式会得到 $0=1$，矛盾。
>
> 伪逆总存在。把 $A$ 写成秩一外积：
> $$
> A=\begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}4&3\end{bmatrix}.
> $$
> 两向量长度分别是 $\sqrt5$、$5$，故唯一非零奇异值为 $5\sqrt5=\sqrt{125}$，并可取
> $$
> u_1=\frac1{\sqrt5}\begin{bmatrix}1\\2\end{bmatrix},\qquad
> v_1=\frac15\begin{bmatrix}4\\3\end{bmatrix}.
> $$
> 因此
> $$
> A^+=\frac1{\sigma_1}v_1u_1^T
> =\frac1{125}\begin{bmatrix}4&8\\3&6\end{bmatrix}.
> $$
> 检查投影：
> $$
> AA^+=\frac15\begin{bmatrix}1&2\\2&4\end{bmatrix}=P_{C(A)},
> $$
> $$
> A^+A=\frac1{25}\begin{bmatrix}16&12\\12&9\end{bmatrix}=P_{C(A^T)}.
> $$
> 两者均对称且幂等，符合 Moore–Penrose 条件。

### 3.8.8 边界、反例与易错点

- 左逆要求满列秩，右逆要求满行秩；记忆时看单位矩阵尺寸：$LA=I_n$、$AR=I_m$。
- $(A^TA)^{-1}A^T$ 只有在 $A$ 满列秩时才存在，不能拿来定义所有矩阵的伪逆。
- 小奇异值取倒数会放大噪声；理论伪逆存在不代表数值计算稳定。实际常用截断 SVD 或正则化。
- $A^+A$ 与 $AA^+$ 通常不是单位矩阵，而是两个不同空间上的正交投影。

### 3.8.9 三道自检题

> [!question]- 自检 1（存在性）
> $A\in\mathbb R^{3\times5}$ 能有左逆吗？能有右逆吗？

> [!success]- 答案
> 不可能有左逆，因为秩至多 $3<5$；若秩为 $3$，则有右逆。

> [!question]- 自检 2（投影）
> $A^+A$ 投影到哪里？其零空间是什么？

> [!success]- 答案
> 投影到行空间 $C(A^T)$，零空间为 $N(A)$。

> [!question]- 自检 3（最小范数）
> 为什么 $A^+b$ 在所有最小二乘解中范数最小？

> [!success]- 答案
> $A^+b\in C(A^T)$；其他最小二乘解为 $A^+b+z$、$z\in N(A)$。两空间正交，所以平方范数增加 $\|z\|^2$。

### 知识链小结

$$
\text{满列秩}\to\text{左逆},\qquad
\text{满行秩}\to\text{右逆},
$$

$$
A=U\Sigma V^T\Longrightarrow A^+=V\Sigma^+U^T
\Longrightarrow\text{两个投影、最小二乘与最小范数}.
$$

---

## Session 3.9 Exam 3 review

### 本节问题、官方范围与资料

Session 3.9 对应 Lecture 32 的 Exam 3 review。官方明确说明：**Exam 3 的主要范围截至 Session 3.5 SVD**；Session 3.6 线性变换、3.7 换基/压缩和 3.8 伪逆主要进入 Final Exam。不过前三单元知识会交叉，因此复习必须能把早先的特征值、微分方程、投影和 Markov 矩阵与本单元结构连接起来。

资料入口：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.9sum.pdf#page=1|Review summary p.1]] · [[MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S09_Lecture_Exam_3_Review.pdf#page=1|Lecture 32 transcript p.1]] · [[MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S09_Recitation_Exam_3_Problem_Solving.pdf#page=1|Review recitation p.1]]

### 3.9.1 一张表完成题目分流

| 看到的结构 | 立刻检查 | 首选表示 |
|---|---|---|
| $A=A^T$ | 实特征值、正交特征基 | $A=Q\Lambda Q^T$ |
| $A\succ0$ | $x^TAx$、特征值、主元、顺序主子式 | $Q\Lambda Q^T$ 或 $LDL^T$ |
| $A^T=-A$ | 纯虚或零特征值、长度保持流 | unitary 谱分解、$e^{tA}$ |
| $B=M^{-1}AM$ | 相同谱、Jordan 块、幂与指数 | 相似标准形 |
| 任意 $A_{m\times n}$ | $A^TA$、秩、四子空间 | $A=U\Sigma V^T$ |
| 投影 $P^2=P$ | 特征值 $0,1$ | 正交投影时 $P=P^T$ |
| $u'=Au$ | $e^{tA}$、特征值实部、Jordan 块 | $u(t)=e^{tA}u(0)$ |

### 3.9.2 Review Problem 1：斜对称微分方程

设

$$
u'=Au,\qquad
A=\begin{bmatrix}0&-1&0\\1&0&-1\\0&1&0\end{bmatrix},\qquad A^T=-A.
$$

#### (a) 特征值与解的形态

$$
\det(A-\lambda I)=-\lambda^3-2\lambda
=-\lambda(\lambda^2+2),
$$

故

$$
\lambda_1=0,\qquad \lambda_{2,3}=\pm\sqrt2\,i.
$$

可取特征向量

$$
x_1=\begin{bmatrix}1\\0\\1\end{bmatrix},\quad
x_2=\begin{bmatrix}-1\\\sqrt2\,i\\1\end{bmatrix},\quad
x_3=\begin{bmatrix}-1\\-\sqrt2\,i\\1\end{bmatrix}
=\overline{x_2}.
$$

所以复形式通解为

$$
u(t)=c_1x_1+c_2e^{\sqrt2it}x_2+c_3e^{-\sqrt2it}x_3.
$$

因 $A$ 实且初值实，取 $c_1\in\mathbb R$、$c_3=\overline{c_2}$，则后两项互为共轭：

$$
c_2e^{\sqrt2it}x_2+c_3e^{-\sqrt2it}x_3
=2\operatorname{Re}\!\left(c_2e^{\sqrt2it}x_2\right),
$$

所以最终 $u(t)$ 为实。这里特意选择 $x_3=\overline{x_2}$，避免把特征向量额外乘 $-1$ 后遗漏系数中的负号。

#### (b) 周期与有界性

非零频率为 $\sqrt2$，因此所有解都满足共同周期

$$
T=\frac{2\pi}{\sqrt2}=\pi\sqrt2.
$$

只要解含非零振荡分量，这也是它的基本周期；若初值完全落在零特征空间，解为常向量，只能说任意正数都是周期，并不存在最小的正基本周期。

这里 $A$ 是实斜对称，亦是[[Normal Matrix|normal matrix]]；它可 unitary 对角化，没有非平凡 Jordan 块，所以纯虚谱确实给出有界振荡。不能把这个结论推广到带 Jordan 块的任意纯虚谱矩阵。

#### (c) 正交性与共轭

斜 Hermitian 矩阵 $A^*=-A$ 的不同特征值对应特征向量在 Hermitian 内积下正交。计算复向量内积时必须使用 $x_i^*x_j$。对上面选取的向量逐项验算：

$$
x_1^*x_2=-1+0+1=0,
\qquad
x_1^*x_3=-1+0+1=0,
$$

并且

$$
x_2^*x_3
=(-1)(-1)+(-\sqrt2\,i)(-\sqrt2\,i)+1\cdot1
=1-2+1=0.
$$

这完成了 Review Problem 1(c) 要求的显式验证，而不只引用一般定理。

#### (d) 计算 $e^{tA}$

若 $A=S\Lambda S^{-1}$，则

$$
e^{tA}=Se^{t\Lambda}S^{-1},\qquad
e^{t\Lambda}=\operatorname{diag}(1,e^{\sqrt2it},e^{-\sqrt2it}).
$$

另一个结构检查是：

$$
\frac{d}{dt}\|u(t)\|^2
=u^T(A^T+A)u=0,
$$

所以流保持长度。

### 3.9.3 Review Problem 2：由谱数据识别矩阵类别

已知 $3\times3$ 矩阵 $A$ 的特征值为 $0,c,2$，对应特征向量

$$
x_1=\begin{bmatrix}1\\1\\1\end{bmatrix},\quad
x_2=\begin{bmatrix}1\\-1\\0\end{bmatrix},\quad
x_3=\begin{bmatrix}1\\1\\-2\end{bmatrix}.
$$

三向量两两正交，因而一定独立。

1. **何时可对角化？** 对所有 $c$ 都可，因为已经给出三个独立特征向量，即使特征值重合也不影响这组基的独立性。
2. **何时对称？** 若 $c\in\mathbb R$，把特征向量归一化组成正交 $Q$，则 $A=Q\operatorname{diag}(0,c,2)Q^T$ 对称。若 $c$ 非实，则不可能是实对称矩阵。
3. **何时正定？** 永不正定，因为固定有零特征值。$c\ge0$ 时是正半定。
4. **是否可能是 Markov 矩阵？** 不可能；Markov 矩阵的谱半径为 $1$，而这里有特征值 $2$。
5. **$P=A/2$ 何时可能是投影？** 投影特征值必须为 $0$ 或 $1$，故 $c/2\in\{0,1\}$，即 $c=0$ 或 $c=2$。又已有正交特征基，所以此时确实为正交投影。

### 3.9.4 Review Problem 3：只看 $\Sigma$ 读出结构

若

$$
\Sigma=\begin{bmatrix}3&0\\0&2\end{bmatrix},
$$

且 $U,V$ 都是 $2\times2$ 正交矩阵，则 $A=U\Sigma V^T$ 是可逆 $2\times2$ 矩阵，秩为 $2$。

- $\operatorname{diag}(3,-5)$ 不能作为 SVD 的 $\Sigma$，因为奇异值必须非负。
- 若 $\Sigma=\operatorname{diag}(3,0)$，则 $\operatorname{rank}(A)=1$，$\dim N(A)=1$；$V$ 的第二列张成 $N(A)$，$U$ 的第二列张成 $N(A^T)$。

### 3.9.5 Review Problem 4：矩阵同时对称且正交

若 $A=A^T$ 且 $A^TA=I$，则 $A^2=I$。因此每个特征值满足

$$
\lambda^2=1,
$$

即 $\lambda=\pm1$。由此：

- $A$ 未必正定，例如 $\operatorname{diag}(1,-1)$；
- 特征值可以重复；
- $A$ 可正交对角化并一定可逆；
- $P=(A+I)/2$ 是投影，因为
  $$
  P^2=\frac14(A^2+2A+I)=\frac12(A+I)=P,
  $$
  且 $P^T=P$，所以是正交投影。

### 3.9.6 Recitation：投影、旋转与反射的谱

令 $a=(3,4)^T$。

1. 投影
   $$
   P=\frac{aa^T}{a^Ta}=\frac1{25}\begin{bmatrix}9&12\\12&16\end{bmatrix}
   $$
   有特征对：$a\leftrightarrow1$，$(-4,3)^T\leftrightarrow0$。
2. 旋转
   $$
   Q=\begin{bmatrix}0.6&-0.8\\0.8&0.6\end{bmatrix}
   $$
   有特征值 $0.6\pm0.8i$，对应复特征向量可取 $(1,\mp i)^T$。非平凡平面旋转没有实特征方向。
3. 反射 $R=2P-I$ 与 $P$ 有同一组特征向量；特征值由 $\lambda_R=2\lambda_P-1$ 得 $1,-1$。投影方向保持，垂直方向反向。

### 3.9.7 三道自检题

> [!question]- 自检 1（谱分类）
> 一个实对称矩阵所有特征值都在 $\{0,1\}$，它一定是什么矩阵？

> [!success]- 答案
> 正交投影矩阵。谱分解给 $P^2=Q\Lambda^2Q^T=Q\Lambda Q^T=P$。

> [!question]- 自检 2（SVD）
> 若 $A\in\mathbb R^{5\times3}$ 有两个非零奇异值，四个基本子空间维数是什么？

> [!success]- 答案
> $\dim C(A)=\dim C(A^T)=2$，$\dim N(A)=3-2=1$，$\dim N(A^T)=5-2=3$。

> [!question]- 自检 3（微分方程）
> “$A$ 的特征值实部都不正，所以 $e^{tA}$ 有界”是否总正确？

> [!success]- 答案
> 不正确。实部为零的特征值若有非平凡 Jordan 块，会产生 $t,t^2,\ldots$ 的多项式增长。若所有实部严格负则指数衰减；实部为零时还需半单/可对角化条件。

### Review 知识链

$$
\text{特殊矩阵结构}
\Longrightarrow\text{谱限制}
\Longrightarrow\text{分解}
\Longrightarrow\text{幂、指数、投影、稳定性与子空间}.
$$

---

## Unit 3 Exam 完整题解

> [!info] 考试材料
> 原题：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex3.pdf#page=1|Unit 3 Exam p.1]]；官方答案：[[MIT_OCW_18.06SC_PDF/02_Exercises/MIT18_06SCF11_ex3s.pdf#page=1|Official solutions p.1]]。原题 PDF 的部分文本层乱码，本节按页面视觉内容与官方答案逐项核对。

### Exam Problem 1（34 分）：SVD、谱分解与矩阵类别

> [!question] 题目
> (a) 方阵 $A$ 的全部 $n$ 个奇异值都等于 $1$。它必属于哪些基本矩阵类别：singular、symmetric、orthogonal、positive definite/semidefinite、diagonal？
>
> (b) $H$ 的标准正交列是 $B$ 的特征向量：
> $$
> H=\frac12\begin{bmatrix}
> 1&1&-1&-1\\
> 1&-1&-1&1\\
> 1&1&1&1\\
> 1&-1&1&-1
> \end{bmatrix},\qquad H^{-1}=H^T,
> $$
> $B$ 的特征值依次为 $0,1,2,3$。把 $B$ 写成三个具体矩阵的乘积；再把 $C=(B+I)^{-1}$ 写成三个矩阵的乘积。
>
> (c) 从 (a) 的类别列表中分别判断 $B,C$ 属于哪些类。

> [!success]- 完整解答
> **(a) 目标：从奇异值判断 $A^TA$。**
>
> SVD 为 $A=U\Sigma V^T$，而全部奇异值为 $1$，所以 $\Sigma=I$。于是
> $$
> A^TA=V\Sigma^TU^TU\Sigma V^T=VII V^T=I.
> $$
> 因而 $A$ 必为 **orthogonal matrix**，并且必可逆，所以不 singular。
>
> 其他类别都不必成立。反例取非平凡旋转
> $$
> \begin{bmatrix}0&-1\\1&0\end{bmatrix};
> $$
> 它奇异值全为 $1$，但不对称、不对角，也不是正定/半正定矩阵。因此唯一必选类别是 orthogonal。
>
> **(b) 目标：使用正交谱分解。**
>
> 令
> $$
> \Lambda=\operatorname{diag}(0,1,2,3).
> $$
> 因 $H$ 的列是按此顺序排列的标准正交特征向量，
> $$
> \boxed{B=H\Lambda H^T}.
> $$
> 这已经是三个题目指定的具体矩阵的乘积。
>
> $B+I$ 与 $B$ 有相同特征向量，特征值分别变成 $1,2,3,4$：
> $$
> B+I=H(\Lambda+I)H^T.
> $$
> 每个特征值都非零，所以可逆；利用 $(H^T)^{-1}=H$，
> $$
> \boxed{
> C=(B+I)^{-1}
> =H\operatorname{diag}\left(1,\frac12,\frac13,\frac14\right)H^T}.
> $$
>
> **(c) 分类。**
>
> $B$ 有正交谱分解且谱为 $0,1,2,3$，所以：
>
> - 有零特征值，故 singular；
> - $B=H\Lambda H^T$，故 symmetric；
> - 所有特征值非负，故 positive semidefinite；
> - 因有零特征值，不 positive definite；
> - 不必且实际并非 orthogonal 或 diagonal。
>
> $C$ 的特征值 $1,1/2,1/3,1/4$ 全正，且 $C=HC_\Lambda H^T$，所以 symmetric positive definite。它可逆，不 singular；特征值模长不全为 $1$，故不 orthogonal；标准基下也不 diagonal。
>
> **验点。** $BC$ 不必是 $I$，因为 $C$ 是 $(B+I)^{-1}$；正确检查是
> $$
> (B+I)C=H(\Lambda+I)(\Lambda+I)^{-1}H^T=I.
> $$

### Exam Problem 2（33 分）：对角化、矩阵幂与 $A^TA$

> [!question] 题目
> 设
> $$
> A=\begin{bmatrix}-1&2&4\\0&0&5\\0&0&1\end{bmatrix}.
> $$
> (a) 求三个特征值和一个特征向量矩阵 $S$。
>
> (b) 解释为什么 $A^{1001}=A$；$A^{1000}$ 是否等于 $I$？写出 $e^{At}$ 的三个对角元。
>
> (c) 已知
> $$
> A^TA=\begin{bmatrix}1&-2&-4\\-2&4&8\\-4&8&42\end{bmatrix}.
> $$
> 不直接求根，说明它有几个正、零、负特征值；它与 $A$ 是否有相同特征向量？

> [!success]- 完整解答
> **(a) 求谱与 $S$。**
>
> $A$ 为上三角矩阵，特征值就是对角元：
> $$
> \lambda_1=-1,\qquad \lambda_2=0,\qquad \lambda_3=1.
> $$
>
> 对 $\lambda=-1$，解 $(A+I)x=0$，可取
> $$
> x_1=\begin{bmatrix}1\\0\\0\end{bmatrix}.
> $$
>
> 对 $\lambda=0$，解 $Ax=0$。第三行给 $x_3=0$，第一行给 $-x_1+2x_2=0$，可取
> $$
> x_2=\begin{bmatrix}2\\1\\0\end{bmatrix}.
> $$
>
> 对 $\lambda=1$，解 $(A-I)x=0$。第二行给 $-x_2+5x_3=0$，故 $x_2=5x_3$；第一行给 $-2x_1+2x_2+4x_3=0$，故 $x_1=7x_3$。取 $x_3=1$：
> $$
> x_3=\begin{bmatrix}7\\5\\1\end{bmatrix}.
> $$
>
> 三个特征值互异，特征向量独立，所以
> $$
> S=\begin{bmatrix}1&2&7\\0&1&5\\0&0&1\end{bmatrix},\qquad
> A=S\operatorname{diag}(-1,0,1)S^{-1}.
> $$
>
> **(b) 矩阵幂与指数。**
>
> 对任何正整数 $k$，
> $$
> A^k=S\operatorname{diag}((-1)^k,0^k,1^k)S^{-1}.
> $$
> 当 $k=1001$ 为奇数时，对角阵仍为 $\operatorname{diag}(-1,0,1)$，所以
> $$
> \boxed{A^{1001}=A}.
> $$
> 当 $k=1000$ 时，
> $$
> A^{1000}=S\operatorname{diag}(1,0,1)S^{-1}\ne I,
> $$
> 因为中间特征方向仍被映为零；更快的检查是 $A$ singular，所以 $A^{1000}$ 也 singular，不可能等于可逆的 $I$。
>
> 矩阵指数为
> $$
> e^{At}=S\operatorname{diag}(e^{-t},1,e^t)S^{-1}.
> $$
> 由于 $S$、$S^{-1}$ 都是单位对角的上三角矩阵，$e^{At}$ 的三个对角元为
> $$
> \boxed{e^{-t},\ 1,\ e^t}.
> $$
> 也可由上三角矩阵幂级数逐项看出。
>
> **(c) $A^TA$ 的惯性与特征向量。**
>
> 对任意 $x$，$x^TA^TAx=\|Ax\|^2\ge0$，故 $A^TA$ 没有负特征值。$A$ 的前两列相关，第三列不在它们的张成中，所以 $\operatorname{rank}(A)=2$。又
> $$
> \operatorname{rank}(A^TA)=\operatorname{rank}(A)=2.
> $$
> 因而 $3\times3$ 的 $A^TA$ 有两个正特征值、一个零特征值、零个负特征值。
>
> 它不与 $A$ 共享整组特征向量。例如 $e_1$ 是 $A$ 对应 $-1$ 的特征向量，但
> $$
> A^TAe_1=\begin{bmatrix}1\\-2\\-4\end{bmatrix}
> $$
> 不是 $e_1$ 的倍数。一般而言，$A$ 的特征向量与 $A^TA$ 的右奇异向量是不同概念；只有 normal 等特殊情形才可能一致。

### Exam Problem 3（33 分）：正交特征基中的求逆与解方程

> [!question] 题目
> $A\in\mathbb R^{n\times n}$ 有标准正交特征向量 $q_1,\ldots,q_n$ 和正特征值 $\lambda_1,\ldots,\lambda_n$，即 $Aq_j=\lambda_jq_j$。
>
> (a) $A^{-1}$ 的特征值和特征向量是什么？证明。
>
> (b) 若 $b=c_1q_1+\cdots+c_nq_n$，利用正交性快速求 $c_1$。
>
> (c) 若 $A^{-1}b=d_1q_1+\cdots+d_nq_n$，快速求 $d_1$。

> [!success]- 完整解答
> **前置检查。** 所有 $\lambda_j>0$，所以 $A$ 无零特征值并可逆。标准正交特征向量矩阵 $Q=[q_1\ \cdots\ q_n]$ 满足 $Q^TQ=I$，且
> $$
> A=Q\Lambda Q^T,\qquad \Lambda=\operatorname{diag}(\lambda_j).
> $$
>
> **(a) 求逆的特征对。**
>
> 从
> $$
> Aq_j=\lambda_jq_j
> $$
> 左乘 $A^{-1}$：
> $$
> q_j=\lambda_jA^{-1}q_j.
> $$
> 再除以非零的 $\lambda_j$：
> $$
> A^{-1}q_j=\frac1{\lambda_j}q_j.
> $$
> 所以 $A^{-1}$ 与 $A$ 有相同特征向量，特征值变成倒数 $1/\lambda_j$。等价地，
> $$
> A^{-1}=Q\Lambda^{-1}Q^T.
> $$
>
> **(b) 用内积抽取系数。**
>
> 左乘 $q_1^T$：
> $$
> q_1^Tb=c_1q_1^Tq_1+\sum_{j=2}^nc_jq_1^Tq_j.
> $$
> 标准正交性给 $q_1^Tq_1=1$、$q_1^Tq_j=0$，所以
> $$
> \boxed{c_1=q_1^Tb}.
> $$
> 若只知正交而未归一化，则应为 $c_1=(q_1^Tb)/(q_1^Tq_1)$。
>
> **(c) 求解系数。**
>
> 对展开式逐项作用 $A^{-1}$：
> $$
> A^{-1}b
> =\sum_{j=1}^nc_jA^{-1}q_j
> =\sum_{j=1}^n\frac{c_j}{\lambda_j}q_j.
> $$
> 因此
> $$
> \boxed{d_1=\frac{c_1}{\lambda_1}
> =\frac{q_1^Tb}{\lambda_1}}.
> $$
>
> **验点与意义。** 把 $x=A^{-1}b$ 代回：
> $$
> Ax=\sum_j\lambda_j\frac{c_j}{\lambda_j}q_j
> =\sum_jc_jq_j=b.
> $$
> 这说明在特征向量坐标中，解 $Ax=b$ 只是把第 $j$ 个坐标除以 $\lambda_j$。很小的 $\lambda_j$ 会显著放大该方向上的误差，这与 SVD 中小奇异值导致病态完全平行。

---

## 本单元最终检查表

### 理论与证明

- [ ] 能用 $x^*x>0$ 正确证明实对称/Hermitian 矩阵的特征值为实，不再误用复数情形的 $x^Tx$。
- [ ] 能证明不同特征值的特征向量正交，并写出 $A=Q\Lambda Q^T$ 或 $Q\Lambda Q^*$。
- [ ] 能说明正定的各判据及其全部前提，尤其是对称性与不换行的 $LDL^T$ 主元。
- [ ] 能把配方、$LDL^T$、Hessian 和唯一极小值连成一条推导。
- [ ] 能区分相似与合同，并说明 Jordan 形的域/分裂假设。
- [ ] 能由 $e^{tJ}$ 解释 Jordan 块的多项式因子，以及纯虚谱不自动保证有界。
- [ ] 能从谱定理逐步构造 SVD，并说明四个基本子空间分别由哪些奇异向量张成。
- [ ] 能从基向量的像构造线性变换矩阵，并写清输入/输出基。
- [ ] 能从 $x=Wc$ 推出向量换基与算子换基公式。
- [ ] 能证明左右逆的秩条件，以及伪逆的两个投影与最小范数性质。

### 计算技能

- [ ] 会用特征值、顺序主子式、主元和二次型至少两种方法判断正定。
- [ ] 会对角化 Hermitian 矩阵并正确使用共轭转置。
- [ ] 会写 $F_4$，解释 Fourier 列正交与 FFT 偶奇分解。
- [ ] 会识别相似不变量，并用几何重数/Jordan 块排除错误相似关系。
- [ ] 会按“$A^TA\to V,\Sigma\to u_i=Av_i/\sigma_i$”稳定地计算 SVD。
- [ ] 会由 $\Sigma$ 直接读秩、零空间维数、谱范数和四子空间。
- [ ] 会在两组基之间换向量坐标和变换矩阵。
- [ ] 会计算满列秩左逆、满行秩右逆以及 SVD 伪逆。
- [ ] 已独立重做三道 Unit 3 Exam，并用定义或代回检查最终答案。

## 全单元知识链

$$
\boxed{
\begin{array}{c}
\text{对称/Hermitian}\to\text{正交谱分解}\to\text{正定与二次型}\\[2mm]
\text{一般方阵}\to\text{相似与 Jordan}\to\text{幂、指数与稳定性}\\[2mm]
\text{任意矩阵}\to\text{SVD}\to\text{四子空间、低秩近似与伪逆}\\[2mm]
\text{抽象线性变换}\to\text{选基得到矩阵}\to\text{换基与压缩}
\end{array}}
$$
