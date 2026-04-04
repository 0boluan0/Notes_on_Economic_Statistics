---
aliases:
  - MIT 18.06SC Unit I
  - Ax=b 与四个基本子空间
tags:
  - 线性代数
  - mit-ocw
  - course-note
---

# Ax = b and the Four Subspaces

## 单元速览

- 这一单元真正回答的是：`Ax=b` 为什么会有解 / 无解 / 不唯一，以及这些现象如何被子空间语言统一解释。
- 你最后必须能把题目自动分流到两条线：
  - 算法线：elimination、pivot、row reduction、LU。
  - 结构线：[[Column Space]]、[[Null Space]]、[[Row Space]]、[[Left Nullspace]]、rank、basis、dimension。
- 如果复习时间很少，至少要保住这 5 个抓手：
  - `b` 是否在 [[Column Space]] 里。
  - 解是否等于 `particular solution + [[Null Space]]`。
  - pivot 个数就是 rank。
  - `rank + nullity = n`。
  - 四个基本子空间分别活在 $\mathbb{R}^m$ 还是 $\mathbb{R}^n$。

## 这页怎么用

- 第一次复习：先看每个 session 前面的 `快速回忆`，能口述后再读正文。
- 第二次复习：直接跳到每讲的 `你要掌握` 和本页结尾的 `本单元复习清单`。
- 做题卡住时，不要立刻回头重读整页，先问自己：
  - 这是解结构题、basis 题，还是四个基本子空间题？
  - 当前卡住的是算法步骤，还是空间解释？

## Session 回忆索引

- 1.1：row picture / column picture / matrix picture。
- 1.2：整门课为什么会从消元一路走到 SVD。
- 1.3-1.5：消元、主元、可逆性、LU。
- 1.6-1.10：vector spaces、column/null space、special solutions、basis、dimension。
- 1.11：四个基本子空间的总收束。
- 1.12-1.13：rank-1、matrix spaces、graphs / incidence matrix。
- 1.14：Exam 1 前的统一闭环。

## 本单元主线

Unit I 讲的是整门线性代数最基础、也最耐用的问题：给定矩阵 $A$ 和向量 $b$，方程 $Ax=b$ 什么时候有解、什么时候唯一、什么时候会出现自由度。MIT 这门课的做法不是把它停留在“解方程技巧”，而是把问题提升成“矩阵怎样把一个空间映到另一个空间”。

因此这一单元有两条主线同时展开。第一条是算法线：高斯消元、回代、主元、换行、LU 分解。第二条是结构线：[[Column Space]] 决定哪些 $b$ 可达，[[Null Space]] 决定解为什么会不唯一，[[Row Space]] 与 [[Left Nullspace]] 则把矩阵的行结构和约束结构完整补齐。学完这一单元以后，任何“线性方程组”题都应该自动翻译成“列空间、零空间、秩和基”的问题。

## 使用资料

- 总入口：[[00_MIT OCW 18.06SC course map|课程总览]]
- 题目索引：[[MIT_OCW_18.06SC_PDF/index|MIT OCW 18.06SC 题目与资料索引]]
- 注意：本地 PDF 的 `An Overview of Key Ideas` 编号是 `Ses1.13sum.pdf`，但这里按 syllabus 放在 Session 1.2。

## Session 1.1 The geometry of linear equations

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf|summary]]

关联卡片：[[Linear system solution structure]]、[[Column Space]]、[[Vector Space]]

>[!note] 快速回忆
> - 这讲要回答：同一个线性系统为什么能同时看成几何交点问题和列向量组合问题。
> - 你要立刻想起：row picture 看交点，column picture 看 `b` 是否由列向量生成，matrix picture 看线性变换。
> - 典型题型：把一个二元/三元系统翻译成三种 picture，并解释唯一解、无解、无穷多解。
> - 它接到下一讲：从几何直觉升级到整门课的结构地图。

### 核心问题

线性方程组一开始看起来像若干个公式，但它本质上是一个几何问题。每个方程都定义了一个超平面，解就是这些超平面的交点；把矩阵写出来以后，同一个问题又可以被看成“某个向量 $b$ 是否能由矩阵列向量线性组合出来”。

### 三种图像

- `row picture`：每一行给一个方程。在二维里是若干条直线，在三维里是若干个平面。解集是这些几何对象的交。
- `column picture`：若
  $$
  A=\begin{bmatrix}a_1&a_2&\cdots&a_n\end{bmatrix},
  $$
  那么
  $$
  Ax=x_1a_1+\cdots+x_na_n.
  $$
  求解 $Ax=b$ 等价于问 $b$ 能否由这些列向量生成，以及它在这组列向量中的系数是什么。
- `matrix picture`：把整个系统压成一个对象 $A$，从此以后不再是“若干方程”，而是“一个线性映射”。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-row-column-picture.svg|760]]

上图把这两种最早出现的视角放在一起。左边强调“解是交点”，右边强调“解是系数”。这两个画法看起来不同，但问的是同一件事。课堂上很多同学会在这里第一次感到抽象，因为他们还把矩阵看成一个算术表格；真正跨过去的标志，是你开始把矩阵看成“组织一组向量并作用到整个空间上的规则”。

>[!example] 典型例子
>
> 对系统
> $$
> \begin{cases}
> 2x-y=0\\
> -x+2y=3
> \end{cases}
> $$
> row picture 是两条直线在平面中的交点；column picture 是
> $$
> x\begin{bmatrix}2\\-1\end{bmatrix}
> +y\begin{bmatrix}-1\\2\end{bmatrix}
> =\begin{bmatrix}0\\3\end{bmatrix}.
> $$
> 当你解出 $(x,y)=(1,2)$ 时，同时完成了两件事：找到了直线交点，也找到了右端向量在列向量基底下的坐标。
>
> 再往前一步看，线性系统总共有三种几何命运：
>
> - 若几何对象交于一点，对应唯一解；
> - 若它们互相平行或相容条件不成立，对应无解；
> - 若它们部分重合，对应无穷多解。
>
> 这三种命运在代数上分别对应“列独立且 $b$ 可达”“$b$ 不在列空间里”“列有冗余且 $b$ 可达”。也就是说，后面学到的列空间、零空间与秩，并不是突然引入的新对象，而是把这三种几何命运系统化后的语言。
>
### 结构结论

如果列向量张成的空间太小，那么有些 $b$ 根本不在里面，系统就无解；如果列向量之间有冗余，那么同一个 $b$ 可能对应多组系数，系统就不唯一。换句话说，“有解”由列空间控制，“唯一”由列向量是否独立控制。

### 你要掌握

- 能把一个二元或三元线性系统同时画成 row picture 和 column picture。
- 能用“列向量线性组合”解释矩阵乘法。
- 能从几何上区分无解、唯一解、无穷多解。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.2 An overview of key ideas

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.13sum.pdf|summary]]

关联卡片：[[Linear Algebra-hub|线性代数 Hub]]、[[Vector Space]]、[[Matrix Rank]]

>[!note] 快速回忆
> - 这讲要回答：为什么 elimination、subspace、orthogonality、eigen、SVD 会串成一门课。
> - 你要立刻想起：整门课的推进顺序是 `Ax=b -> 子空间 -> 正交/最佳逼近 -> determinant/eigen -> 更一般的分解`。
> - 典型题型：用一句话说明某个章节在整门线代里的角色。
> - 它接到下一讲：课程地图落回具体算法，从消元开始。

### 这讲在课程里的位置

这一讲不是新算法，而是整门课的路线图。Strang 在这里强调：线性代数不是一堆互不相干的章节，而是一张图。你先通过消元看到解的结构，再通过子空间理解可解性，再通过正交和投影处理无解系统，最后用行列式、特征值和奇异值分解去描述矩阵的深层结构。

### 从差分矩阵看到“可逆”与“不可逆”

summary 里一个重要对比是“普通差分矩阵”和“循环差分矩阵”。前者的列独立，所以对每个 $b$ 都有唯一解；后者因为首尾也相连，会出现一条线性依赖，于是
$$
Cx=0
$$
有非零解，而
$$
Cx=b
$$
只有在 $b_1+b_2+b_3=0$ 之类的兼容条件满足时才有解。这个例子说明：矩阵一旦丢失秩，就同时带来两个后果，某些方向打不出来，某些输入又会被压成 0。

### 课程地图

- 消元回答“怎么解”。
- 子空间回答“为什么有解/无解/不唯一”。
- [[Orthogonality]] 与 [[Orthogonal Projection]] 回答“无解时的最佳近似是什么”。
- [[Determinant]] 回答“矩阵是否把空间压扁，以及体积缩放了多少”。
- [[Eigenvalues]] 与 [[Eigenvectors]] 回答“反复施加这个矩阵时，哪些方向会保留下来”。
- [[Singular Value Decomposition]] 最终把一般矩阵分解成若干最自然的输入方向与输出方向。

### 核心结构观

整门课最值得养成的习惯是：不要只盯着单个数值步骤，而要不断问“这个矩阵的秩是多少，它的列空间和零空间长什么样，它在空间中丢掉了哪些方向”。秩控制有效维数，零度控制丢失自由度，二者通过
$$
\operatorname{rank}(A)+\operatorname{nullity}(A)=n
$$
连在一起。

### 你要掌握

- 能用一句话说清楚每个后续章节在整门课中的作用。
- 能解释“可逆矩阵”和“列独立、满秩、零空间平凡、每个 $b$ 唯一可解”其实是同一件事的不同表述。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.3 Elimination with matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.2sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.2prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.2sol.pdf|solution]]

关联卡片：[[Matrix Rank]]、[[LU Decomposition]]

>[!note] 快速回忆
> - 这讲要回答：消元究竟在做什么，以及它为什么不改变解集。
> - 你要立刻想起：非零主元的个数就是 rank；遇到 0 主元要么换行，要么接受秩下降。
> - 典型题型：手算消元、读 pivot / rank / free variables。
> - 它接到下一讲：从消元步骤过渡到可逆性和逆矩阵。

### 消元在做什么

高斯消元的目标不是“把矩阵弄好看”，而是系统地消掉下三角元素，把原方程组变成一个容易回代的上三角系统。每一步都只是把一行减去另一行的倍数，因此不会改变解集。

对矩阵
$$
A=\begin{bmatrix}
1&2&1\\
3&8&1\\
0&4&1
\end{bmatrix},
$$
先用第 1 行消去第 2 行首项，再用第 2 行消去第 3 行第二项，可以得到
$$
U=\begin{bmatrix}
1&2&1\\
0&2&-2\\
0&0&5
\end{bmatrix}.
$$
这时原系统 $Ax=b$ 已经变成了更容易处理的 $Ux=c$。

### 主元与秩

每次成功选到一个非零主元，意味着发现了一个新的独立方向。主元数就是秩，主元列是原矩阵中真正不可由前面列组合出来的列。没有主元的列会对应自由变量，说明矩阵存在冗余。

### 消元矩阵

把“第二行减去三倍第一行”这种操作写成矩阵，就是左乘一个消元矩阵。例如
$$
E_{21}=
\begin{bmatrix}
1&0&0\\
-3&1&0\\
0&0&1
\end{bmatrix}
$$
满足 $E_{21}A$ 正是做完第一步消元后的结果。因为 $E_{21}$ 可逆，消元过程本质上是在做一串可逆变换，所以解集不变。

### 为什么有时必须换行

如果某个主元位置为 0，而该列下面存在非零元素，就必须交换两行，不然消元会被卡住。如果整列都为 0，则这一列不可能提供主元，秩会下降，矩阵不再可逆。这个现象告诉你：数值上看到的“0 主元”，结构上对应的就是“没有足够独立的信息”。

### 你要掌握

- 能手工对一个 $3\times 3$ 或 $4\times 4$ 系统完成前向消元和回代。
- 能根据主元位置读出秩和自由变量个数。
- 能解释为什么消元不改变解集。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.4 Multiplication and inverse matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.3prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.3sol.pdf|solution]]

关联卡片：[[Matrix Inverse]]、[[Singular Matrix]]、[[Invertible Matrix Equivalence Chain]]

>[!note] 快速回忆
> - 这讲要回答：矩阵乘法怎样理解，以及 inverse 到底意味着什么。
> - 你要立刻想起：矩阵乘法最重要的视角是“复合变换”；可逆等价于“每个 $b$ 唯一可解、零空间平凡、主元充满”。
> - 典型题型：判断矩阵是否可逆、解释 singular 到底坏在哪。
> - 它接到下一讲：把消元过程系统整理成 LU / PA=LU。

### 矩阵乘法的四种理解

矩阵乘法至少要能从四个角度理解：

- 行乘列：$(AB)_{ij}$ 是 A 的第 $i$ 行与 B 的第 $j$ 列点积。
- 列组合：$Ax$ 是 A 的列向量按 $x$ 的系数组合。
- 行组合：$y^TA$ 是 A 的行向量按 $y$ 的系数组合。
- 复合变换：$AB$ 表示先做 B，再做 A。

真正最重要的是最后一种。矩阵不是表格，而是线性变换；矩阵乘法对应变换复合。

### 逆矩阵的含义

[[Matrix Inverse]] 的本质不是一个公式，而是“撤销 A 的作用”。若
$$
A^{-1}A=I,\qquad AA^{-1}=I,
$$
则 A 把每个输入一一对应地送到一个输出，再由 $A^{-1}$ 精确取回。于是
$$
Ax=b \iff x=A^{-1}b.
$$

### 可逆性的等价条件

对 $n\times n$ 方阵，下列说法等价：

- $A$ 可逆。
- $Ax=b$ 对每个 $b$ 都有唯一解。
- $Ax=0$ 只有零解。
- A 的列向量线性无关。
- A 的主元有 $n$ 个。
- A 可以通过消元变成单位矩阵。

这些等价条件值得熟到不用思考，因为后续任何题目都在它们之间切换。

### 奇异矩阵在结构上意味着什么

如果 A 是 [[Singular Matrix]]，那不是“某个公式算坏了”，而是 A 把空间压扁了。存在非零向量被送到 0，也存在一些目标向量永远无法被打出来。所以“零空间非平凡”和“列空间不满”其实是同一个几何现象从输入端和输出端的两种描述。

### 你要掌握

- 能从“复合变换”解释 $(AB)^{-1}=B^{-1}A^{-1}$ 为何顺序反过来。
- 能用上面的等价条件快速判断一个方阵是否可逆。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.5 Factorization into A = LU

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.4prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.4sol.pdf|solution]]

关联卡片：[[LU Decomposition]]、[[Permutation Matrix]]

>[!note] 快速回忆
> - 这讲要回答：为什么消元算法能写成矩阵分解。
> - 你要立刻想起：不换行时是 `A=LU`；需要换行时是 `PA=LU`。
> - 典型题型：手写 LU / PA=LU，并解释 L、U、P 各自记录了什么。
> - 它接到下一讲：从算法对象转向“什么叫向量空间与子空间”。

### 从算法到分解

高斯消元本来是“过程”，LU 分解把这个过程固化成“结构”。如果消元中没有换行，那么对矩阵 A 进行的每一步消元都可以写成左乘一个消元矩阵，最终得到
$$
E_k\cdots E_2E_1A=U.
$$
把这些消元矩阵反过来收集，就得到
$$
A=LU,
$$
其中 $L$ 是单位下三角矩阵，记录消元乘子，$U$ 是上三角矩阵，记录被消元后的结果。

### 一个直观例子

对
$$
A=\begin{bmatrix}
1&2&1\\
3&8&1\\
0&4&1
\end{bmatrix},
$$
消元乘子是 $l_{21}=3,\ l_{32}=2$，因此
$$
L=\begin{bmatrix}
1&0&0\\
3&1&0\\
0&2&1
\end{bmatrix},
\qquad
U=\begin{bmatrix}
1&2&1\\
0&2&-2\\
0&0&5
\end{bmatrix}.
$$
L 记住“消掉时用了谁的多少倍”，U 记住“最后剩下什么”。

如果右端是
$$
b=\begin{bmatrix}1\\4\\2\end{bmatrix},
$$
那么利用 LU 解方程的过程会被清楚地拆成两段：
$$
Ly=b,\qquad Ux=y.
$$
第一段只是沿着消元留下的依赖关系往下传信息，第二段才是标准的上三角回代。这个拆分的重要意义在于，它把“算法中的前向消元”变成了可重复使用的结构因子，所以你对同一个 A 更换多个不同的右端时，不必每次都从头消元。

### 为什么有时要写成 PA=LU

一旦消元中出现换行，单纯的 $A=LU$ 就不够了，因为换行不是消元矩阵而是 [[Permutation Matrix]]。这时正确写法是
$$
PA=LU.
$$
这里 P 把原矩阵的行重新排序，让每一步都能选到合适主元。

### LU 的实际价值

若要反复求解同一个 A 对应的多个不同右端 $b^{(1)},b^{(2)},\dots$，直接重复消元很浪费。先做一次 LU 之后，每次只需解
$$
Ly=b,\qquad Ux=y.
$$
这就是数值线性代数里“分解一次，重复使用”的基本思想。

### 你要掌握

- 能手工求一个小矩阵的 LU 分解。
- 能解释 L 和 U 分别保存了什么信息。
- 知道何时要改写成 $PA=LU$。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.6 Transposes, permutations, vector spaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.5prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.5sol.pdf|solution]]

关联卡片：[[Permutation Matrix]]、[[Vector Space]]、[[Subspace]]

>[!note] 快速回忆
> - 这讲要回答：为什么线代里到处在说“空间”，以及 transpose / permutation 在结构上意味着什么。
> - 你要立刻想起：transpose 交换行列角色；vector space 关注 closure，而 permutation 只是重排坐标/方程。
> - 典型题型：判断一个集合是不是子空间，解释转置后对象活在哪个空间。
> - 它接到下一讲：正式进入 column space 与 nullspace。

### 转置把行和列互换

转置 $A^T$ 的作用是把矩阵关于主对角线翻过去，于是行和列交换位置。这个看似简单的操作有两个重要后果：

- row space 会变成 $A^T$ 的 column space；
- 公式顺序会反转：
  $$
  (AB)^T=B^TA^T.
  $$

以后所有关于对称矩阵、正交矩阵、正规方程的结构，都要靠这个转置规则。

### 置换矩阵

[[Permutation Matrix]] 是把单位矩阵行或列重排得到的矩阵。左乘时重新排列行，右乘时重新排列列。它表示的不是一般线性变换，而是“基向量重新编号”，所以每行每列恰有一个 1，其余是 0。

### 从“集合”升级为“空间”

[[Vector Space]] 的定义非常朴素：对加法和数乘封闭，并包含零向量。[[Subspace]] 是向量空间中的线性子集，因此也必须经过原点。这个“经过原点”极其重要，因为它能一眼排除大多数不是子空间的集合。

在 $R^3$ 里，典型子空间只有四类：$\{0\}$、过原点的直线、过原点的平面、整个 $R^3$。任何不经过原点的平面都不是子空间，因为不包含零向量，也不对数乘封闭。

### 你要掌握

- 会用封闭性判断一个集合是不是 subspace。
- 熟练使用 $(AB)^T=B^TA^T$。
- 知道 row space 与 column space 在转置下如何互换。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.7 Column space and nullspace

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.6prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.6sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Subspace]]

>[!note] 快速回忆
> - 这讲要回答：column space 和 nullspace 为什么是最先必须掌握的两个子空间。
> - 你要立刻想起：[[Column Space]] 控制哪些 $b$ 可达，[[Null Space]] 控制为什么解不唯一。
> - 典型题型：判断系统是否相容，解释冗余列带来的后果。
> - 它接到下一讲：先把 `Ax=0` 的结构吃透。

### 两个最先出现的子空间

[[Column Space]] 是 A 的所有列向量线性组合形成的空间：
$$
C(A)=\{Ax:x\in\mathbb{R}^n\}\subseteq \mathbb{R}^m.
$$
它回答的是“哪些右端向量可以被 A 打出来”。

[[Null Space]] 是所有满足 $Ax=0$ 的向量组成的空间：
$$
N(A)=\{x\in\mathbb{R}^n:Ax=0\}\subseteq \mathbb{R}^n.
$$
它回答的是“哪些输入在经过 A 之后会消失”。

### 为什么这两个空间足够重要

如果 $b\notin C(A)$，那么 $Ax=b$ 不可能有解；如果 $N(A)$ 里除了 0 还有别的向量，那么一旦某个 $x_p$ 是解，所有
$$
x=x_p+x_n,\qquad x_n\in N(A)
$$
也都是解。于是可解性由列空间决定，不唯一性由零空间决定。

### 一个简单但关键的例子

若
$$
A=\begin{bmatrix}
1&2&3\\
2&4&6
\end{bmatrix},
$$
第二行是第一行的两倍，所以列空间其实只是一条直线；同时变量之间存在冗余，零空间至少是一维。你从“输出空间太小”和“输入空间有多余方向”两边都能看到同一个秩亏损。

### 你要掌握

- 能从题目语句判断它问的是列空间还是零空间。
- 知道列空间活在 $\mathbb{R}^m$，零空间活在 $\mathbb{R}^n$，两者不是同一个空间里的对象。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.8 Solving Ax = 0: pivot variables, special solutions

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.7prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.7sol.pdf|solution]]

关联卡片：[[Null Space]]、[[Matrix Rank]]

>[!note] 快速回忆
> - 这讲要回答：为什么齐次系统是理解一切解结构的起点。
> - 你要立刻想起：free variable 的个数就是零空间自由度；special solutions 构成 [[Null Space]] 的基。
> - 典型题型：从 rref 直接写出 `Ax=0` 的通解与基。
> - 它接到下一讲：再把一般系统写成“特解 + 零空间”。

### 齐次系统为什么最重要

求解 $Ax=0$ 是理解零空间最直接的入口。因为右端为 0，所以永远至少有零解；真正的问题是是否还有非零解。只要出现自由变量，就一定存在非零解。

### 从 rref 读结构

当消元把矩阵化到行最简形
$$
R=\begin{bmatrix}
I&F\\
0&0
\end{bmatrix},
$$
前面的主元列对应主变量，后面的非主元列对应自由变量。设自由变量为参数，就能显式写出所有解。

### Special solutions

每个自由变量都对应一个 `special solution`：把这个自由变量设为 1，其余自由变量设为 0，然后解出主变量。这些 special solutions 构成 [[Null Space]] 的一组基。若 A 有 $n$ 列、秩为 $r$，则自由变量个数为 $n-r$，因此
$$
\dim N(A)=n-r.
$$

>[!example] 例子
>
> 若
> $$
> R=\begin{bmatrix}
> 1&0&2&-1\\
> 0&1&-3&4\\
> 0&0&0&0
> \end{bmatrix},
> $$
> 则 $x_3,x_4$ 自由。令 $(x_3,x_4)=(1,0)$ 和 $(0,1)$，分别得到两组 special solutions；它们张成整个零空间。这里最容易犯的错，是只写出某一组非零解，却忘了整个解集是一个子空间。
>
### 你要掌握

- 能从 rref 直接写出零空间的一组基。
- 知道自由变量个数就是零空间维数。
- 明白“special solutions 不是某些偶然例子，而是零空间基向量”。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.9 Solving Ax = b: row reduced form R

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.8sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.8prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.8sol.pdf|solution]]

关联卡片：[[Linear system solution structure]]、[[Null Space]]、[[Column Space]]

>[!note] 快速回忆
> - 这讲要回答：一般系统 `Ax=b` 的全部解怎样描述。
> - 你要立刻想起：所有解都写成 `particular solution + [[Null Space]]`；是否有解先看相容条件。
> - 典型题型：判断相容、找一个特解、再参数化全部解。
> - 它接到下一讲：从“会解”走到“哪些向量算 independent / basis”。

### 把一般系统拆成“是否相容”与“如何参数化”

当右端变成一般的 $b$ 时，问题分成两步：

1. 系统是否相容，也就是 $b$ 是否落在 [[Column Space]] 中。
2. 如果相容，所有解如何描述。

行最简形最适合回答第二个问题。因为它把主变量与自由变量的关系完全显出来。

### particular solution + nullspace

若 $x_p$ 是某个特解，而 $x_n\in N(A)$，则
$$
A(x_p+x_n)=Ax_p+Ax_n=b+0=b.
$$
反过来，任意两个解之差都在零空间里。因此一般解有统一形式
$$
x=x_p+x_n,\qquad x_n\in N(A).
$$
这不是技巧，而是整个线性系统理论的骨架。

### 相容条件从哪里来

如果 rref 的某一行变成
$$
0=1,
$$
系统就矛盾；这等价于说 $b$ 不在列空间里。把这个条件翻成“某个线性关系必须由 b 也满足”，就会自然引出左零空间和 Fredholm 风格的兼容条件。

>[!example] 例子
>
> 假设
> $$
> \begin{bmatrix}
> 1&2\\
> 2&4
> \end{bmatrix}
> \begin{bmatrix}x_1\\x_2\end{bmatrix}
> =
> \begin{bmatrix}b_1\\b_2\end{bmatrix}.
> $$
> 由于第二行是第一行的两倍，系统有解的必要充分条件是 $b_2=2b_1$。一旦这个条件成立，解不是唯一的，而是一条仿射直线，因为零空间是一维。
>
### 你要掌握

- 能把一般解写成“特解 + 零空间”。
- 能从消元结果读出相容条件。
- 明白无解的根源是 $b$ 不在列空间里。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.10 Independence, basis, and dimension

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.9prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.9sol.pdf|solution]]

关联卡片：[[Vector Space]]、[[Subspace]]、[[Matrix Rank]]

>[!note] 快速回忆
> - 这讲要回答：什么叫 linear independence、basis、dimension。
> - 你要立刻想起：basis 是“最少还能张成 / 最多仍保持独立”的那组向量；pivot column 往往直接指向 basis。
> - 典型题型：从一组向量里挑 basis、求 dimension、说明为什么 independent。
> - 它接到下一讲：把这些语言全部收束到四个基本子空间。

### 线性无关到底在说什么

向量组 $v_1,\dots,v_k$ 线性无关，意思是
$$
c_1v_1+\cdots+c_kv_k=0
$$
只能由全零系数给出。这表示这些向量中没有一个是其余向量的线性组合，因此每个向量都提供了新的方向信息。

### 基与维数

一组向量如果既线性无关又能张成整个空间，就构成一个 basis。基的意义不是“某组好用的坐标轴”，而是“把空间中每个向量都唯一表示出来”的最小完整系统。所有基的向量数相同，这个公共数叫做维数。

在矩阵问题里，基和维数会不断出现：

- 列空间的基来自主元列。
- 零空间的基来自 special solutions。
- 行空间的基来自 rref 的非零行。

### 主元与独立性

矩阵的主元列恰好对应线性无关的列；非主元列则可由前面的主元列表示。于是消元不仅是求解工具，也是判断 independence 和寻找 basis 的工具。

### 典型结论

- $n$ 个独立向量在 $\mathbb{R}^n$ 中自动构成一组基。
- 超过 $n$ 个向量在 $\mathbb{R}^n$ 中必定线性相关。
- 若一个子空间维数是 $r$，任何基都恰有 $r$ 个向量。

### 你要掌握

- 能判断一组向量是否独立。
- 能从矩阵的主元结构提取某个子空间的一组基。
- 不把“生成”与“独立”混为一谈。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.11 The four fundamental subspaces

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.10prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.10sol.pdf|solution]]

关联卡片：[[Column Space]]、[[Null Space]]、[[Row Space]]、[[Left Nullspace]]、[[Reading the Four Fundamental Subspaces from RREF]]

>[!note] 快速回忆
> - 这讲要回答：一个矩阵为什么天然带出四个基本子空间。
> - 你要立刻想起：`C(A), N(A), C(A^T), N(A^T)` 分别活在不同 ambient space；正交关系和维数关系一起组成闭环。
> - 典型题型：列出四个子空间的基、维数、所在空间与正交关系。
> - 它接到下一讲：从向量空间走到“矩阵本身构成的空间”和 rank-1 视角。

### 四个空间的完整表

对任意 $m\times n$ 矩阵 A，都有四个基本子空间：

- [[Column Space]] $C(A)\subseteq \mathbb{R}^m$
- [[Null Space]] $N(A)\subseteq \mathbb{R}^n$
- [[Row Space]] $C(A^T)\subseteq \mathbb{R}^n$
- [[Left Nullspace]] $N(A^T)\subseteq \mathbb{R}^m$

它们不是零散定义，而是一个闭合系统。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-four-subspaces.svg|820]]

这张图最值得反复看。很多人第一次学四个基本子空间时，会把它们都当成“和 A 有关的四个集合”，但真正应该看到的是：输入空间 $\mathbb{R}^n$ 被拆成 row space 和 nullspace，输出空间 $\mathbb{R}^m$ 被拆成 column space 和 left nullspace；A 真正可逆的部分，只发生在 row space 到 column space 之间。

### 维数关系

若 $\operatorname{rank}(A)=r$，则
$$
\dim C(A)=r,\qquad \dim C(A^T)=r,
$$
$$
\dim N(A)=n-r,\qquad \dim N(A^T)=m-r.
$$
这四个数字把输入空间和输出空间都正好拆满了。于是 rank-nullity 不只是一个公式，而是四个子空间之间的维数账本。

### 正交关系

row space 与 nullspace 在 $\mathbb{R}^n$ 中互为正交补，column space 与 left nullspace 在 $\mathbb{R}^m$ 中互为正交补。理由很直接：若 $x\in N(A)$，则每一行和 $x$ 的点积都为 0；若 $y\in N(A^T)$，则每一列和 $y$ 的点积都为 0。

把这件事说得更“做题化”一点：只要你在某题里看见了“兼容条件”“误差正交”“残差与列空间垂直”，本质上都在调用这对正交补结构。Unit II 的最小二乘几乎就是把“$b$ 不在 column space 中怎么办”这件事系统化。

### 如何找各自的基

- 列空间的基：原矩阵中的主元列。
- 零空间的基：rref 对应的 special solutions。
- 行空间的基：rref 的非零行。
- 左零空间的基：求解 $A^Ty=0$，或者从增广消元矩阵里读出。

### 为什么这一讲是第一单元的核心

前面所有内容在这里汇总。可解性、不唯一性、主元、秩、自由变量、正交关系，都可以用这四个空间重新表述。后面的投影与最小二乘，本质上也只是在列空间与左零空间这对正交补之间工作。

### 你要掌握

- 能画出四个基本子空间分别活在哪个空间里。
- 能背出四个维数公式，并理解它们为什么成立。
- 能用四个基本子空间重述“有解、唯一、最小二乘”等问题。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.12 Matrix spaces; rank 1; small world graphs

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.11sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.11prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.11sol.pdf|solution]]

关联卡片：[[Matrix Rank]]、[[Symmetric Matrix]]

>[!note] 快速回忆
> - 这讲要回答：为什么 rank-1 值得被单独拎出来，以及矩阵空间本身怎样组织。
> - 你要立刻想起：rank-1 矩阵是外积块，是更复杂矩阵的基本构件。
> - 典型题型：判断 rank-1、说明矩阵空间的维数与基。
> - 它接到下一讲：把线代语言放进 graph / network 模型。

### 矩阵本身也构成向量空间

所有 $m\times n$ 矩阵在逐项加法和数乘下形成一个维数为 $mn$ 的向量空间。于是“上三角矩阵集合”“对称矩阵集合”“对角矩阵集合”都可以作为这个大空间中的子空间来研究。这一视角很重要，因为它告诉你：线性代数不只处理列向量，也处理函数、矩阵、信号等各种线性对象。

### rank-1 矩阵是最基本的构件

若
$$
A=uv^T,
$$
则 A 的每一列都是 $u$ 的倍数，每一行都是 $v^T$ 的倍数，所以 A 的秩至多为 1。rank-1 矩阵可以被看成“一个输出方向乘一个输入系数模式”。一般矩阵可以拆成若干 rank-1 矩阵之和，这为后面的 [[Singular Value Decomposition]] 埋下伏笔。

### 为什么 rank-1 值得重视

rank-1 是“最简单但非平凡”的矩阵。它只有一个真正的方向被保留下来，其他所有信息都被压扁。因此研究 rank-1，相当于研究矩阵如何从复杂映射退化到单方向映射。

### small world graphs 的视角

这一讲把矩阵与图联系起来。图的邻接关系可以写成矩阵，局部连接和长距离连接会影响矩阵的稀疏结构和路径长度。虽然这里没有深入图论算法，但要记住：矩阵并不只来自方程组，也来自网络、关系和数据结构。

### 你要掌握

- 能把“矩阵空间”当成普通向量空间来做维数和子空间判断。
- 能解释 rank-1 矩阵为什么等于一个列向量与一个行向量的外积。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.13 Graphs, networks, incidence matrices

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.12sum.pdf|summary]] | [[MIT_OCW_18.06SC_PDF/03_Homework_Problems/MIT18_06SCF11_Ses1.12prob.pdf|problem]] | [[MIT_OCW_18.06SC_PDF/04_Homework_Solutions/MIT18_06SCF11_Ses1.12sol.pdf|solution]]

关联卡片：[[Incidence Matrix]]、[[Null Space]]、[[Left Nullspace]]

>[!note] 快速回忆
> - 这讲要回答：矩阵怎样编码 graph / network 结构。
> - 你要立刻想起：incidence matrix 把节点与边连起来；零空间和左零空间会对应回路与约束。
> - 典型题型：写 incidence matrix、解释其 rank 和 nullspace。
> - 它接到下一讲：Exam 1 前把算法线与结构线收成一个闭环。

### 网络中的矩阵

对一个图或网络，可以给每条边任意指定方向，然后构造 [[Incidence Matrix]]。矩阵的列对应边，行对应节点。若一条边从节点 $i$ 指向节点 $j$，对应列通常在第 $i$ 行放 $-1$、第 $j$ 行放 $1$，其余位置为 0。

![[98_attachment/linear_algebra/mit18_06sc/mit18.06sc-unit1-incidence-network.svg|760]]

把图变成矩阵以后，离散网络就被送进了线性代数的统一语言。你不再只是“看见三条边和三个点”，而是在看一个从边空间到点空间的线性映射。

### 关联矩阵为什么重要

它把图结构直接翻译成线性代数语言：

- 列空间描述可能的净流入/净流出模式。
- 零空间描述循环流，也就是沿回路流动而在每个节点守恒的边流。
- 左零空间反映节点势能中“不改变所有边差值”的自由度，例如整体加一个常数。

### 连接到物理与工程

电路里的 Kirchhoff 定律、网络流、离散梯度和势函数，都可以写成关联矩阵问题。一个连通图的关联矩阵通常秩为 $n-1$，因为所有行加起来为 0；这说明“总流量守恒”带来一条必然依赖。

### 一个核心例子

若 $B$ 是图的 incidence matrix，则
$$
B^Ty
$$
给出节点势函数 $y$ 在各条边上的差值；而
$$
Bx
$$
则把边流 $x$ 变成各节点的净流量。这个“从点到边”和“从边到点”的结构，是离散数学和偏微分方程离散化里的基础模式。

这组式子之所以重要，是因为它把“梯度”和“散度”的离散版本直接写了出来。$B^Ty$ 像是把节点势能差分成边上的落差，$Bx$ 像是把边上的流量汇总成节点净流入。于是零空间和左零空间不再只是抽象空间名称，而是在网络里分别对应“绕圈流动却不积累”和“整体加常数不改变边差”的具体现象。

### 你要掌握

- 知道 incidence matrix 的行和列分别表示什么。
- 能解释为什么连通图的 incidence matrix 往往少一维秩。
- 能把零空间理解成网络中的循环流。

### 回忆检查

- 不看正文，我能说出这讲要解决的问题。
- 我能写出本讲最关键的公式、结论或判别条件。
- 我知道这讲最典型的题型，以及它如何接到下一讲。

## Session 1.14 Exam 1 review

资料：[[MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.14sum.pdf|summary]]

>[!note] 快速回忆
> - 这讲要回答：Exam 1 前最少必须保住哪条链。
> - 你要立刻想起：`几何图像 -> 消元/秩 -> column/null space -> basis/dimension -> four fundamental subspaces -> incidence matrix`。
> - 典型题型：消元、general solution、basis、四个基本子空间、网络矩阵。
> - 复习时如果哪一步说不顺，就直接回对应 session，而不是从头重读整篇。

### Unit I 的核心闭环

Exam 1 前你应该已经把以下链条串起来：

1. 先用 row picture 和 column picture 理解 $Ax=b$。
2. 再用高斯消元找到主元、秩和自由变量。
3. 用 LU 把算法过程整理成矩阵分解。
4. 用列空间和零空间解释“有解”和“不唯一”。
5. 最后用四个基本子空间统一整个单元。

### 一定要会的典型题型

- 对一个矩阵做消元，给出主元列、秩和自由变量。
- 求解 $Ax=0$，写出零空间的一组基。
- 判断 $Ax=b$ 是否可解，并写出全部解。
- 从原矩阵中挑出列空间基，从 rref 中挑出行空间基。
- 写出四个基本子空间的维数和正交关系。
- 对简单网络写出 incidence matrix 并解释其零空间。

### 考前最容易混淆的点

- 列空间的基要从原矩阵取主元列，不是从 rref 直接取列。
- 行空间的基可以从 rref 的非零行取，因为行变换保持行空间。
- 一般解是“特解 + 零空间”，不是“随便找一个解再加常数”。
- 列空间和左零空间活在 $\mathbb{R}^m$，行空间和零空间活在 $\mathbb{R}^n$。

### 你要掌握

- 能把 Unit I 压缩成 `几何图像 -> 消元 -> 子空间 -> 基与维数 -> 四个基本子空间` 这一条链。
- 能从题目表述快速判断该回哪一节，而不是重读整篇。
- 能说清楚 Exam 1 里最常见的计算题和概念题分别在考什么。

### 回忆检查

- 不看正文，我能口头复述 Unit I 的主线推进顺序。
- 我能立即举出 Unit I 最典型的三类题：消元、general solution、four fundamental subspaces。
- 如果我在某一环卡住，我知道应该回哪个 session，而不是只能从头重读。

## 本单元复习清单

- 我能把一个线性系统翻译成 row picture、column picture 和 matrix picture。
- 我能用消元求秩、主元和自由变量，并能解释这些量的几何意义。
- 我能从 rref 写出零空间基，并从原矩阵写出列空间基。
- 我能说清楚四个基本子空间分别是什么、在哪里、维数是多少。
- 我能把“可逆、唯一解、零空间平凡、主元充满、列独立”这些说法互相转换。
- 我能看懂 incidence matrix，并把它和流、势、回路联系起来。
