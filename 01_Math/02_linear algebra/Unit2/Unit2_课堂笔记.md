---
aliases: []
tags: [linear-algebra, MIT-18.06SC, unit2]
date: 2026-02-09
科目: Math
---

# Unit2 课堂笔记（MIT 18.06SC）

## Unit 导航

| Session | 主题 | Summary PDF |
|---|---|---|
| Session 2.1 | 正交向量与正交子空间 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf|Ses2.1sum.pdf]] |
| Session 2.2 | 子空间投影 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf|Ses2.2sum.pdf]] |
| Session 2.3 | 投影矩阵与最小二乘 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf|Ses2.3sum.pdf]] |
| Session 2.4 | 正交矩阵与 Gram-Schmidt | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|Ses2.4sum.pdf]] |
| Session 2.5 | 行列式性质 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.5sum.pdf|Ses2.5sum.pdf]] |
| Session 2.6 | 余子式与行列式公式 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|Ses2.6sum.pdf]] |
| Session 2.7 | 克拉默法则、逆与体积 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|Ses2.7sum.pdf]] |
| Session 2.8 | 特征值与特征向量 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.8sum.pdf|Ses2.8sum.pdf]] |
| Session 2.9 | 对角化与 A 的幂 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.9sum.pdf|Ses2.9sum.pdf]] |
| Session 2.10 | 微分方程与矩阵指数 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.10sum.pdf|Ses2.10sum.pdf]] |
| Session 2.11 | 马尔可夫矩阵与傅里叶级数 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|Ses2.11sum.pdf]] |
| Session 2.12 | Unit2 考前复盘 | [[../MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.12sum.pdf|Ses2.12sum.pdf]] |

## Session 笔记

### Session 2.1 正交向量与正交子空间

> 对应 Summary: *Orthogonal vectors and subspaces*

**核心概念**
- 正交是内积结构：`u^Tv=0` 表示无投影耦合。
- 子空间的正交补 `S^\perp` 收集所有与 `S` 正交的向量。
- 正交分解是最小二乘和傅里叶展开的核心。

**关键公式**
$$
u^Tv=0
$$
$$
\mathbb{R}^n=S\oplus S^\perp
$$

**几何/直觉解释**
- 正交基让坐标互不干扰，计算最稳定。
- “误差与模型子空间正交”是拟合最优的几何判据。
- 正交补给出“解释不了的部分”。

**易错点**
- 把“垂直”只理解成二维图像。
- 未规范化就直接当正交标准基使用。
- 忘记内积定义可能带权。

**1道例题（含简解）**
- 题：`u=(1,2,2), v=(2,-1,0)` 是否正交？
- 解：`u^Tv=2-2+0=0`，故正交。
- 若再单位化可构造正交标准基。

### Session 2.2 子空间投影

> 对应 Summary: *Projections onto subspaces*

**核心概念**
- 向量投影到子空间是“最短距离逼近”。
- 对单向量方向 `a` 的投影是 `p=rac{a^Tb}{a^Ta}a`。
- 投影残差与子空间正交。

**关键公式**
$$
p=\frac{a^Tb}{a^Ta}a
$$
$$
e=b-p,\ a^Te=0
$$

**几何/直觉解释**
- 投影把不可达目标分解为“可解释部分+误差”。
- 误差正交意味着任何可达方向都无法再降低误差。
- 这是最优化的一阶条件几何版。

**易错点**
- 分母写成 `\|a\|` 而非 `a^Ta`。
- 把投影到“向量”与投影到“子空间”混写。
- 漏掉残差正交检查。

**1道例题（含简解）**
- 题：将 `b=(3,1)` 投影到 `a=(1,1)`。
- 解：系数 `c=(a^Tb)/(a^Ta)=4/2=2`，`p=(2,2)`。
- 残差 `e=(1,-1)` 与 `a` 点积为0。

### Session 2.3 投影矩阵与最小二乘

> 对应 Summary: *Projection matrices and least squares*

**核心概念**
- 列满秩时最小二乘解满足正规方程。
- 投影矩阵 `P=A(A^TA)^{-1}A^T` 把 `b` 映到 `C(A)`。
- `P` 对称且幂等。

**关键公式**
$$
A^TA\hat{x}=A^Tb
$$
$$
P=A(A^TA)^{-1}A^T,\ P^2=P
$$

**几何/直觉解释**
- 最小二乘不是“解方程”，而是“找最近可达点”。
- `\hat{b}=Pb` 是模型解释部分，残差在 `N(A^T)`。
- 投影矩阵把几何条件编码成代数算子。

**易错点**
- `A^TA` 不可逆时仍硬算，应改用 QR 或伪逆。
- 把 `P` 当可逆矩阵。
- 把残差最小与方程精确满足混淆。

**1道例题（含简解）**
- 题：拟合 `ypprox c` 到数据 `1,2,4`。
- 解：`A=[1,1,1]^T,b=[1,2,4]^T`，`\hat{c}=(1+2+4)/3=7/3`。
- 即常数最小二乘是均值。

### Session 2.4 正交矩阵与 Gram-Schmidt

> 对应 Summary: *Orthogonal matrices and Gram-Schmidt*

**核心概念**
- 正交矩阵满足 `Q^TQ=I`，保长度、保角度。
- Gram-Schmidt 把线性无关向量组转为正交（标准）基。
- 由此得到 `A=QR`，连接最小二乘高效计算。

**关键公式**
$$
Q^TQ=I
$$
$$
A=QR
$$

**几何/直觉解释**
- 正交化像“去相关”，让各坐标独立。
- `R` 收集原向量在正交基下的坐标。
- 正交矩阵数值稳定，是计算首选。

**易错点**
- 忘记每步减去“所有已生成方向”的投影。
- 只正交不单位化。
- 把列正交和行正交混淆。

**1道例题（含简解）**
- 题：`a1=(1,1,0), a2=(1,0,1)` 做 Gram-Schmidt 第一步。
- 解：`q1=a1/||a1||=(1,1,0)/\sqrt2`；`u2=a2-(q1^Ta2)q1`。
- 继续单位化得 `q2`。

### Session 2.5 行列式性质

> 对应 Summary: *Properties of determinants*

**核心概念**
- 行列式衡量线性变换对体积与方向的缩放。
- 三条基本行操作对应行列式的可预测变化。
- `\det(A)=0` 等价于列相关与不可逆。

**关键公式**
$$
\det(AB)=\det(A)\det(B)
$$
$$
\det(A^T)=\det(A)
$$

**几何/直觉解释**
- 体积缩放为0意味着某个维度被压扁。
- 符号正负对应方向是否翻转。
- 乘法性质让复杂变换体积变化可分解。

**易错点**
- 把“某行乘 c”误写成“det 加 c”。
- 交换两行不改符号（错误）。
- 把非方阵也谈 det。

**1道例题（含简解）**
- 题：`A=\begin{bmatrix}1&2\3&4\end{bmatrix}`。
- 解：`\det(A)=1\cdot4-2\cdot3=-2`。
- 结论：可逆且发生方向翻转。

### Session 2.6 余子式与行列式公式

> 对应 Summary: *Determinant formulas and cofactors*

**核心概念**
- 代数余子式展开给出递归计算行列式的方法。
- 伴随矩阵与逆矩阵公式相关，但数值上不如消元稳。
- 理解 cofactor 有助于理论推导。

**关键公式**
$$
\det(A)=\sum_j a_{ij}C_{ij}
$$
$$
A^{-1}=\frac{1}{\det(A)}\operatorname{adj}(A)
$$

**几何/直觉解释**
- 余子式展开本质是在做“按一行加权的低维体积拆分”。
- 稀疏行/列展开会显著降算量。
- 理论证明常用 cofactor，工程计算多用 LU。

**易错点**
- 符号 `(-1)^{i+j}` 漏写。
- 把 minor 与 cofactor 混用。
- 在大矩阵上硬用展开计算。

**1道例题（含简解）**
- 题：对 `A=\begin{bmatrix}2&1&0\1&3&1\0&1&2\end{bmatrix}` 以第一行展开。
- 解：`\det(A)=2\det\begin{bmatrix}3&1\1&2\end{bmatrix}-1\det\begin{bmatrix}1&1\0&2\end{bmatrix}=2(5)-2=8`。
- 故 `A` 可逆。

### Session 2.7 克拉默法则、逆与体积

> 对应 Summary: *Cramer’s rule, inverse matrix, and volume*

**核心概念**
- 克拉默法则给出方阵可逆时各变量的显式分式表达。
- 其本质仍是“体积比”解释：替换一列后的体积占比。
- 适合理论与小规模手算，不适合大规模数值计算。

**关键公式**
$$
x_i=\frac{\det(A_i(b))}{\det(A)}
$$
$$
\det(A)\neq0\Rightarrow A^{-1}\text{ exists}
$$

**几何/直觉解释**
- 每个变量都在问：把第 i 个基方向替成目标向量后，体积贡献有多大。
- 分母为0时法则失效，正对应不可逆。
- 把它看成“显式解的存在证明”更合适。

**易错点**
- `A_i(b)` 替换错列。
- 分母接近0仍直接算，数值不稳定。
- 把克拉默法则当通用算法。

**1道例题（含简解）**
- 题：`A=\begin{bmatrix}1&2\3&4\end{bmatrix}, b=(5,11)^T`。
- 解：`\det(A)=-2`；`\det(A_1)= -2, \det(A_2)= -4`。
- 故 `x_1=1,x_2=2`。

### Session 2.8 特征值与特征向量

> 对应 Summary: *Eigenvalues and eigenvectors*

**核心概念**
- 特征向量是方向不变向量，特征值是该方向伸缩倍数。
- 求解来自特征方程 `\det(A-\lambda I)=0`。
- 特征结构决定长期动力学与稳定性。

**关键公式**
$$
Av=\lambda v
$$
$$
\det(A-\lambda I)=0
$$

**几何/直觉解释**
- 一般向量会被旋转+拉伸，特征向量是“只拉不偏”的特殊方向。
- 离散迭代 `x_{k+1}=Ax_k` 的主导行为由最大模特征值决定。
- 系统稳定常看谱半径是否小于1。

**易错点**
- 把特征值当矩阵元素。
- 求特征向量时忘记解的是齐次系统。
- 代数重数与几何重数混淆。

**1道例题（含简解）**
- 题：`A=\begin{bmatrix}2&0\0&3\end{bmatrix}` 的特征结构。
- 解：特征值 `2,3`；对应特征向量分别沿 `e1,e2`。
- `A^k` 直接对角元幂次化。

### Session 2.9 对角化与 A 的幂

> 对应 Summary: *Diagonalization and powers of A*

**核心概念**
- 若 `A` 有 n 个线性无关特征向量，则可对角化。
- 对角化把复杂幂运算转成对角幂，极大简化迭代分析。
- 不可对角化时需更一般的 Jordan 结构。

**关键公式**
$$
A=S\Lambda S^{-1}
$$
$$
A^k=S\Lambda^k S^{-1}
$$

**几何/直觉解释**
- 换到特征基后，系统各坐标独立演化。
- 对角化就是“找到最适合这类变换的坐标系”。
- 幂次增长速度由 `|\lambda_i|` 决定。

**易错点**
- 把“有特征值”误当“可对角化”。
- `S` 不是正交矩阵时误用 `S^{-1}=S^T`。
- 忽略复特征值情形。

**1道例题（含简解）**
- 题：`A=\begin{bmatrix}1&1\0&2\end{bmatrix}`，求 `A^k` 思路。
- 解：先求特征值 `1,2` 与特征向量，组 `S,\Lambda`。
- 再用 `A^k=S\Lambda^kS^{-1}`。

### Session 2.10 微分方程与矩阵指数

> 对应 Summary: *Differential equations and e At*

**核心概念**
- 线性系统 `x'(t)=Ax(t)` 的解由矩阵指数给出。
- 若可对角化，`e^{At}=Se^{\Lambda t}S^{-1}`。
- 连续时间稳定性由特征值实部决定。

**关键公式**
$$
x\'(t)=Ax(t)
$$
$$
x(t)=e^{At}x(0)
$$

**几何/直觉解释**
- 矩阵指数是标量指数在矩阵上的推广。
- 每个特征方向按 `e^{\lambda t}` 演化。
- 实部负则衰减，实部正则爆发。

**易错点**
- 把 `e^{At}` 当逐元素指数。
- 忘记 `A` 与 `B` 不交换时 `e^{A+B}
eq e^Ae^B`。
- 把离散稳定判据误用于连续系统。

**1道例题（含简解）**
- 题：`x'=-2x` 的矩阵形式 `A=[-2]`。
- 解：`x(t)=e^{-2t}x(0)`。
- 二维对角情形同理逐分量指数衰减。

### Session 2.11 马尔可夫矩阵与傅里叶级数

> 对应 Summary: *Markov matrices; Fourier series*

**核心概念**
- 马尔可夫矩阵列（或行）和为1，描述概率转移。
- 稳态分布满足 `A\pi=\pi`，对应特征值1。
- 傅里叶级数可视为在正交基上的投影展开。

**关键公式**
$$
A\pi=\pi
$$
$$
f(t)=\sum_k c_k\phi_k(t)
$$

**几何/直觉解释**
- 反复转移后系统趋向主特征结构。
- 傅里叶把复杂信号拆成频率基向量。
- 两者都体现“基展开 + 主模态主导”思想。

**易错点**
- 把转移矩阵归一化方向弄反。
- 忽略不可约性导致稳态不唯一。
- 傅里叶系数内积计算漏掉归一化常数。

**1道例题（含简解）**
- 题：两状态转移 `A=[[0.9,0.2],[0.1,0.8]]`（列随机），求稳态。
- 解：解 `A\pi=\pi,\ \pi_1+\pi_2=1`，得 `\pi=(2/3,1/3)`。
- 长时间分布趋近该向量。

### Session 2.12 Unit2 考前复盘

> 对应 Summary: *Exam 2 Review*

**核心概念**
- Unit2 主线：正交/投影/行列式/谱分解。
- 重点在“几何条件 ↔ 代数方程”双向转写。
- 考试常把多个主题混在一道综合题。

**关键公式**
$$
A^TA\hat{x}=A^Tb
$$
$$
A=S\Lambda S^{-1}
$$

**几何/直觉解释**
- 先判结构（对称? 正交? 可对角化?）再选工具。
- 同一题可用投影法或分解法交叉验证。
- 记住“谱信息控制动力学”。

**易错点**
- 把正交投影和一般最小二乘解步骤混写。
- 行列式技巧过度使用导致失误。
- 忽略题目中的稳定性问法。

**1道例题（含简解）**
- 题：给对称矩阵 `A`，解释为什么可正交对角化。
- 解：`A=A^T` 保证存在正交特征向量基，故 `A=Q\Lambda Q^T`。
- 由此推最小/最大二次型值。

## Unit 总结

### 主线回顾
- 正交与投影进入最小二乘
- 行列式、特征值与动力学
- 对角化与连续系统建模

### 与下一 Unit 的衔接
- 下一单元会在当前结构上加入更强的几何解释与数值算法视角。
