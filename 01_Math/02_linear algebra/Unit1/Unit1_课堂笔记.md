---
aliases: []
tags: [linear-algebra, MIT-18.06SC, unit1]
date: 2026-02-09
科目: Math
---

# Unit1 课堂笔记（MIT 18.06SC）

## Session 笔记

### Session 1.1 线性方程的几何图像

> 对应 Summary: *The geometry of linear equations*

**核心概念**
- 把线性方程组统一写成 $Ax=b$，并用两种视角理解：row picture（方程交点）与 column picture（列向量线性组合）。
- “有解”对应 `b` 落在 `A` 的列空间 `C(A)`；“唯一解”还要求零空间只有零向量。
- 本节建立整门课最核心问题：线性代数本质上是在研究线性映射与子空间关系。

**关键公式**
$$
Ax=b
$$
$$
b\in C(A)\iff \exists x\text{ s.t. }Ax=b
$$

**几何/直觉解释**
- row picture 关注“几条线/平面是否交于一点”；column picture 关注“目标向量能否被列向量拼出来”。
- 二维里是一组直线，三维里是平面；维数升高后几何图像看不见，但代数结构保持不变。
- 把“解方程”理解成“找坐标”会更稳：`x` 就是 `b` 在列向量生成系统中的坐标。

**易错点**
- 把“有交点”误认为“唯一交点”；平行、重合、欠定都会改变解结构。
- 只在数值层面算消元，不检查 `A` 的列是否足够生成 `b`。
- 忽略 `m×n` 形状信息，导致把方阵结论错误套到非方阵。

**1道例题（含简解）**
- 题：求解 $2x-y=0,\ -x+2y=3$。
- 解：由第一式得 $y=2x$，代入第二式 $-x+4x=3$，故 $x=1,y=2$。
- 检查：$A\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}0\\3\end{bmatrix}=b$，与 row/column 两种视角一致。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 4, 300)
y1 = 2 * x
y2 = 0.5 * x + 1.5

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(x, y1, label="2x-y=0")
ax.plot(x, y2, label="-x+2y=3")
ax.scatter([1], [2], color="red")
ax.annotate("(1,2)", (1, 2), xytext=(1.2, 2.2))
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-2, 4)
ax.set_ylim(-2, 8)
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.2 矩阵消元与主元

> 对应 Summary: *Elimination with matrices*

**核心概念**
- 高斯消元通过初等行变换把矩阵化为阶梯形，上三角/行阶梯形中的主元（pivot）决定秩。
- 消元不改变方程组解集（对增广矩阵同时做行变换），因此是“等价重写”而非“近似”。
- 主元列对应基础方向，非主元列对应自由变量。

**关键公式**
$$
E_k\cdots E_2E_1A=U
$$
$$
\operatorname{rank}(A)=\#\text{pivot columns}
$$

**几何/直觉解释**
- 每一步行变换都在“消掉重复信息”，把约束关系压缩成更容易回代的形状。
- 主元可以看成“新信息出现的位置”；没有主元的列只能依赖已有信息。
- 先消元后回代，本质是把耦合系统拆成逐层可解。

**易错点**
- 把行变换与列变换混用：求解 $Ax=b$ 时默认只做行变换。
- 看到零主元不换行，导致后续除零或错误判断无解。
- 把“零行”直接理解成矛盾；要结合增广列判断是多解还是无解。

**1道例题（含简解）**
- 题：$x+2y+z=2,\ 2x+5y+3z=5,\ x+y+z=1$。
- 解：行变换得 `U` 后回代：先得 $y=1,z=0$，再得 $x=0$。
- 结论：解为 `(0,1,0)`，主元列为第1、2、3列。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot([0, 1, 2], [2, 5, 8], marker="o", label="before elimination")
ax.plot([0, 1, 2], [2, 3, 4], marker="s", label="after elimination")
ax.scatter([0, 1], [2, 3], color="red", label="pivot rows")
ax.set_xlabel("column index")
ax.set_ylabel("value")
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.3 矩阵乘法与逆矩阵

> 对应 Summary: *Multiplication and inverse matrices*

**核心概念**
- 矩阵乘法表示线性变换复合：先做右边变换，再做左边变换。
- 可逆矩阵 $A^{-1}$ 满足 $A^{-1}A=I=AA^{-1}$，意味着变换可完全恢复。
- 逆存在当且仅当矩阵满秩且零空间只有零向量。

**关键公式**
$$
(AB)x=A(Bx)
$$
$$
A^{-1}A=I
$$

**几何/直觉解释**
- “先袜子后鞋子”对应 $(AB)^{-1}=B^{-1}A^{-1}$，顺序一定反过来。
- 把列看作基向量像：若列彼此独立，映射不压扁维度，才可能可逆。
- 求逆可看成“把每个标准基向量都拉回原位”。

**易错点**
- 把 $AB=BA$ 当默认性质；一般不成立。
- 把“行列式非零”当定义而非等价判据。
- 非方阵谈双侧逆，结论会失真。

**1道例题（含简解）**
- 题：$A=\begin{bmatrix}1&3\\2&7\end{bmatrix}$，求 $A^{-1}$。
- 解：$[A\mid I]$ 行变换到 $[I\mid A^{-1}]$，得 $A^{-1}=\begin{bmatrix}7&-3\\-2&1\end{bmatrix}$。
- 验证：$AA^{-1}=I$。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

A = np.array([[1, 3], [2, 7]], dtype=float)
pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
img = (A @ pts.T).T

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(pts[:, 0], pts[:, 1], label="unit square")
ax.plot(img[:, 0], img[:, 1], label="A transform")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.4 A=LU 分解

> 对应 Summary: *Factorization into A = LU*

**核心概念**
- 若消元过程中不需要换行，可把消元乘数收集成下三角矩阵 `L`，得到 $A=LU$。
- `U` 记录“消元后的约束结构”，`L` 记录“如何从 A 走到 U”。
- LU 让多右端向量 $Ax=b_i$ 的求解显著提速。

**关键公式**
$$
A=LU
$$
$$
Ly=b,\ Ux=y
$$

**几何/直觉解释**
- 一次分解，多次回代：先解下三角，再解上三角。
- `L` 像操作日志，`U` 像结果快照。
- 数值计算里 LU 是最核心基建之一。

**易错点**
- 需要换行时仍强行写 $A=LU$；应改用 $PA=LU$。
- 把 `L` 的对角线写错（通常取 1）。
- 把行交换信息遗漏，导致回代错误。

**1道例题（含简解）**
- 题：$A=\begin{bmatrix}2&1\\4&3\end{bmatrix}$。
- 解：消元乘数 $m_{21}=2$，得 $L=\begin{bmatrix}1&0\\2&1\end{bmatrix},\ U=\begin{bmatrix}2&1\\0&1\end{bmatrix}$。
- 故 $A=LU$。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

v = np.array([1, 1], dtype=float)
L = np.array([[1, 0], [2, 1]], dtype=float)
U = np.array([[2, 1], [0, 1]], dtype=float)
uv = U @ v
luv = L @ uv

fig, ax = plt.subplots(figsize=(5, 4))
ax.quiver([0, 0, 0], [0, 0, 0], [v[0], uv[0], luv[0]], [v[1], uv[1], luv[1]], angles="xy", scale_units="xy", scale=1)
ax.text(v[0], v[1], "v")
ax.text(uv[0], uv[1], "Uv")
ax.text(luv[0], luv[1], "LUv")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 8)
ax.grid(True)
plt.show()
```


### Session 1.5 转置与置换矩阵

> 对应 Summary: *Transposes, permutations, spaces Rn*

**核心概念**
- 转置把行列角色对调，连接行空间与列空间。
- 置换矩阵 `P` 表示行重排，`PA` 等价于把 `A` 的行重新排序。
- 在含换行的消元中，标准形式是 $PA=LU$。

**关键公式**
$$
(AB)^T=B^TA^T
$$
$$
PA=LU
$$

**几何/直觉解释**
- $A^T$ 把“对谁做内积”这个动作反向编码。
- 置换矩阵本质是单位矩阵的行重排，几何上是坐标轴重排。
- 很多“对称”性质都通过转置表达：$A=A^T$。

**易错点**
- 误写 $(AB)^T=A^TB^T$。
- 认为置换改变解集；其实它只改变方程顺序。
- 把列置换和行置换混同。

**1道例题（含简解）**
- 题：给 $P=\begin{bmatrix}0&1\\1&0\end{bmatrix}$ 与 $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$，求 $PA$。
- 解：交换两行，$PA=\begin{bmatrix}3&4\\1&2\end{bmatrix}$。
- 说明：$P^{-1}=P^T=P$。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=float)
P = np.array([[0, 1], [1, 0]], dtype=float)
PA = P @ A

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot([1, 2], A[0], marker="o", label="row1(A)")
ax.plot([1, 2], A[1], marker="o", label="row2(A)")
ax.plot([1, 2], PA[0], "--", marker="s", label="row1(PA)")
ax.plot([1, 2], PA[1], "--", marker="s", label="row2(PA)")
ax.set_xticks([1, 2])
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.6 列空间与零空间

> 对应 Summary: *Column space and nullspace*

**核心概念**
- 列空间 `C(A)` 是所有 `Ax` 可达向量集合；零空间 `N(A)` 是被映到零向量的输入集合。
- $Ax=b$ 可解当且仅当 $b\in C(A)$。
- 零空间维数刻画“解不唯一”的自由度。

**关键公式**
$$
C(A)=\{Ax:x\in\mathbb{R}^n\}
$$
$$
N(A)=\{x:Ax=0\}
$$

**几何/直觉解释**
- 列空间描述输出能力，零空间描述信息损失。
- 同一个 `b` 的所有解差一个零空间向量。
- “可达 + 不可辨识”共同决定系统行为。

**易错点**
- 把列空间看成行空间。
- 把 `N(A)` 中向量误解为“无意义噪声”；它是结构性自由度。
- 只看方程数量不看秩。

**1道例题（含简解）**
- 题：$A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$，求 $C(A)$ 与 $N(A)$。
- 解：$C(A)=\text{span}\{(1,2)^T\}$；$x_1+2x_2=0$，故 $N(A)=\text{span}\{(-2,1)^T\}$。
- 由此知对某些 `b` 无解，对可解时有无穷多解。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-2, 2, 200)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(t, 2 * t, label="C(A)=span((1,2))")
ax.scatter([1, 2], [2, 4], color="green", label="in C(A)")
ax.scatter([1, 3], [1, 0], color="red", label="not in C(A)")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-2, 2)
ax.set_ylim(-4, 4)
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.7 齐次系统与特解基

> 对应 Summary: *Solving Ax = 0: pivot variables, special solutions*

**核心概念**
- 齐次系统永远有零解，关键是是否存在非零解。
- 自由变量个数等于 `n-r`，每个自由变量可构造一个特解方向。
- 零空间基由这些特解向量组成。

**关键公式**
$$
\dim N(A)=n-r
$$
$$
x_n=c_1v_1+\cdots+c_kv_k
$$

**几何/直觉解释**
- 自由变量是“可以自主选择的坐标”，主变量由约束被动决定。
- 齐次解集一定过原点，是线性子空间。
- 零空间基越多，系统可辨识性越弱。

**易错点**
- 把主变量也当可自由设定。
- 把特解写成单个向量，忘记线性组合系数。
- RREF 回代时符号出错。

**1道例题（含简解）**
- 题：$A=\begin{bmatrix}1&2&3\\2&4&6\end{bmatrix}$，求 $N(A)$。
- 解：约束 $x_1+2x_2+3x_3=0$，设 $x_2=s,x_3=t$，则 $x_1=-2s-3t$。
- 故 $x=s(-2,1,0)^T+t(-3,0,1)^T$。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-3, 3, 200)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(-2 * t, t, label="N(A)=span((-2,1))")
ax.scatter([0], [0], color="red")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-6, 6)
ax.set_ylim(-3, 3)
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.8 非齐次系统通解

> 对应 Summary: *Solving Ax = b: row reduced form R*

**核心概念**
- $Ax=b$ 一旦可解，通解可写成“一个特解 + 零空间通解”。
- RREF 同时给出可解性判据与参数化表达。
- 几何上是仿射子空间：平移后的线性子空间。

**关键公式**
$$
x=x_p+x_n
$$
$$
Ax=b\Rightarrow A(x_p+x_n)=b
$$

**几何/直觉解释**
- 先找一个落点 $x_p$，再沿 `N(A)` 方向移动仍留在解集。
- 这解释了“多解”不是离散点，而是一条线/平面。
- 零空间是解集方向，特解是解集位置。

**易错点**
- 找到特解后停止，漏掉全部通解。
- 把 $x_p$ 误当零空间向量。
- 忽略一致性条件导致“伪通解”。

**1道例题（含简解）**
- 题：$x+y+z=1,\ 2x+2y+2z=2$。
- 解：约束等价于一条平面方程，设 $y=s,z=t$，得 $x=1-s-t$。
- 通解 $x_p=(1,0,0)^T + s(-1,1,0)^T + t(-1,0,1)^T$。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(-2, 2, 200)
x = 1 - t
y = t

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(x, y, label="x+y=1 solutions")
ax.scatter([1], [0], color="red", label="x_p")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.9 线性无关、基与维数

> 对应 Summary: *Independence, basis, and dimension*

**核心概念**
- 线性无关意味着没有向量可由其余向量线性表示。
- 基是“既能张成又线性无关”的最小生成集。
- 维数是基向量个数，与基的选取无关。

**关键公式**
$$
c_1v_1+\cdots+c_kv_k=0\Rightarrow c_i=0
$$
$$
\dim V=\#\text{basis vectors}
$$

**几何/直觉解释**
- 基就是坐标系统；换基只改坐标，不改几何对象。
- 无关性保证表示唯一，张成性保证表示存在。
- 这两件事构成向量表达的“存在-唯一”闭环。

**易错点**
- 把“向量个数多”误解为“更能表示”；冗余会破坏无关。
- 在子空间里用环境空间维数判定。
- 忘记检查零向量会立刻导致线性相关。

**1道例题（含简解）**
- 题：判断 $v1=(1,0,1), v2=(0,1,1), v3=(1,1,2)$ 是否无关。
- 解：$v3=v1+v2$，故相关。
- 可取基 `{v1,v2}`，维数为2。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([1, 1])

fig, ax = plt.subplots(figsize=(5, 4))
ax.quiver([0, 0, 0], [0, 0, 0], [v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], angles="xy", scale_units="xy", scale=1)
ax.text(1, 0, "v1")
ax.text(0, 1, "v2")
ax.text(1, 1, "v3=v1+v2")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 2)
ax.grid(True)
plt.show()
```


### Session 1.10 四个基本子空间

> 对应 Summary: *The four fundamental subspaces*

**核心概念**
- 四个基本子空间：$C(A), N(A), C(A^T), N(A^T)$。
- 秩-零度定理连接列空间与零空间维数。
- 行空间与零空间正交，列空间与左零空间正交。

**关键公式**
$$
\operatorname{rank}(A)+\dim N(A)=n
$$
$$
\dim C(A)=\dim C(A^T)=r
$$

**几何/直觉解释**
- 这是整门课的“地图”：输入端（列/零）与输出端（行/左零）成对出现。
- 正交关系说明“能解释的数据”和“解释不了的残差”互相垂直。
- 很多优化问题都靠这张地图定位。

**易错点**
- 把 $C(A^T)$ 误称为列空间。
- 忽略左零空间 $N(A^T)$，导致最小二乘理解断层。
- 秩的两个定义（pivot 数 / 独立列数）混用不清。

**1道例题（含简解）**
- 题：`A` 为 `3×4` 且 $rank=2$，求四空间维数。
- 解：$dim C(A)=2, dim N(A)=2, dim C(A^T)=2, dim N(A^T)=1$。
- 由维数和约束可快速定位解结构。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 200)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(x, x, label="Row/Column direction")
ax.plot(x, -x, label="Null/Left-null direction")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.grid(True)
ax.legend()
plt.show()
```


### Session 1.11 矩阵空间与秩1分解

> 对应 Summary: *Matrix spaces; rank 1; small world graphs*

**核心概念**
- 矩阵本身也构成向量空间，可在该空间讨论基与维数。
- 秩1矩阵可写为外积 $uv^T$，是复杂矩阵的基础砖块。
- 低秩分解把结构性信息与噪声分离。

**关键公式**
$$
A=\sum_{i=1}^r u_iv_i^T
$$
$$
\operatorname{rank}(uv^T)=1
$$

**几何/直觉解释**
- 每个秩1项表示一个“模式”；叠加多个模式得到完整矩阵。
- 图网络邻接矩阵常可近似低秩，反映社区结构。
- 秩越低，压缩潜力越高。

**易错点**
- 把“元素多”当“秩高”。
- 把任意分解都当最优低秩近似。
- 忽略奇异值衰减特征。

**1道例题（含简解）**
- 题：$A=\begin{bmatrix}2&4\\1&2\end{bmatrix}$ 是否秩1？
- 解：第二列是第一列2倍，故 rank=1，可写 $u=[2,1]^T, v=[1,2]^T$。
- 即 $A=uv^T$。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

c1 = np.array([2, 1])
c2 = np.array([4, 2])

fig, ax = plt.subplots(figsize=(5, 4))
ax.quiver([0, 0], [0, 0], [c1[0], c2[0]], [c1[1], c2[1]], angles="xy", scale_units="xy", scale=1)
ax.text(c1[0], c1[1], "c1")
ax.text(c2[0], c2[1], "c2=2c1")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 3)
ax.grid(True)
plt.show()
```


### Session 1.12 图、网络与关联矩阵

> 对应 Summary: *Graphs, networks, incidence matrices*

**核心概念**
- 图可由关联矩阵/拉普拉斯矩阵表示，网络流约束转为线性方程。
- 节点守恒对应 $Bf=s$（流入流出平衡）。
- 线性代数让图问题转为可计算矩阵问题。

**关键公式**
$$
Bf=s
$$
$$
L=BWB^T
$$

**几何/直觉解释**
- 每条边是一个方向变量，节点方程是守恒律。
- 网络问题常出现一个自由度（整体电位平移）。
- 图连通性与矩阵秩直接相关。

**易错点**
- 边方向任意选但必须全局一致。
- 忽略参考节点导致系统奇异。
- 把加权图当无权图处理。

**1道例题（含简解）**
- 题：三节点链式网络，边流 `f1,f2`，源汇 $s=(1,0,-1)$，写守恒方程。
- 解：关联矩阵 `B` 建立后解 $Bf=s$，得 $f1=f2=1$。
- 说明中间节点净流为0。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

nodes = np.array([[0, 0], [1.5, 1], [3, 0]])
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(nodes[:, 0], nodes[:, 1], s=80)
for i, (x, y) in enumerate(nodes, 1):
    ax.text(x + 0.05, y + 0.05, f"v{i}")
ax.plot(nodes[[0, 1], 0], nodes[[0, 1], 1], "-k")
ax.plot(nodes[[1, 2], 0], nodes[[1, 2], 1], "-k")
ax.arrow(0.2, 0.1, 1.0, 0.6, width=0.01, head_width=0.08)
ax.arrow(1.7, 0.9, 1.0, -0.6, width=0.01, head_width=0.08)
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.8, 1.6)
ax.set_aspect("equal", "box")
ax.axis("off")
plt.show()
```


### Session 1.13 关键思想总览

> 对应 Summary: *An overview of key ideas*

**核心概念**
- 本节回看主线：消元、子空间、秩、可逆性是同一结构的不同切面。
- “方程是否可解”由列空间决定，“解是否唯一”由零空间决定。
- 面向后续考试，强调等价命题链的快速互推。

**关键公式**
$$
A\text{ invertible}\iff \operatorname{rank}(A)=n
$$
$$
\det(A)\neq0\iff A^{-1}\text{ exists}
$$

**几何/直觉解释**
- 把知识点串成因果链比孤立记忆更稳。
- 一题多法（消元/空间/分解）是检查理解深度的标准。
- 先结构后计算，能减少计算失误。

**易错点**
- 把等价命题记成“单向蕴含”。
- 复习只刷题不回顾定义。
- 遇到非方阵仍套方阵判据。

**1道例题（含简解）**
- 题：给 `A` 为 `4×4` 且 $rank=3$，判断可逆与解结构。
- 解：不可逆；`N(A)` 至少1维；$Ax=b$ 可能无解或多解。
- 核心依据是秩信息。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

scales = np.array([0.5, 1, 1.5, 2, 2.5])
vol = np.abs(scales)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(scales, vol, marker="o")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("1D scaling factor a")
ax.set_ylabel("|det([a])|")
ax.grid(True)
plt.show()
```


### Session 1.14 Unit1 考前复盘

> 对应 Summary: *Exam 1 review*

**核心概念**
- 按“方程-空间-分解”复盘 Unit1。
- 优先掌握可解性、唯一性、参数化求解三件事。
- 建立错题标签：维度错、符号错、逻辑错。

**关键公式**
$$
x=x_p+x_n
$$
$$
PA=LU
$$

**几何/直觉解释**
- 考试题通常先给结构信息（秩/主元）再问结论。
- 先写出空间关系，再下计算手。
- 把每题映射到已知模板可提速。

**易错点**
- 临场把 RREF 与 REF 混淆。
- 没做结果回代检查。
- 忽略题目隐含条件（方阵/满秩/正交）。

**1道例题（含简解）**
- 题：已知 `A` 有一个自由变量，写 $Ax=b$ 解集形状。
- 解：若一致，则解集是一条仿射直线 $x_p+t v$；若不一致则无解。
- 判一致性看增广矩阵是否出现矛盾行。
**关键坐标图代码（Python）**
```python
import micropip
await micropip.install("matplotlib")
import matplotlib.pyplot as plt
import numpy as np

rounds = np.arange(1, 8)
error = np.array([0.45, 0.37, 0.31, 0.24, 0.20, 0.17, 0.14])

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rounds, error, marker="o")
ax.set_xlabel("review round")
ax.set_ylabel("error rate")
ax.grid(True)
plt.show()
```


## Unit 总结

### 主线回顾
- $Ax=b$ 建模与几何解释
- 高斯消元、LU、秩与可解性
- 四个基本子空间与网络建模

### 与下一 Unit 的衔接
- 下一单元会在当前结构上加入更强的几何解释与数值算法视角。
