---
aliases:
  - RoPE 是按位置旋转已投影 query 和 key 的位置编码
  - RoPE encodes positions by rotating projected queries and keys
  - Rotary position embedding
  - 旋转位置编码
student_os: knowledge-atom
atom_id: LLM-TF-023
atom_type: definition
status: source-checked
part_of:
  - "[[Transformer.canvas]]"
---

# RoPE 是按位置旋转已投影 query 和 key 的位置编码
<!-- bilingual-en:start -->
*RoPE encodes positions by rotating projected queries and keys*
<!-- bilingual-en:end -->

**RoPE**（rotary position embedding）是一种 [[位置编码]]：先将输入表示投影为 query 和 key，再按各自位置旋转它们。两个旋转后的向量做点积时，显式位置因子表现为相对位移。基本的 QK 形式不要求旋转 value。

<!-- bilingual-en:start -->
**RoPE** (rotary position embedding) is a [[位置编码|positional encoding]] that first projects input representations into queries and keys, then rotates them according to their positions. Their dot product expresses the explicit positional factor as a relative offset. The basic QK construction does not require rotating values.
<!-- bilingual-en:end -->

为写清旋转顺序，本卡局部改用**单 token 列向量**。令 $x_m\in\mathbb R^{d_{\mathrm{model}}}$，$W_q,W_k\in\mathbb R^{d_k\times d_{\mathrm{model}}}$，则

$$
q_m=W_qx_m,\qquad k_n=W_kx_n,
\qquad
\widetilde q_m=R_mq_m,\qquad\widetilde k_n=R_nk_n.
$$

这里 $R_m\in\mathbb R^{d_k\times d_k}$；因此是旋转 [[缩放点积注意力|投影后的 Q/K]]，而非直接给输入 $x_m$ 加位置向量。

<!-- bilingual-en:start -->
This card locally uses **single-token column vectors** to make the operation order explicit. With the stated dimensions, the displayed equations first project $x_m$ and then rotate the resulting [[缩放点积注意力|queries and keys]]. The rotation matrix has shape $d_k\times d_k$; it is not an additive position vector applied to $x_m$.
<!-- bilingual-en:end -->

取偶数 $d_k$，将坐标分成相邻的二维对。每对使用一个旋转块，所有块组成块对角矩阵：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix},
\qquad
R_m=\operatorname{diag}\bigl(R(m\theta_0),\ldots,R(m\theta_{d_k/2-1})\bigr),
$$

$$
\theta_r=10000^{-2r/d_k},\qquad r=0,\ldots,d_k/2-1.
$$

<!-- bilingual-en:start -->
For even $d_k$, pair adjacent coordinates and form a block-diagonal matrix from the two-dimensional rotations shown above. Each pair uses its own frequency; the displayed geometric frequency schedule is the original RoPE construction written with zero-based indices.
<!-- bilingual-en:end -->

旋转满足 $R_m^\top R_n=R_{n-m}$，所以

$$
\begin{aligned}
\widetilde q_m^\top\widetilde k_n
&=(R_mW_qx_m)^\top(R_nW_kx_n)\\
&=x_m^\top W_q^\top R_m^\top R_nW_kx_n\\
&=q_m^\top R_{n-m}k_n.
\end{aligned}
$$

位置通过 $n-m$ 进入**旋转项**；内容仍由 $q_m,k_n$ 决定。给定这两个内容向量，把 $m,n$ 同时平移相同距离不会改变该项，这并不意味着带上下文表示、边界与掩码的整个模型平移不变。

<!-- bilingual-en:start -->
The rotation identity $R_m^\top R_n=R_{n-m}$ yields the displayed dot-product derivation. Position enters the **rotation term** through $n-m$, while $q_m$ and $k_n$ still carry content. Holding those content vectors fixed, shifting both positions equally preserves the term. This does not establish translation invariance of a whole model with contextual representations, boundaries, and masks.
<!-- bilingual-en:end -->

一个只为手算选取频率的二维例子：取 $\theta=\pi/2$、$q=(1,2)^\top$、$k=(3,4)^\top$，位置 $m=1,n=2$。旋转后是 $(-2,1)^\top$ 与 $(-3,-4)^\top$，点积为 $2$；相对旋转算式也给出 $q^\top R(\pi/2)k=2$。若误把 $n-m$ 换成 $m-n$ 而保持其他约定不变，结果是 $-2$。

<!-- bilingual-en:start -->
For a hand-computable example, choose the illustrative frequency $\theta=\pi/2$, vectors $q=(1,2)^\top$ and $k=(3,4)^\top$, and positions $m=1,n=2$. The rotated vectors are $(-2,1)^\top$ and $(-3,-4)^\top$, with dot product $2$, matching $q^\top R(\pi/2)k$. Replacing $n-m$ with $m-n$ while retaining the other conventions incorrectly gives $-2$.
<!-- bilingual-en:end -->

RoPE 与 [[正弦位置编码]] 都可使用多频率 sin/cos，但一个旋转 Q/K，一个生成加到输入的向量。仅凭旋转公式，也不能推出任意内容向量的注意力分数随距离单调衰减，或训练长度之外必然可靠。

<!-- bilingual-en:start -->
Both RoPE and [[正弦位置编码|sinusoidal positional encoding]] can use multiple sine/cosine frequencies, but one rotates Q/K while the other generates vectors added to inputs. The rotation formula alone neither implies monotonic score decay for arbitrary content vectors nor guarantees reliability beyond the training length.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Su et al., *RoFormer*, v5](https://arxiv.org/html/2104.09864v5#S3.SS2)，§3.2.1–3.2.2 Eq. 13–15：支持投影后旋转、二维块和频率构造。本卡点积式直接从 Eq. 14 的列向量定义展开，逐项核对转置、维度及 $n-m$ 的符号；该版本 Eq. 16 末项的下标和转置排印不完整。手算例子使用明确另选的演示频率。

<!-- bilingual-en:start -->
- Su et al., §3.2.1–3.2.2 Equations 13–15, defines rotation after projection, the two-dimensional blocks, and the frequencies. The dot product here is derived directly from Equation 14 with checked transposes, dimensions, and offset sign; the final term of Equation 16 in this version has incomplete subscript and transpose notation. The example explicitly uses a separate illustrative frequency.
<!-- bilingual-en:end -->
