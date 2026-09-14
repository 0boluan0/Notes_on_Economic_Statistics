---
aliases:
  - KV cache 是推理时保留已处理位置的逐层注意力键和值以供后续查询读取的状态
  - A KV cache stores layer-wise attention keys and values for processed positions so later queries can reuse them
  - Key-value cache
student_os: knowledge-atom
atom_id: LLM-INF-008
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# KV cache 是推理时保留已处理位置的逐层注意力键和值以供后续查询读取的状态

<!-- bilingual-en:start -->
*A KV cache stores layer-wise attention keys and values for processed positions so later queries can reuse them*
<!-- bilingual-en:end -->

**KV cache** 是自回归注意力推理中的已计算状态：为每层保留已处理位置的 key 与 value，供后续位置的 query 读取。它保存的是模型计算出的张量，不是原始文本，也不是模型权重。[[增量解码]]利用这些状态，避免每一步重新计算整个前缀。

<!-- bilingual-en:start -->
A **KV cache** is previously computed state used in autoregressive attention inference. It retains the keys and values of processed positions at each layer for later queries to read. These are computed tensors, not raw text or model weights. [[增量解码|Incremental decoding]] reuses them to avoid recomputing the full prefix at every step.
<!-- bilingual-en:end -->

## 当前 query 读取历史与当前的键值

<!-- bilingual-en:start -->
*The current query reads past and current keys and values*
<!-- bilingual-en:end -->

省略 batch 与 head 维度。第 $\ell$ 层已经缓存 $m$ 个位置，记为 $K^{(\ell)}_{1:m},V^{(\ell)}_{1:m}$。新输入位置 $m+1$ 在本层产生自己的 $q,k,v$ 后，[[缩放点积注意力]]读取

<!-- bilingual-en:start -->
Omit batch and head dimensions. At layer $\ell$, denote the cached state for $m$ positions by $K^{(\ell)}_{1:m},V^{(\ell)}_{1:m}$. After new input position $m+1$ produces its own $q,k,v$, [[缩放点积注意力|scaled dot-product attention]] reads
<!-- bilingual-en:end -->

$$
K_{\mathrm{read}}^{(\ell)}=
\begin{bmatrix}K_{1:m}^{(\ell)}\\k_{m+1}^{(\ell)}\end{bmatrix},
\qquad
V_{\mathrm{read}}^{(\ell)}=
\begin{bmatrix}V_{1:m}^{(\ell)}\\v_{m+1}^{(\ell)}\end{bmatrix},
$$

$$
o_{m+1}^{(\ell)}=
\operatorname{softmax}\!\left(
\frac{q_{m+1}^{(\ell)}(K_{\mathrm{read}}^{(\ell)})^\top}{\sqrt{d_k}}
\right)V_{\mathrm{read}}^{(\ell)}.
$$

这里把所有合法历史与当前输入设为可见。新 $k,v$ 同时成为后续步骤的缓存。只需当前 query，是因为这一步要算当前位置的新表示；旧 query 对旧位置做过的读取不必再执行。其他层各自重复这套状态管理。

<!-- bilingual-en:start -->
All valid past positions and the current input are visible in this expression. The new $k,v$ also become cache entries for later steps. Only the current query is needed because this step computes the current position's representation; previous queries' reads need not be repeated. Each layer manages its own state in the same way.
<!-- bilingual-en:end -->

例如2层、3个已处理位置，每层各有3个 key 与3个 value。追加一个输入后，每层各有4个。缓存能够直接追加的前提是旧状态仍正确，具体见[[KV复用条件]]。

<!-- bilingual-en:start -->
For two layers and three processed positions, each layer holds three keys and three values. Processing one additional input grows each set to four. Appending is valid only while existing state remains correct; see [[KV复用条件|KV reuse conditions]].
<!-- bilingual-en:end -->

## 缓存省下重算，也留下读取和容量成本

<!-- bilingual-en:start -->
*Caching avoids recomputation while retaining read and storage costs*
<!-- bilingual-en:end -->

当前 token 仍须经过网络前向，并读取允许访问的历史 KV。常规全注意力的历史越长，需保留的 KV 越多，其[[KV缓存容量|有效字节数]]随已缓存位置增加。滑窗、裁剪或压缩缓存的实现必须另行说明保留范围与表示。

<!-- bilingual-en:start -->
The current token still passes through the network and reads visible historical KV. Longer history in ordinary full attention requires more retained state, increasing [[KV缓存容量|effective KV bytes]]. Sliding-window, truncated, or compressed caches require a separate account of retained positions and representation.
<!-- bilingual-en:end -->

KV 状态与 attention 分数/权重矩阵是不同对象。[[注意力计算与显存]]解释了精确 attention 可避免完整中间矩阵常驻显存；这不自动消除下一步仍需使用的 KV。[[PagedAttention]]进一步改变的是这些 KV 的物理存放与读取方式。

<!-- bilingual-en:start -->
KV state differs from attention score or weight matrices. [[注意力计算与显存|Attention computation and memory]] explains how exact attention can avoid materializing full intermediate matrices, without eliminating KV needed by later steps. [[PagedAttention|PagedAttention]] changes how that KV is physically stored and accessed.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Hugging Face Transformers v4.57.1, Caching](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)，Attention matrices、Cache storage implementation：支持逐层缓存、历史与当前 K/V 拼接及只计算当前 query 的生成路径。
  <!-- bilingual-en:start -->
  Supports layer-wise caching, combining past and current KV, and using the current query during generation.
  <!-- bilingual-en:end -->
- [Kwon et al. (2023), PagedAttention](https://arxiv.org/html/2309.06180v1#S2.SS2)，§2.2、§3：支持 KV 的序列状态含义与容量增长；2层算例为状态计数示意。
  <!-- bilingual-en:start -->
  Sections 2.2 and 3 establish KV as sequence state and explain its capacity growth. The two-layer example illustrates state counts.
  <!-- bilingual-en:end -->
