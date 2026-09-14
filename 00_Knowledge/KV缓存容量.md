---
aliases:
  - 常规全注意力的有效 KV 字节数等于各层各请求已缓存位置的键值元素总量
  - Effective KV bytes for ordinary full attention count the stored key and value elements across layers and requests
  - KV-cache payload size
student_os: knowledge-atom
atom_id: LLM-INF-010
atom_type: mechanism
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 常规全注意力的有效 KV 字节数等于各层各请求已缓存位置的键值元素总量

<!-- bilingual-en:start -->
*Effective KV bytes for ordinary full attention count the stored key and value elements across layers and requests*
<!-- bilingual-en:end -->

估算 [[KV cache]] 的**有效字节数**，就是数出真正保留的 K/V 元素，再乘每元素字节数。先考虑各层结构相同的全注意力模型、各请求独立保留全部历史、K/V 头宽相同且无压缩的情况。这个数量是有内容的状态 payload，不包含预留空槽、分页取整或整机其他显存对象。

<!-- bilingual-en:start -->
Estimating **effective KV bytes** means counting retained K/V elements and multiplying by bytes per element. Begin with identical full-attention layers, independently stored request histories, equal key/value head widths, and no compression. This counts populated state payload rather than reserved slots, page rounding, or other device-memory objects.
<!-- bilingual-en:end -->

## 从一个位置逐层相乘

<!-- bilingual-en:start -->
*Count one position, then extend across layers*
<!-- bilingual-en:end -->

记层数为 $n_\ell$，每层 KV head 数为 $h_{\mathrm{kv}}$，每头宽度为 $d_h$，每个元素为 $b$ 字节，请求 $r$ 已缓存位置数为 $S_r$。一个位置、一个头需要 key 与 value 两份长度 $d_h$ 的向量，因此每位置跨全部层的字节数为

<!-- bilingual-en:start -->
Let $n_\ell$ be the layer count, $h_{\mathrm{kv}}$ the KV heads per layer, $d_h$ the head width, $b$ bytes per element, and $S_r$ the cached positions for request $r$. One head at one position stores both a key and a value vector of width $d_h$, giving total bytes per position across all layers
<!-- bilingual-en:end -->

$$
c=2n_\ell h_{\mathrm{kv}}d_hb,
\qquad
M_{\mathrm{eff}}=c\sum_{r=1}^{B}S_r.
$$

若 $B$ 个请求都缓存 $S$ 个位置，得到 $M_{\mathrm{eff}}=2n_\ell BSh_{\mathrm{kv}}d_hb$。这里数的是 KV 头，而不是未核对的 query 头数；标准[[多头注意力]]、[[多查询注意力]]与[[分组查询注意力]]必须按实际张量区分。

<!-- bilingual-en:start -->
If all $B$ requests cache $S$ positions, then $M_{\mathrm{eff}}=2n_\ell BSh_{\mathrm{kv}}d_hb$. Count KV heads rather than assuming the query-head count. Distinguish [[多头注意力|multi-head]], [[多查询注意力|multi-query]], and [[分组查询注意力|grouped-query attention]] through their actual tensors.
<!-- bilingual-en:end -->

## 4 GiB 的计算

<!-- bilingual-en:start -->
*A 4 GiB calculation*
<!-- bilingual-en:end -->

取32层、8个KV头、头宽128、BF16每元素2字节，则每个已缓存位置跨层占

<!-- bilingual-en:start -->
For 32 layers, eight KV heads, width 128, and two bytes per BF16 element, each cached position occupies
<!-- bilingual-en:end -->

$$
c=2\times32\times8\times128\times2
=131{,}072\ \mathrm{bytes}=128\ \mathrm{KiB}.
$$

8个独立请求各缓存4,096位置，共有32,768个请求内位置，所以

<!-- bilingual-en:start -->
Eight independent requests caching 4,096 positions each contain 32,768 request-specific positions, giving
<!-- bilingual-en:end -->

$$
M_{\mathrm{eff}}
=131{,}072\times32{,}768
=4{,}294{,}967{,}296\ \mathrm{bytes}=4\ \mathrm{GiB}.
$$

位置数或并发翻倍、其余不变时，这份有效状态也翻倍。这里 $1\ \mathrm{GiB}=2^{30}$ bytes；不要把十进制 GB 与 GiB 混用。

<!-- bilingual-en:start -->
Doubling positions or concurrency while holding the other factors fixed doubles this effective state. Here $1\ \mathrm{GiB}=2^{30}$ bytes; decimal GB and binary GiB are different units.
<!-- bilingual-en:end -->

## 有效状态与物理占用分别统计

<!-- bilingual-en:start -->
*Count effective state and physical allocation separately*
<!-- bilingual-en:end -->

上述按请求求和假设没有共享。[[KV复用条件|相同前缀]]若共享物理块，物理数据可少于这份独立存储总量；[[分页KV占用|按块分配]]、预分配、对齐或量化 scale 则可增加额外占用。层的KV头数、K/V宽度或保留历史不同，就应分层重新计数，不能无条件套同构公式。

<!-- bilingual-en:start -->
The per-request sum assumes no sharing. Physical sharing of a [[KV复用条件|matching prefix]] can use less data than independent storage, while [[分页KV占用|block allocation]], preallocation, alignment, or quantization scales can add overhead. Layers with different KV heads, key/value widths, or retained histories require separate counts rather than the homogeneous formula.
<!-- bilingual-en:end -->

多 GPU 时还要核对哪些头或层被切分、哪些状态被复制，不能直接把总量除以卡数。权重、临时激活、workspace等也需要显存；这些共同进入[[推理显存预算]]，本卡公式只回答KV有效payload有多大。

<!-- bilingual-en:start -->
With multiple GPUs, check which heads or layers are sharded and which state is replicated instead of dividing blindly by GPU count. Weights, temporary activations, and workspace also consume memory and enter the [[推理显存预算|inference memory budget]] together; this formula answers only the KV-payload question.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Kwon et al. (2023), PagedAttention](https://arxiv.org/html/2309.06180v1#S3)，§3：支持按 K/V、隐藏宽度、层数与精度计算逐 token KV 大小；§4.1 说明各头各层状态的块管理。本卡按 KV head 显式展开并独立计算4GiB示例。
  <!-- bilingual-en:start -->
  Section 3 counts per-token KV using keys/values, width, layers, and precision; Section 4.1 describes head/layer state in blocks. This note expands the KV-head dimension and independently calculates the 4 GiB example.
  <!-- bilingual-en:end -->
- [Hugging Face Transformers v4.57.1, Caching](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)，Cache storage implementation：支持每层缓存按 batch、head、sequence、head dimension 组织，以及不同 cache 对历史长度采用不同规则。
  <!-- bilingual-en:start -->
  Documents layer-wise cache dimensions and different rules for retained sequence length.
  <!-- bilingual-en:end -->
