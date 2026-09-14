---
aliases:
  - 按需分页把独立序列的 KV 容量取整到块边界并把尾部空位限制在一块以内
  - On-demand paging rounds independent sequences' KV allocation to block boundaries and leaves less than one block of tail slack per sequence
student_os: knowledge-atom
atom_id: LLM-INF-014
atom_type: mechanism
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 按需分页把独立序列的 KV 容量取整到块边界并把尾部空位限制在一块以内

<!-- bilingual-en:start -->
*On-demand paging rounds independent sequences' KV allocation to block boundaries and leaves less than one block of tail slack per sequence*
<!-- bilingual-en:end -->

[[PagedAttention]]支持把KV按需分配为固定大小的块。若一个序列总在已有块写满后才申请下一块，则只有最后一块可能未填满。由此可以把[[KV缓存容量|有效KV payload]]换算为已分配的物理块payload；尾块取整仍存在，所以有效字节数与物理占用通常不同。

<!-- bilingual-en:start -->
[[PagedAttention|PagedAttention]] supports allocating fixed-size KV blocks on demand. If a sequence requests a new block only when existing blocks are full, only its final block can be partially filled. This converts [[KV缓存容量|effective KV payload]] into allocated physical-block payload while retaining tail rounding overhead.
<!-- bilingual-en:end -->

## 独立序列的块数公式

<!-- bilingual-en:start -->
*Block counts for independent sequences*
<!-- bilingual-en:end -->

先限定为各层结构相同、全部层合计每个位置占 $c>0$ 字节、每块容纳正整数 $b_{\mathrm{tok}}$ 个位置、$B\ge1$ 条序列互不共享物理KV的情况。序列 $r$ 已缓存 $S_r$ 个位置；忽略额外预留块、元数据和对齐差异，则

<!-- bilingual-en:start -->
Assume homogeneous layers, $c>0$ bytes per position across all layers, a positive integer $b_{\mathrm{tok}}$ positions per block, and $B\ge1$ sequences without physical KV sharing. Sequence $r$ caches $S_r$ positions. Excluding extra reserved blocks, metadata, and alignment differences,
<!-- bilingual-en:end -->

$$
M_{\mathrm{alloc}}
=c\,b_{\mathrm{tok}}
\sum_{r=1}^{B}\left\lceil\frac{S_r}{b_{\mathrm{tok}}}\right\rceil,
\qquad
M_{\mathrm{eff}}=c\sum_{r=1}^{B}S_r.
$$

$$
0\le M_{\mathrm{alloc}}-M_{\mathrm{eff}}
<c\,b_{\mathrm{tok}}B.
$$

第二行来自每条序列剩余位置数严格小于一块。它界定绝对尾部余量，不表示相对浪费比例总是很小。

<!-- bilingual-en:start -->
The second line follows because each sequence has fewer than one block of unused positions. It bounds absolute tail slack, not a universally small relative waste fraction.
<!-- bilingual-en:end -->

## 两条短序列的取整

<!-- bilingual-en:start -->
*Rounding two short sequences*
<!-- bilingual-en:end -->

取块长4，两个序列分别缓存5与9个位置。有效状态是14个位置；物理分配为 $4\lceil5/4\rceil+4\lceil9/4\rceil=8+12=20$ 个位置，6个位置暂时未用，占已分配位置的30%。两条序列各少于4个空位，仍符合尾块界。序列很短时，“最多一个尾块”也可能占明显比例。

<!-- bilingual-en:start -->
With four positions per block, sequences of lengths five and nine contain 14 effective positions but allocate $4\lceil5/4\rceil+4\lceil9/4\rceil=20$. Six positions are unused, or 30% of allocation. Each sequence still has fewer than four unused positions. Even a one-tail-block bound can be a substantial fraction for short sequences.
<!-- bilingual-en:end -->

若已经为KV块payload划出预算 $U$，上述假设下要求

<!-- bilingual-en:start -->
For a budget $U$ already assigned to KV-block payload, the same assumptions require
<!-- bilingual-en:end -->

$$
\sum_r\left\lceil\frac{S_r}{b_{\mathrm{tok}}}\right\rceil
\le
\left\lfloor\frac{U}{c\,b_{\mathrm{tok}}}\right\rfloor.
$$

块表、临时工作区、模型权重等需另外计入[[推理显存预算|整机预算]]，才能确定这里的 $U$。固定大小的KV池可以避免请求大小不同造成的池内外部碎片，但不意味着整块GPU内存管理没有其他浪费。

<!-- bilingual-en:start -->
Block tables, workspace, model weights, and other objects enter the [[推理显存预算|whole-device memory budget]] separately to determine $U$. Fixed-size blocks avoid variable-request-size external fragmentation within the KV pool without eliminating every source of device-memory waste.
<!-- bilingual-en:end -->

## 共享前缀后数唯一物理块

<!-- bilingual-en:start -->
*Count unique physical blocks after prefix sharing*
<!-- bilingual-en:end -->

满足[[KV复用条件]]的两个请求可以把共同前缀映射到同一物理块。此时不能再按每请求的块数直接求和，应数唯一的已分配物理块。如果两条生成分支要写入同一个尚未填满的共享尾块，写入者必须先取得独占副本，或采用其他隔离方式。

<!-- bilingual-en:start -->
Two requests satisfying [[KV复用条件|KV reuse conditions]] can map a shared prefix to one physical block. Allocation must then count unique physical blocks rather than summing request block counts. If divergent branches need to write into a partially filled shared tail block, a writer must first obtain an exclusive copy or equivalent isolation.
<!-- bilingual-en:end -->

原vLLM采用引用计数与写时复制（copy-on-write）：发现待写块有多个使用者，就分配并复制一个块，让该分支写入自己的副本。若共同前缀刚好占满块，后续新token可以直接分别申请新块；不必复制已经完整且不再修改的前缀块。

<!-- bilingual-en:start -->
Original vLLM uses reference counts and copy-on-write: a writer encountering a shared block allocates and copies a block for its own continuation. When the common prefix ends on a full-block boundary, branches can allocate separate new blocks without copying the complete, unmodified prefix blocks.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Kwon et al. (2023), PagedAttention](https://arxiv.org/html/2309.06180v1#S4.SS2)，§4.2–4.4：支持按需块分配、仅尾块未满、物理共享、引用计数与写时复制。ceil公式、容量不等式与5/9长度例子由这些分配规则直接推导；均限定为文中声明的payload口径。
  <!-- bilingual-en:start -->
  Sections 4.2–4.4 establish on-demand allocation, partial tail blocks, physical sharing, reference counts, and copy-on-write. The ceiling formula, budget inequality, and length-five/nine example are derived under the stated payload assumptions.
  <!-- bilingual-en:end -->
