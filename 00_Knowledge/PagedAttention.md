---
aliases:
  - PagedAttention 通过块表读取可位于非连续物理内存中的 KV 块来计算注意力
  - PagedAttention computes attention by using block tables to read KV blocks that may be physically noncontiguous
student_os: knowledge-atom
atom_id: LLM-INF-013
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# PagedAttention 通过块表读取可位于非连续物理内存中的 KV 块来计算注意力

<!-- bilingual-en:start -->
*PagedAttention computes attention by using block tables to read KV blocks that may be physically noncontiguous*
<!-- bilingual-en:end -->

**PagedAttention** 是一种支持分页 [[KV cache]] 的注意力算法。它把逻辑上的token历史组织成固定大小的KV块；内核根据块表找到实际存放K/V的物理块，再完成注意力读取。因此，一个请求逻辑连续的历史，不必占用物理连续的一大段内存。

<!-- bilingual-en:start -->
**PagedAttention** is an attention algorithm that supports a paged [[KV cache|KV cache]]. Logical token history is organized into fixed-size KV blocks. A kernel follows a block table to the physical K/V blocks and performs attention reads, so a logically contiguous history need not occupy one contiguous physical region.
<!-- bilingual-en:end -->

## 逻辑位置保持顺序，物理地址可以分散

<!-- bilingual-en:start -->
*Logical positions stay ordered while physical addresses may be scattered*
<!-- bilingual-en:end -->

构造一个每块容纳4个token、已缓存10个位置的请求。块表可以是：

<!-- bilingual-en:start -->
Consider a request caching ten positions with four tokens per block. Its block table may be:
<!-- bilingual-en:end -->

| 逻辑块 | 对应token位置 | 物理块编号 | 已填位置数 |
|---|---|---|---|
| 0 | 1–4 | 7 | 4 |
| 1 | 5–8 | 1 | 4 |
| 2 | 9–10 | 9 | 2 |

<!-- bilingual-en:start -->
Logical blocks 0, 1, and 2 map to physical blocks 7, 1, and 9, holding four, four, and two positions respectively. Physical adjacency is unnecessary because the table preserves the mapping.
<!-- bilingual-en:end -->

当前query按映射取回允许读取的K/V。数学上仍执行[[缩放点积注意力]]对全部合法key的归一化；不能把每块各自归一化后的输出简单平均，当作全上下文attention的结果。

<!-- bilingual-en:start -->
The current query uses this mapping to retrieve visible K/V. Mathematically, [[缩放点积注意力|scaled dot-product attention]] still normalizes over all allowed keys. Simply averaging separately normalized block outputs would not compute the same full-context attention.
<!-- bilingual-en:end -->

## 地址映射带来分配与共享能力

<!-- bilingual-en:start -->
*Address mapping enables allocation and sharing*
<!-- bilingual-en:end -->

新增KV写满尾块后，管理器可再分配任一空闲物理块，把它记到请求块表中，而不用为未知最终长度预先找到足够大的连续空间。多个请求也可在满足[[KV复用条件]]时映射到同一前缀块；具体物理占用与分支写入规则见[[分页KV占用]]。

<!-- bilingual-en:start -->
When a growing cache fills its last block, the manager can allocate any free physical block and add its mapping, without first finding a large contiguous region for an unknown final length. Requests can also map to a shared prefix block when [[KV复用条件|reuse conditions]] hold. [[分页KV占用|Paged KV allocation]] covers physical occupancy and writes after branching.
<!-- bilingual-en:end -->

分页没有删去可见历史，所以不自动缩小每个token的[[KV缓存容量|有效KV字节数]]或消除随历史增长的attention读取。实际内存传输和速度仍受块大小与kernel影响；它的定义性改变在于KV布局与地址访问。

<!-- bilingual-en:start -->
Paging does not remove visible history, so it does not automatically reduce [[KV缓存容量|effective KV bytes per token]] or eliminate attention reads over growing history. Actual memory traffic and speed still depend on block size and kernel behavior; the defining change is KV layout and address access.
<!-- bilingual-en:end -->

[[注意力计算与显存]]讨论是否物化完整attention中间矩阵，是另一个层面。原vLLM示例在prefill使用常规self-attention算法，再在后续生成中按块读取KV；支持分页不意味着整个请求的每个阶段都必须调用同一个kernel。

<!-- bilingual-en:start -->
[[注意力计算与显存|Attention computation and memory]] addresses whether full intermediate attention matrices are materialized, a separate question. The original vLLM example uses a conventional self-attention algorithm for prefill and paged KV reads for later generation. Supporting paging does not require every phase to use one kernel.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Kwon et al. (2023), PagedAttention](https://arxiv.org/html/2309.06180v1#S4.SS1)，§4.1–4.3、Fig.5–6：支持非连续物理块、块表读取、按需增长，以及prefill与生成阶段的实现区分。4位置块表为本卡独立示意；归一化要求来自完整attention算子。
  <!-- bilingual-en:start -->
  Sections 4.1–4.3 and Figures 5–6 establish noncontiguous blocks, table-based access, growth, and phase-specific execution. The four-position block example is constructed here. Normalization follows the full attention operator.
  <!-- bilingual-en:end -->
