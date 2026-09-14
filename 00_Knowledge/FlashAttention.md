---
aliases:
  - FlashAttention 是通过分块与在线归一化减少显存往返的精确注意力算法
  - FlashAttention is an exact attention algorithm that reduces device-memory traffic through tiling and online normalization
  - Flash Attention
student_os: knowledge-atom
atom_id: LLM-INF-036
atom_type: definition
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# FlashAttention 是通过分块与在线归一化减少显存往返的精确注意力算法
<!-- bilingual-en:start -->
*FlashAttention is an exact attention algorithm that reduces device-memory traffic through tiling and online normalization*
<!-- bilingual-en:end -->

FlashAttention 将注意力计算切成能在较小、较快的片上存储中处理的块，并维护归一化统计量与累计输出。它仍计算给定 Q/K/V 和可见性规则下的 [[缩放点积注意力]]，但不先把完整分数和权重矩阵写到高带宽显存（HBM）再读回来。

<!-- bilingual-en:start -->
FlashAttention processes attention in blocks that fit in smaller, faster on-chip memory, maintaining normalization statistics and accumulated output. It computes the same [[缩放点积注意力|scaled dot-product attention]] for the specified inputs and visibility rules without first writing full score and weight matrices to HBM and reading them back.
<!-- bilingual-en:end -->

理解它先看一次数据往返：朴素实现计算 $S=QK^\top/\sqrt{d_k}$ 并写出，之后读入 $S$ 计算 $A=\operatorname{softmax}(S)$ 再写出，最后读入 $A$ 与 V 相乘。FlashAttention 在块内完成点积、掩码、归一化和 value 聚合，只保留继续合并所需的状态。单独对每块做 Softmax 后直接平均不对；跨块分母的衔接由 [[在线Softmax合并]] 完成。

<!-- bilingual-en:start -->
A straightforward implementation writes scores, rereads them to write normalized weights, then rereads those weights for value aggregation. FlashAttention combines these operations within blocks and retains the state needed to merge results. Averaging independently normalized blocks is incorrect; [[在线Softmax合并|online Softmax merging]] preserves the common denominator.
<!-- bilingual-en:end -->

这改变的是执行方式。完整稠密读取依然处理同样的合法 query–key 配对；[[注意力计算与显存]]解释了为何减少中间存储不等于消除二次运算量。这里的“精确”表示同一个数学算子，不保证不同浮点执行顺序逐 bit 相同。论文另有 block-sparse 扩展，它改变读取模式，应与这里的精确稠密版本分开。

<!-- bilingual-en:start -->
This changes execution. Dense attention still evaluates the same permitted query–key pairs; [[注意力计算与显存|computation and memory accounting]] explains why less intermediate storage does not eliminate quadratic work. Exactness refers to the mathematical operator, not bitwise identity across floating-point execution orders. The paper's separate block-sparse extension changes the read pattern.
<!-- bilingual-en:end -->

[[KV cache]] 保存跨生成步骤复用的 K/V；FlashAttention 避免的是本次读取中的完整 $S/A$ 中间矩阵。两者可以同时使用。训练反向传播时，原算法还能用保存的统计量和输入重新算局部分数，而不保存整个注意力矩阵；仅做推理时没有这段反向工作。实际首 token 或后续生成收益应按 [[推理性能可比性|对应阶段和输入形状]] 测量，不能套用某次训练跑分。

<!-- bilingual-en:start -->
The [[KV cache]] retains K/V across generation steps; FlashAttention avoids full score/weight intermediates within an attention read. They can coexist. Training additionally recomputes local intermediates during backpropagation, which inference does not perform. Serving gains require [[推理性能可比性|stage- and shape-matched measurement]], not transfer of a training speedup.
<!-- bilingual-en:end -->

## 来源与核验

- [Dao et al. (2022), *FlashAttention*](https://arxiv.org/html/2205.14135v2)，§2.2、§3.1 Algorithm 1、§3.3、Appendix B：支持朴素中间矩阵往返、分块与归一化合并、精确/稀疏版本区别，以及训练反向重算。
- 数学存储边界与复杂度复用 [[注意力计算与显存]]；本卡负责算法入口及数据流。

<!-- bilingual-en:start -->
The cited sections establish matrix traffic, tiling, normalization, the sparse extension, and backward recomputation. This card introduces the algorithm; the shared computation/memory atom owns its storage-complexity boundary.
<!-- bilingual-en:end -->
