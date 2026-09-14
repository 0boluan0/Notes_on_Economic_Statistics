---
aliases:
  - 复用 KV 状态要求所代表的前缀计算与当前请求兼容
  - Reusing KV state requires its prefix computation to be compatible with the current request
  - KV-cache reuse conditions
student_os: knowledge-atom
atom_id: LLM-INF-009
atom_type: boundary
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 复用 KV 状态要求所代表的前缀计算与当前请求兼容

<!-- bilingual-en:start -->
*Reusing KV state requires its prefix computation to be compatible with the current request*
<!-- bilingual-en:end -->

一份 [[KV cache]] 能用于后续计算的条件是：缓存张量仍代表当前模型在当前前缀条件下应使用的键和值。检查对象包括模型权重与 adapter、精确输入前缀、位置与可见性规则、其他参与模型计算的输入，以及缓存表示与运行时的兼容性。文字看起来相同或意思相近，都不足以单独建立这一条件。

<!-- bilingual-en:start -->
A [[KV cache|KV cache]] is reusable when its tensors still represent the keys and values required by the current model under the current prefix conditions. Relevant dependencies include model weights and adapters, exact inputs, positions and visibility, other model inputs, and cache/runtime representation compatibility. Similar meaning or matching visible text alone is insufficient.
<!-- bilingual-en:end -->

## 为什么向右追加通常可复用旧状态

<!-- bilingual-en:start -->
*Why appending usually preserves earlier state*
<!-- bilingual-en:end -->

在固定推理模型中，[[因果注意力掩码]]阻止旧位置读取未来输入。若其他计算条件也不变，在前缀右侧追加 token 不会改变旧位置本来可见的信息；因此旧 K/V 可保留，只需为新增位置计算状态。这个结论依赖既定的因果计算，不能直接套用到会因新输入改变旧表示的双向注意力。

<!-- bilingual-en:start -->
For a fixed inference model, a [[因果注意力掩码|causal attention mask]] prevents earlier positions from reading future inputs. With other computation conditions unchanged, appending tokens to the right preserves the information visible to previous positions, so their KV can be retained. The argument does not automatically apply to bidirectional attention, where new inputs can change earlier representations.
<!-- bilingual-en:end -->

## 共享的是相同计算前缀

<!-- bilingual-en:start -->
*Sharing requires a matching computation prefix*
<!-- bilingual-en:end -->

把 A、B、C、D 当作 token ID。请求一是 [A,B,C]，请求二是 [A,B,D]。在相同模型与位置条件下，两者前两个位置的 KV 可以共享；第三个位置不同，不能共享。若另一请求是 [D,B,C]，虽然末尾也有 C，它的前缀条件已经变化，不能仅按“这个 token 也是 C”复用第三位置的状态。

<!-- bilingual-en:start -->
Treat A, B, C, and D as token IDs. Requests [A,B,C] and [A,B,D] can share KV for their first two positions under matching model and position conditions. Their third positions differ. A request [D,B,C] also ends in C, but its prefix has changed, so that token identity alone cannot justify reusing the third position's state.
<!-- bilingual-en:end -->

同样，改了 adapter 或图像输入后，文字 token 序列相同也不保证中间表示相同。[[Token口径]]解释为何应核对编码后的输入；自定义生成循环还要让位置索引、attention mask 和缓存插入位置正确衔接。

<!-- bilingual-en:start -->
Changing an adapter or image input can likewise change representations even when text tokens match. [[Token口径|Token-counting conventions]] explain why encoded inputs must be checked. A custom generation loop must also align position indices, attention masks, and cache insertion positions.
<!-- bilingual-en:end -->

## 可复用与实际命中是两道条件

<!-- bilingual-en:start -->
*Reusability and a cache hit are separate conditions*
<!-- bilingual-en:end -->

跨请求的 prefix caching 还要求运行时已经存有匹配状态、状态尚未驱逐，而且匹配与共享规则允许使用。vLLM 的设计用前块 hash、本块精确 token 与 LoRA/多模态等额外标识匹配缓存块；这是将计算依赖纳入缓存身份的一种实现。合法共享后，分支续写仍须保护共有状态，见[[分页KV占用|共享块的写时复制]]。

<!-- bilingual-en:start -->
Cross-request prefix caching also requires matching state to be resident, not evicted, and eligible under the runtime's matching and sharing rules. vLLM combines the parent hash, exact block tokens, and extra identifiers such as LoRA or multimodal inputs. This is one implementation of computation-aware cache identity. Divergent continuations must still protect shared state; see [[分页KV占用|copy-on-write for shared blocks]].
<!-- bilingual-en:end -->

复用的正确性针对兼容计算的结果，不承诺不同硬件或 kernel 顺序下逐 bit 一致。若修改了 KV 的量化、布局或位置约定，需要相应的兼容读取或转换；不能把旧张量原样交给不兼容的计算。

<!-- bilingual-en:start -->
Correct reuse concerns compatible computation, not bitwise identity across hardware or kernel orderings. Changes to KV quantization, layout, or position conventions require compatible reading or conversion rather than passing old tensors unchanged to incompatible computation.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Kwon et al. (2023), PagedAttention](https://arxiv.org/html/2309.06180v1#S2.SS2)，§2.2、§4.4：支持 KV 依赖前缀与位置、共享前缀及分支状态分离。
  <!-- bilingual-en:start -->
  Sections 2.2 and 4.4 establish prefix/position dependence and shared prefixes with separate continuation state.
  <!-- bilingual-en:end -->
- [Hugging Face Transformers v4.57.1, Caching](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)，Attention matrices、Cache class、Cache position：支持因果缓存复用，以及 mask 与位置衔接要求。
  <!-- bilingual-en:start -->
  Supports causal state reuse and the requirements for masks and position alignment.
  <!-- bilingual-en:end -->
- [vLLM, Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)，block hash 的三类组成及多模态例子：支持 parent hash、token、LoRA/多模态身份匹配。将模型和输入依赖概括成“兼容前缀计算”是本卡明确提出的正确性判断。
  <!-- bilingual-en:start -->
  Documents parent hashes, tokens, and LoRA/multimodal identities. Generalizing model and input dependencies into compatible prefix computation is the correctness criterion derived in this note.
  <!-- bilingual-en:end -->
