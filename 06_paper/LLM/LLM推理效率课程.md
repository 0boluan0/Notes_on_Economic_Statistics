---
aliases:
  - LLM Inference Efficiency
  - LLM Inference Optimization
  - 大模型推理优化
---

# LLM 推理效率：从一次请求到可用的服务
<!-- bilingual-en:start -->
*LLM inference efficiency: from one request to a usable service*
<!-- bilingual-en:end -->

这一课从一个具体问题开始：你把提示交给模型之后，为什么先等一会儿，随后答案才陆续出现？沿着一次请求往下追，就能把生成步骤、缓存、显存、调度和加速方法放到各自的位置。最终要能说明：某种方法省了什么，它保留或改变了什么，怎样确认这种节省真的改善了使用体验。

<!-- bilingual-en:start -->
Start with one concrete question: after sending a prompt, why is there an initial wait followed by a stream of output? Following a request connects generation steps, caches, memory, scheduling, and acceleration. The aim is to explain what a method saves, what it preserves or changes, and how to verify that the saving improves actual use.
<!-- bilingual-en:end -->

这里主要研究已经训练好的 decoder-only 因果 Transformer。默认例子是单条生成分支；换成投机解码时，会明确指出状态与步骤怎样改变。所有手算时间和字节场景都是教学构造，不是本机或某个模型的跑分。

<!-- bilingual-en:start -->
The main setting is a trained decoder-only causal Transformer with a single generation branch. The speculative-decoding section explicitly changes the baseline's steps and state handling. Worked timing and storage scenarios are teaching constructions, not measurements of this machine or a particular model.
<!-- bilingual-en:end -->

基础不熟时先读 [[06_paper/LLM/Transformer课程|Transformer 课程]]，尤其是语言模型输出头和因果可见性。关系总图在 [[00_Knowledge/LLM 推理效率.canvas|LLM 推理效率 Canvas]]；下面按顺序连续阅读，嵌入内容与独立打开的知识卡是同一个对象。

## 1. 先追踪一个输出 token 是怎样产生的
<!-- bilingual-en:start -->
*Follow how an output token is produced*
<!-- bilingual-en:end -->

提示中的 token 已经给定，模型可以一起计算它们的表示；回答里的后续 token 尚未选出，不能提前当作真实输入。这是下面两个阶段最根本的差别。先特别留意首个输出：处理完提示后，最后位置的输出头已经能给出它的分布。

<!-- bilingual-en:start -->
Prompt tokens are already known, so their representations can be computed together. Later output tokens have not yet been selected and cannot be treated as real inputs in advance. In particular, the prompt's final output head already supplies the first generated token's distribution.
<!-- bilingual-en:end -->

![[预填充]]

首 token 被选出来，不等于它已经作为输入跑过模型。下一步正是把这个已选 token 送进去，并继续预测。

<!-- bilingual-en:start -->
Selecting the first token does not mean it has already been forwarded as an input. The next step feeds it into the model to predict another token.
<!-- bilingual-en:end -->

![[增量解码]]

这样做还要保存什么？不是把整段提示反复重算，也不是把过去所有注意力权重都留下；后续 query 要读的是各层以前产生的 K/V。

<!-- bilingual-en:start -->
What must persist? Neither repeated recomputation of the whole prompt nor every past attention weight: later queries need the K/V previously produced at each layer.
<!-- bilingual-en:end -->

![[KV cache]]

读到这里，试着画三个提示 token、三个输出 token 的时间线，并在每次“选出输出”后标出缓存长度。再把输出数量改成 1：还需要几次增量 forward？这一步若没分清，后面的时间和容量都会差一个位置。

<!-- bilingual-en:start -->
Draw a timeline with three prompt tokens and three outputs, marking cache length after each selection. Then request just one output: how many incremental forwards remain? This distinction controls both timing and storage counts.
<!-- bilingual-en:end -->

## 2. 用户等待与机器执行怎样对应
<!-- bilingual-en:start -->
*Relate user-visible waits to machine execution*
<!-- bilingual-en:end -->

现在给时间线加上观测点。客户端等待还可能包括队列、网络和缓冲，不能把一段 GPU 执行时间直接叫作“用户首 token 延迟”。先逐个确定指标，再把它们组合起来。

<!-- bilingual-en:start -->
Add observation points to the timeline. Client waits may include queues, transport, and buffering; a GPU interval alone is not client time to first token. Define individual metrics before combining them.
<!-- bilingual-en:end -->

![[首Token延迟]]

![[Token间隔]]

流式输出既可以问“某两个 token 之间卡了多久”，也可以问“这次请求后半段平均多快”。后一种平均会掩盖局部停顿，且不同请求的平均不能随意与所有间隔混合平均。

<!-- bilingual-en:start -->
Streaming can be described by an individual gap or a request's average post-first-token pace. The latter may hide stalls, and averaging request averages is not interchangeable with pooling every interval.
<!-- bilingual-en:end -->

![[平均输出Token时间]]

![[请求完成延迟]]

服务系统还同时处理许多请求。单个用户等待多久与整台系统单位时间交付多少，是另外两个视角；吞吐数字增长并不保证用户更快看到答案。

<!-- bilingual-en:start -->
A service handles many requests at once. Individual waiting and total work delivered per unit time are different perspectives; higher throughput need not bring an answer to one user sooner.
<!-- bilingual-en:end -->

![[推理吞吐率]]

![[达标吞吐量]]

## 3. 从可复用状态算到真实显存
<!-- bilingual-en:start -->
*From reusable state to physical memory*
<!-- bilingual-en:end -->

缓存带来两个问题：已有内容还能不能用，以及继续生成时能不能装下。先判断它是否确实代表当前前缀；字符串看起来相同，不足以保证模型、位置和其他输入也相同。

<!-- bilingual-en:start -->
Caches raise two questions: whether existing state is reusable, and whether continued generation fits. Matching visible text is insufficient unless the state represents the current model, positions, and other inputs.
<!-- bilingual-en:end -->

![[KV复用条件]]

![[KV缓存容量]]

上面的 4 GiB 是有效数据量。内存怎么分配是另一层：请求的逻辑顺序可以保持不变，而物理上使用不相邻的块。

<!-- bilingual-en:start -->
The 4 GiB above counts useful payload. Allocation is another layer: a request's logical order can remain intact while its physical blocks are noncontiguous.
<!-- bilingual-en:end -->

![[PagedAttention]]

![[分页KV占用]]

有效 payload、物理块、整台设备上的同时占用，至此是三本不同口径的账。最后把权重和执行工作区也放回去，才有可用的部署预算。

<!-- bilingual-en:start -->
Effective payload, physical blocks, and simultaneous device occupancy are three accounting scopes. Add weights and execution workspace to obtain a usable deployment budget.
<!-- bilingual-en:end -->

![[推理显存预算]]

## 4. 装得下之后，时间花在计算还是搬运
<!-- bilingual-en:start -->
*Once it fits, is time spent on computation or data movement?*
<!-- bilingual-en:end -->

容量回答“能否容纳”，带宽回答“搬得多快”，算力回答“算得多快”。三者不能互换。这里会区分 HBM（高带宽显存）与 L2（片上二级缓存）：同一数据在不同存储层级之间搬运，流量的统计边界也不同。先用一个可计算的比例连接带宽与算力，再看 prefill 与 decode 为何可能出现不同瓶颈。

<!-- bilingual-en:start -->
Capacity determines what fits, bandwidth how quickly data moves, and compute capacity how quickly arithmetic runs. We distinguish HBM (high-bandwidth device memory) from L2 (on-chip level-two cache): traffic counts depend on which memory boundary data crosses. Connect bandwidth and compute capacity with a calculable ratio before examining prefill and decode.
<!-- bilingual-en:end -->

![[算术强度]]

![[Roofline模型]]

![[批量复用权重]]

这也解释了为何不能只数“每次 forward”。一次处理更多已知位置可能增加算术，却更充分地复用权重；与此同时，各请求历史的 K/V 仍须读取。最终要检查实际形状和执行记录。

<!-- bilingual-en:start -->
Counting forwards alone misses the trade-off. Processing more known positions may add arithmetic while improving weight reuse, yet each request's historical K/V still needs to be read. Actual shapes and execution records determine the outcome.
<!-- bilingual-en:end -->

## 5. 多个请求怎样共享执行机会
<!-- bilingual-en:start -->
*How requests share execution opportunities*
<!-- bilingual-en:end -->

合批有利于复用，但请求长度不同，结束时间也不同。调度要决定的是谁在下一轮获得执行机会，而不是把所有请求的上下文混成一条序列。

<!-- bilingual-en:start -->
Batching supports reuse, but requests have different lengths and completion times. Scheduling decides who executes next; it does not merge their contexts into one sequence.
<!-- bilingual-en:end -->

![[静态批处理]]

![[连续批处理]]

允许中途补位之后，新请求仍要先处理提示。一条很长的提示可能占据已有请求等待的那段时间；把提示切成有前后依赖的片段，能给调度器更细的插入机会。

<!-- bilingual-en:start -->
Newly admitted requests still need prompt processing. A long prompt can occupy time during which existing requests wait. Splitting it into dependent chunks gives the scheduler finer interleaving opportunities.
<!-- bilingual-en:end -->

![[分块预填充]]

## 6. 同一个注意力结果能否少搬数据
<!-- bilingual-en:start -->
*Can the same attention result require less data movement?*
<!-- bilingual-en:end -->

现在进入算子内部。先保留相同输入和相同可见集合，只改变计算与存储的顺序；不要把这与后面减少 KV 头数的结构改动混在一起。

<!-- bilingual-en:start -->
Move inside the operator. Keep inputs and visibility fixed while changing computation and storage order. This differs from the later architectural change of reducing KV heads.
<!-- bilingual-en:end -->

![[FlashAttention]]

![[在线Softmax合并]]

在三位置例子中，两个块为什么不能等权平均？能回答这个问题，就能理解“分块但仍精确”靠的是什么。再回到原 Transformer 课程已有的计算/存储边界，确认省下的究竟是哪一项。下面的二次阶针对 $n$ 行 query 读取 $n$ 行 key 的完整注意力；单 token 增量解码只有一行新 query，读取 $S$ 个已处理位置时，核心配对运算随 $S$ 线性增长，见 [[稠密注意力二次成本|按实际矩阵形状核算]]。这不是端到端延迟的线性保证。

<!-- bilingual-en:start -->
Why can't the two blocks in the three-position example be averaged equally? Answering that explains how blocking preserves exactness. Return to the shared Transformer distinction to identify precisely what is saved. The quadratic bound below concerns $n$ query rows attending to $n$ key rows. A single incremental token supplies one query row over $S$ processed positions, so its core pairwise work grows linearly with $S$; [[稠密注意力二次成本|use the actual matrix shape]]. This does not guarantee linear end-to-end latency.
<!-- bilingual-en:end -->

![[注意力计算与显存]]

![[FlashAttention2]]

## 7. 多个 query 头能否共用 K/V
<!-- bilingual-en:start -->
*Can query heads share K/V?*
<!-- bilingual-en:end -->

这次改变的是模型里的投影组织。先看共享的标准多头定义，再比较“头与 KV 组怎样对应”；输出 query 头的数量，不等于缓存中必须保存的 KV 头数量。

<!-- bilingual-en:start -->
Now change how the model organizes projections. Start from the shared standard multi-head definition, then compare query-to-KV-group mappings. Query head count need not equal cached KV head count.
<!-- bilingual-en:end -->

![[多头注意力]]

![[多查询注意力]]

![[分组查询注意力]]

把容量公式中的 KV 头数换成实际组数，就能核算对应状态量。但若拿一个已有 MHA checkpoint 直接合并投影，为什么不能声称只是“压缩存储，输出不变”？

<!-- bilingual-en:start -->
Use the actual group count in the KV formula to account for state. But why can't merging projections in an existing MHA checkpoint be described as storage compression with unchanged outputs?
<!-- bilingual-en:end -->

![[KV共享非等价改写]]

## 8. 更少的位怎样表示数值，又改变了什么
<!-- bilingual-en:start -->
*How fewer bits represent values, and what changes*
<!-- bilingual-en:end -->

头共享改了投影结构，量化则把数值放到有限的表示等级上。先亲手做一次编码和解码，再看范围、共享参数与训练流程；这样读到 W4A16 时，才能知道它真正规定了哪些东西。

<!-- bilingual-en:start -->
Head sharing changes projections; quantization maps values to finite representational levels. Work through encoding and decoding before considering ranges, shared parameters, and training procedures, so a label such as W4A16 becomes interpretable.
<!-- bilingual-en:end -->

![[数值量化]]

![[量化舍入与截断]]

![[量化粒度]]

有了目标网格，可以从已有模型寻找合适表示，也可以在训练时让参数适应网格。两条路线的区别是怎样得到可部署模型，不是每个算子最终都用什么位宽。

<!-- bilingual-en:start -->
Given a target grid, one can derive a representation from an existing model or train parameters to adapt to it. These routes concern how deployment parameters are obtained, not the eventual precision of every operator.
<!-- bilingual-en:end -->

![[训练后量化]]

![[量化感知训练]]

![[量化对象]]

编码误差接下来要经过真实输入和多层计算。仅知道权重误差或压缩比例，还不能回答任务质量和速度这两个问题。

<!-- bilingual-en:start -->
Encoding error interacts with actual inputs and subsequent computation. Weight error or compression ratio alone answers neither task quality nor speed.
<!-- bilingual-en:end -->

![[量化输出误差]]

![[量化加速条件]]

## 9. 先猜几个 token，再让目标模型验收
<!-- bilingual-en:start -->
*Propose several tokens, then let the target verify them*
<!-- bilingual-en:end -->

普通增量解码每一步要先等真实的前一个 token。投机方法尝试先给出一条候选，让目标模型并行计算这些已知候选位置。候选不等于事实，因此必须先理解如何修正概率，再理解哪些前缀可以保留。

<!-- bilingual-en:start -->
Ordinary incremental decoding waits for the actual previous token. Speculation supplies a candidate chain so the target can compute its known positions together. Candidates are not committed outputs, making probability correction and prefix validity essential.
<!-- bilingual-en:end -->

![[投机解码]]

![[投机采样修正]]

![[投机前缀验收]]

这里的“保持”有一个明确对象：目标采样器的概率分布。它不同于每次都得到同一段随机文本，也不同于不同数值实现逐 bit 一致。

<!-- bilingual-en:start -->
The preserved object is the target sampler's probability distribution, not identical random text in every run or bitwise equality between numerical implementations.
<!-- bilingual-en:end -->

![[投机分布保持边界]]

![[投机加速条件]]

## 10. 回到一次可验证的部署决策
<!-- bilingual-en:start -->
*Return to a verifiable deployment decision*
<!-- bilingual-en:end -->

到这里已经能提出具体假设：减少哪个对象的字节、改变哪种计算形状、缩短哪段等待，或提高一次验证交付的 token 数。最后一步是把假设放回同一任务和同一服务要求，而不是比较两张条件不同的跑分截图。

<!-- bilingual-en:start -->
You can now formulate specific hypotheses about bytes, computation shapes, waits, or tokens delivered per verification. Test them within the same task and service requirements rather than ranking incomparable benchmark screenshots.
<!-- bilingual-en:end -->

![[推理性能可比性]]

![[推理瓶颈诊断]]

稀疏专家模型沿用这里的测量方法，但路由、通信和专家负载还有额外成本，继续到 [[MoE 吞吐与延迟]]；模型选择覆盖训练和长期部署时，继续到 [[生命周期最优]]。这两处使用其他主题的同一份共享对象。

<!-- bilingual-en:start -->
Sparse expert models use the same measurement discipline but add routing, communication, and expert-load costs; continue to [[MoE 吞吐与延迟|MoE throughput and latency]]. Decisions covering training and long-term deployment continue to [[生命周期最优|lifecycle optimality]]. These are the same shared objects used in their respective themes.
<!-- bilingual-en:end -->

> [!question]- 用一段话完成这一课的自检
> “W4 模型文件只有原来的四分之一，所以单请求会快四倍；再加 PagedAttention 和 FlashAttention，长上下文的 KV 就不用保存；投机解码每轮最多输出四个 token，因此还能再快四倍。”这段判断分别混淆了什么？
>
> 答案要覆盖：权重负载与总显存/实际瓶颈，KV 状态与注意力中间矩阵，物理分配与有效数据量，以及最大候选交付数与实际接受率、草稿/验证成本。最后还需匹配质量、长度、负载和指标口径，才能判断真实收益。
>
> <!-- bilingual-en:start -->
> Explain why quarter-sized W4 files do not imply fourfold single-request speed, why paging and FlashAttention do not remove historical KV, and why a four-token speculative maximum does not imply fourfold speed. Distinguish storage objects, physical allocation, bottlenecks, acceptance, and verification cost, then state the matching quality and workload conditions.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

具体主张、公式、条件与手算依据在每个嵌入原子的末尾。主线使用 PagedAttention、Orca、Sarathi-Serve、FlashAttention 1/2、MQA/GQA、量化原论文和严格投机采样论文；客户端与服务端指标则明确采用所引工具的观测约定。本文的连接段负责解释这些对象怎样共同描述一次服务请求。

<!-- bilingual-en:start -->
Each embedded atom ends with sources for its claims, formulas, conditions, and calculations. Primary serving, attention, quantization, and speculative-sampling papers establish the mechanisms; metric conventions identify the cited tools' observation boundaries. The connecting prose explains how these objects describe a service request together.
<!-- bilingual-en:end -->

论文继续阅读从 [[06_paper/LLM/LLM Map Index|论文索引]] 进入原文。索引保留论文出处和主题关系，不替代原论文，也不把来源已核验当作学习者已经掌握。
