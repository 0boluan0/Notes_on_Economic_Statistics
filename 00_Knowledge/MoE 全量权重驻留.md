---
aliases:
  - "稀疏激活不等于部署时无需存放或访问全部专家权重"
  - Sparse activation does not remove the need to store or access all expert weights at deployment
  - MoE 权重存储边界
student_os: knowledge-atom
atom_id: LLM-MOE-014
atom_type: systems-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 稀疏激活不等于部署时无需存放或访问全部专家权重

<!-- bilingual-en:start -->
*Sparse activation does not remove the need to store or access all expert weights at deployment*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 一个 token 只执行少数 expert，并不让其余 expert 从模型中消失。为服务任意可能的路由，部署系统仍须让完整 expert 权重集驻留在某处或可按需访问。稀疏激活主要降低当次算术路径；checkpoint 大小和系统级权重容量仍由总参数决定。
>
> <!-- bilingual-en:start -->
> Executing only a few experts for one token does not make the others disappear. To serve any possible routing decision, the deployment system must keep the complete expert weight set resident somewhere or accessible on demand. Sparse activation mainly reduces the current arithmetic path; checkpoint size and system-level weight capacity still follow total parameters.
> <!-- bilingual-en:end -->

## 自然解释

常见 expert-parallel 部署把 expert 权重分片到多张设备：每张卡只保存一部分，所以单卡显存可能可控，但整个设备组仍共同保存全体 expert。router 选择远端 expert 时，通常发送 token activation 和返回结果，而不是临时搬整套 expert 权重。

<!-- bilingual-en:start -->
A common expert-parallel deployment shards expert weights across devices. Per-device memory can remain manageable, but the device group collectively stores all experts. When the router selects a remote expert, the system usually sends the token activation and returns its result rather than moving the full expert weights on demand.
<!-- bilingual-en:end -->

分片不等于删除：只要精确服务仍允许 router 选择全体 expert，本地没有的权重就必须在别处可访问，远端命中也会带来相应通信。蒸馏、剪枝，或把路由改成可抽取任务/句子子网，可以缩小实际部署集合，但它们改变了模型或路由契约。量化、压缩、缓存和 offload 同样是额外系统设计，不能从“MoE 稀疏”四个字自动推出。

<!-- bilingual-en:start -->
Sharding is not deletion. As long as exact serving permits the router to choose from every expert, weights absent locally must remain accessible elsewhere, and a remote hit entails the corresponding communication. Distillation, pruning, or routing schemes that expose a task- or sentence-level subnetwork can reduce the deployed set, but they change the model or routing contract. Quantisation, compression, caching, and offload are likewise additional systems designs, not automatic consequences of MoE sparsity.
<!-- bilingual-en:end -->

> [!warning] 边界
> “需要完整权重可访问”不等于“每台设备都复制完整权重”，也不等于“每个 token 搬全部权重”。区分系统总存储、单设备驻留和单 token 通信，才能正确描述成本。
>
> <!-- bilingual-en:start -->
> “The full weight set must be accessible” does not mean every device replicates it, nor that every token moves all weights. Separate system-wide storage, per-device residency, and per-token communication.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> MoE 每 token 只激活 2/64 个 expert，为什么 checkpoint 不会自动缩成原来的 1/32？
>
> **答案：** checkpoint 要保存 64 个 expert 的已学习参数；2/64 只描述一次输入执行哪些路径，不描述模型需要保存哪些路径。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§5.4–5.5：说明 unique expert weights 跨设备划分，以及 data/model/expert parallel 组合下的驻留与通信。
- Lepikhin et al. (2020), [*GShard*](https://arxiv.org/abs/2006.16668)，§3.1–3.3：给出 expert 权重分片与 token activation dispatch 的系统布局。
- Zoph et al. (2022), [*ST-MoE*](https://arxiv.org/abs/2202.08906)，§8：明确指出 sparse model 的完整参数存储会增加 serving 难度，并把蒸馏、剪枝或可抽取子网列为改变部署集合的额外办法。
