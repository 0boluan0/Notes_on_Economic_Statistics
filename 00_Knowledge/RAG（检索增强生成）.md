---
aliases:
  - "RAG"
  - "检索增强生成"
  - "Retrieval-Augmented Generation"
status: source-checked
---

# RAG（检索增强生成）
<!-- bilingual-en:start -->
*Retrieval-Augmented Generation (RAG)*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在模型参数之外提供任务所需资料：RAG 先检索相关片段，长上下文则把更多材料直接放入一次输入。
> **具体锚点：** 回答公司最新政策时，不该期待预训练记住；系统应找到当前政策原文，把相关段落和出处交给模型。
> **核心难点：** “资料在上下文里”不等于模型会找到、理解并忠实引用；检索召回、排序、切块、位置与生成约束都会丢失信息。
> **为什么重要：** 这决定知识更新、证据可追溯、成本和延迟，也决定错误来自没找到还是没用对。
> **继续：** 先画清检索—生成管线，再按资料规模和更新频率选择 RAG、长上下文或混合；效率见 [[LLM 推理效率]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Supply task-relevant evidence outside model parameters. RAG retrieves selected passages, while a long-context design directly places more material into one input.
> **Concrete anchor:** A system answering questions about the latest company policy should retrieve the current source document and pass supported passages and provenance to the model rather than expecting pretraining to remember it.
> **Central difficulty:** Evidence appearing in context does not mean the model will find, understand, and cite it faithfully. Recall, ranking, chunking, placement, and generation constraints can each lose information.
> **Why it matters:** The design controls knowledge freshness, traceability, cost, and latency and reveals whether an error came from failing to retrieve evidence or failing to use it.
> **Continue with:** Trace the retrieval-to-generation pipeline below, compare it with [[长上下文语言模型|Long-Context Language Models]], and use [[LLM 推理效率|LLM Inference Efficiency]] for serving cost.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
> - The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
> <!-- bilingual-en:end -->

## 检索—生成管线
<!-- bilingual-en:start -->
*Retrieval-to-Generation Pipeline*
<!-- bilingual-en:end -->

典型流程是摄取与清洗 → 切块/索引 → 查询改写 → 召回 → rerank → 组装上下文 → 生成与引用。每一步都要保留来源身份；否则答案即使正确，也无法知道证据是哪一版。
<!-- bilingual-en:start -->
A typical pipeline is ingestion and cleaning → chunking and indexing → query rewriting → retrieval → reranking → context assembly → generation and citation. Source identity must survive every stage; otherwise even a correct answer cannot reveal which document version supported it.
<!-- bilingual-en:end -->

把每个阶段写成可检查的输入—输出：索引记录文档 id、版本、权限与位置；检索返回分数和候选；reranker 改变顺序但不丢来源；组装器记录实际送入模型的片段；回答的每项主张再指向支持片段。这样才能定位故障而不是笼统“调 RAG”。
<!-- bilingual-en:start -->
Treat each stage as an inspectable input–output boundary. The index records document identity, version, permissions, and location; retrieval returns scores and candidates; reranking changes order without losing provenance; the assembler records passages actually sent to the model; and each answer claim points back to supporting passages. This makes failures localizable instead of inviting generic “RAG tuning.”
<!-- bilingual-en:end -->

## 检索方法
<!-- bilingual-en:start -->
*Retrieval Methods*
<!-- bilingual-en:end -->

稀疏检索善于精确词匹配，稠密检索用向量语义召回，hybrid 结合二者。ColBERT 类迟交互保留 token 级匹配但成本更高。先测 recall@k，若正确文档未召回，调生成模型通常无效。
<!-- bilingual-en:start -->
Sparse retrieval excels at exact lexical matching, dense retrieval uses vector similarity for semantic recall, and hybrid retrieval combines both. Late-interaction methods such as ColBERT preserve token-level matching at higher cost. Measure recall at k first; tuning the generator rarely helps when the correct document is never retrieved.
<!-- bilingual-en:end -->

query rewriting 可补全缩写、时间或实体，却也可能改变用户意图；filter 可按权限、版本和日期缩小空间，却可能误排除证据。保留原始 query、改写 query、filter 和候选列表，才能重放一次检索。
<!-- bilingual-en:start -->
Query rewriting can expand abbreviations, times, or entities but may alter user intent. Filters can narrow by permission, version, and date but may wrongly exclude evidence. Preserve the original query, rewritten query, filters, and candidate list so a retrieval can be replayed.
<!-- bilingual-en:end -->

## Chunking、排序与引用
<!-- bilingual-en:start -->
*Chunking, Ranking, and Citation*
<!-- bilingual-en:end -->

块太小会失去上下文，太大则稀释证据并占窗口。按文档结构切、保留标题/页码/时间和重叠，再用 reranker 选择。引用必须能回到实际支持该句的段落，不能只列一个看似相关的文档。
<!-- bilingual-en:start -->
Chunks that are too small lose context, while oversized chunks dilute evidence and consume the window. Chunk along document structure, preserve title, page, time, and overlap, then use a reranker. A citation must return to the passage that actually supports the claim rather than merely naming a related-looking document.
<!-- bilingual-en:end -->

切块单位应随证据单位变化：定义和条款可按标题/段落，表格要保留表头和行列语义，代码要保留函数边界，跨页证明需要父子 chunk 或邻接扩展。固定字符数只是一条 baseline。
<!-- bilingual-en:start -->
Chunk units should follow evidence units: definitions and policies can follow headings and paragraphs, tables must retain headers and row–column semantics, code should preserve function boundaries, and cross-page proofs may need parent–child chunks or neighbor expansion. A fixed character count is only a baseline.
<!-- bilingual-en:end -->

reranker 的目标是把已经召回的候选重新排序，不能找回召回阶段从未出现的文档。引用核验还要做 entailment：片段是否真正支持该主张，是否过期，是否与其他片段冲突。
<!-- bilingual-en:start -->
A reranker reorders retrieved candidates; it cannot recover a document absent from the candidate set. Citation verification also requires entailment checks: whether the passage actually supports the claim, whether it is current, and whether it conflicts with other evidence.
<!-- bilingual-en:end -->

## Worked example：一条过期政策答案怎样定位
<!-- bilingual-en:start -->
*Worked Example: Diagnose an Outdated Policy Answer*
<!-- bilingual-en:end -->

用户问“目前报销上限是多少”，系统回答旧数值。先检查当前政策是否已进入索引；若没有，是 ingestion failure。若在索引但不在 top-k，是 retrieval failure；若在候选却被旧版排前，是 ranking/version failure；若新版已进上下文但仍输出旧值，是 evidence-use failure。
<!-- bilingual-en:start -->
A user asks for the current reimbursement limit, and the system answers with an old value. First check whether the current policy entered the index; if not, ingestion failed. If indexed but absent from top k, retrieval failed. If retrieved but ranked below an old version, ranking or version control failed. If the current passage reached context but the model still gave the old value, evidence use failed.
<!-- bilingual-en:end -->

若答案列出新版文档却引述旧数值，citation correctness 也失败：引用存在不等于引用支持。修复可要求每个数值主张附 passage id，并在生成后验证答案 span 能否由该 passage 推出。
<!-- bilingual-en:start -->
If the answer names the current document but quotes the old value, citation correctness also fails: having a citation does not mean it supports the claim. A repair can require a passage identifier for every numerical claim and verify afterward that the answer span follows from that passage.
<!-- bilingual-en:end -->

## 怎么选
<!-- bilingual-en:start -->
*How to Choose*
<!-- bilingual-en:end -->

频繁更新、需要出处、语料大时优先 RAG；少量材料需跨全文综合时长上下文方便；高价值任务常用检索缩小范围后给较长上下文。若知识可稳定结构化，直接查询数据库/API 可能比两者都可靠。
<!-- bilingual-en:start -->
Prefer RAG for frequently updated, citation-sensitive, large corpora. Long context is convenient for synthesizing a small number of full documents. High-value systems often retrieve to narrow the set and then provide a longer context. If knowledge is stably structured, a direct database or API query can be more reliable than either.
<!-- bilingual-en:end -->

## 诊断顺序
<!-- bilingual-en:start -->
*Diagnostic Order*
<!-- bilingual-en:end -->

把错误分成：索引缺失、召回失败、排序失败、上下文组装错误、证据冲突、生成未遵循证据。分别记录检索指标和有证据条件下的生成正确率，避免只看最终答案。
<!-- bilingual-en:start -->
Classify errors as missing index entries, retrieval failures, ranking failures, context-assembly errors, evidence conflicts, or generation that ignores evidence. Record retrieval metrics separately from generation accuracy conditioned on available evidence instead of observing only the final answer.
<!-- bilingual-en:end -->

- recall@k 低：先检查 ingestion、parser、chunk、embedding/query 与 filters，不要改回答 prompt。
  <!-- bilingual-en:start -->
  Recall at k is low: inspect ingestion, parsing, chunks, embeddings or queries, and filters before changing the answer prompt.
  <!-- bilingual-en:end -->
- recall 高但答案错：查看 rerank、实际上下文、冲突版本、模型是否引用证据和上下文中的指令注入。
  <!-- bilingual-en:start -->
  Recall is high but answers are wrong: inspect reranking, actual context, conflicting versions, evidence adherence, and prompt injection inside retrieved text.
  <!-- bilingual-en:end -->
- 引用很多但不可核验：测 citation precision/entailment，而不是以引用数量作为可信度。
  <!-- bilingual-en:start -->
  Many citations cannot be verified: measure citation precision and entailment rather than using citation count as a proxy for trustworthiness.
  <!-- bilingual-en:end -->
- 权限资料泄露：访问控制必须在 retrieval 层执行并记录，不能指望生成 prompt 隐藏已经送入上下文的信息。
  <!-- bilingual-en:start -->
  Restricted material leaks: enforce and log access control at retrieval time; a generation prompt cannot reliably hide information already placed in context.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 正确文档从未进入 top-k 时，优先修哪一层？
<!-- bilingual-en:start -->
*Which layer should be fixed first when the correct document never enters top k?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 检索/索引层；生成器没有看到证据，换提示通常不能根治。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Fix retrieval or indexing. The generator never saw the evidence, so prompt changes usually cannot address the root cause.
<!-- bilingual-en:end -->

### 政策资料既更新快又要求可引用，应优先哪种设计？
<!-- bilingual-en:start -->
*Which design should be preferred for rapidly changing policy material that must be cited?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 带版本与来源元数据的 RAG，必要时再把召回文档以较长上下文综合。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Use RAG with version and provenance metadata, then provide the retrieved documents in a longer context for synthesis when needed.
<!-- bilingual-en:end -->

### 正确 passage 已召回但答案仍错，下一步看什么？
<!-- bilingual-en:start -->
*What should be inspected next when the correct passage was retrieved but the answer remains wrong?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 看 rerank 后位置、实际组装上下文、版本冲突、生成指令与答案是否由 passage 支持，把排序与证据使用分开测。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Inspect rank after reranking, the assembled context, version conflicts, generation instructions, and whether the passage entails the answer; measure ranking and evidence use separately.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
  <!-- bilingual-en:start -->
  [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
  <!-- bilingual-en:end -->
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
  <!-- bilingual-en:start -->
  The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
  <!-- bilingual-en:end -->
- [Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)：核验 retriever—generator 结合、外部非参数记忆与来源检索的基本框架。
  <!-- bilingual-en:start -->
  [Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) verifies the basic retriever–generator framework, external nonparametric memory, and source retrieval.
  <!-- bilingual-en:end -->
- [Karpukhin et al. (2020), Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)：核验双编码稠密 passage retrieval 与 top-k 召回评测。
  <!-- bilingual-en:start -->
  [Karpukhin et al. (2020), Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) verifies dual-encoder dense passage retrieval and top-k recall evaluation.
  <!-- bilingual-en:end -->

