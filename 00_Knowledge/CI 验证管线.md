---
aliases:
  - "CI 验证管线应把依赖安装、构建、检查与产物交接写成显式步骤"
  - Explicit CI verification pipeline
  - CI 显式验证管线
student_os: knowledge-atom
atom_id: CS-BUILD-021
atom_set: build-dependencies-ci
atom_type: design-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[CI 工作流结构]]"
  - "[[CI 缓存与 artifact]]"
related:
  - "[[锁文件解析结果]]"
  - "[[本机与 CI 统一入口]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# CI 验证管线应把依赖安装、构建、检查与产物交接写成显式步骤
<!-- bilingual-en:start -->
*A CI verification pipeline should make dependency installation, building, checks, and artifact hand-offs explicit*
<!-- bilingual-en:end -->

> [!summary] 原子原则
> 一条可诊断的验证管线明确表示：依据 manifest，以及该生态使用时的 lockfile，取得依赖；运行必要的生成和编译；执行 lint、静态检查与测试；再把确实需要保留或交给下游 job 的输出上传为 artifact。每个工作单元都应能指出输入、命令、输出与失败位置。
> <!-- bilingual-en:start -->
> A diagnosable verification pipeline explicitly acquires dependencies from the manifest and, where the ecosystem uses one, the lockfile; performs required generation and compilation; runs linting, static checks, and tests; and uploads outputs that must survive or reach a downstream job as artifacts. Each work unit should expose its inputs, command, outputs, and point of failure.
> <!-- bilingual-en:end -->

依赖关系决定执行顺序：只有消费某个生成物的 job 才依赖生产它的 job；互不依赖的检查可以并行。若必要验证失败，依赖其结论的 package 或 release 工作不能被当作成功；独立的日志收集与诊断步骤仍可按明确条件运行。
<!-- bilingual-en:start -->
Dependencies determine execution order: only a job that consumes a generated output depends on the job that produced it, while independent checks may run in parallel. If required verification fails, packaging or release work that depends on that conclusion cannot be treated as successful; independent log collection and diagnostic steps may still run under explicit conditions.
<!-- bilingual-en:end -->

## 边界

“安装 → 构建 → 检查 → 打包”是常见骨架，不是所有项目必须照抄的线性清单。解释型项目可能没有 compile，多个测试矩阵可以并行，部署也可以属于独立 workflow。原则是让真实依赖与交接可见，而不是增加仪式步骤。
<!-- bilingual-en:start -->
“Install → build → check → package” is a common skeleton, not a mandatory linear checklist. An interpreted project may have no compile step, test matrices may run in parallel, and deployment may belong to a separate workflow. The principle is to expose real dependencies and hand-offs, not to add ceremonial stages.
<!-- bilingual-en:end -->

> [!question]- 自检
> 测试 job 读取生成的 schema，但 workflow 既没声明生成步骤，也没从上游下载 artifact。应把偶尔命中的 cache 当修复吗？
>
> **答案：** 不应。应声明 schema 的生成者和交接关系；cache 只能加速这一正式步骤。

## 来源与核验

- [GitHub Actions Documentation, Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)：核对 workflows、jobs、runners、steps 与 job 工作单元。
- [GitHub Actions Documentation, Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching)：核对 cache 与 artifact 的不同交接角色。
- [GitHub Actions Documentation, Continuous integration](https://docs.github.com/en/actions/get-started/continuous-integration?learn=continuous_integration)：核对 build、lint 与 tests 等 CI 检查；上面的管线形状是基于这些机制的设计原则，不是平台强制模板。
- [The Cargo Book, Cargo.toml vs Cargo.lock](https://doc.rust-lang.org/cargo/guide/cargo-toml-vs-cargo-lock.html)：以 Cargo 为例核对 manifest 与 lockfile 的不同职责；正文未把 lockfile 泛化为所有生态的必需文件。
<!-- bilingual-en:start -->
- [GitHub Actions Documentation: Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows) was checked for workflows, jobs, runners, steps, and job work units.
- [GitHub Actions Documentation: Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching) was checked for the distinct hand-off roles of caches and artifacts.
- [GitHub Actions Documentation: Continuous integration](https://docs.github.com/en/actions/get-started/continuous-integration?learn=continuous_integration) was checked for CI checks such as builds, linting, and tests. The pipeline shape above is a design principle based on those mechanisms, not a mandatory platform template.
- [The Cargo Book: Cargo.toml vs Cargo.lock](https://doc.rust-lang.org/cargo/guide/cargo-toml-vs-cargo-lock.html) was checked as one ecosystem's example of distinct manifest and lockfile roles; the text does not generalize lockfiles into a requirement for every ecosystem.
<!-- bilingual-en:end -->
