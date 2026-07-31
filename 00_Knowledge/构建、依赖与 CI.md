---
aliases:
  - "Build Systems, Dependencies, and CI"
  - "构建与持续集成"
status: source-checked
---

# 构建、依赖与 CI
<!-- bilingual-en:start -->
*Build Systems, Dependencies, and Continuous Integration*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 从声明的源文件、工具链和依赖重复地产生同一类可验证产物。
> **具体锚点：** build graph 只重建输入已改变的目标；CI 则在干净环境中验证这个声明是否足够完整。
> **核心难点：** 本机缓存、未声明工具和浮动依赖会让“我这里能构建”无法复现。
> **为什么重要：** 可重复构建是协作、发布、回滚与供应链审计的共同基础。
> **继续：** 先画目标与依赖图，再锁定工具/依赖版本，最后让 CI 从空环境执行同一入口。
> <!-- bilingual-en:start -->
> **Problem addressed:** Repeatedly produce verifiable artifacts from declared sources, toolchains, and dependencies.
> **Concrete anchor:** A build graph rebuilds only targets whose inputs changed; CI checks in a clean environment whether the declaration is complete.
> **Central difficulty:** Local caches, undeclared tools, and floating dependencies make “it builds on my machine” irreproducible.
> **Why it matters:** Reproducible builds support collaboration, release, rollback, and supply-chain review.
> **Continue with:** Draw the target-dependency graph, pin tools and dependencies, and make CI invoke the same entry point from an empty environment.
> <!-- bilingual-en:end -->

## 构建图、依赖解析与验证流水线
<!-- bilingual-en:start -->
*Build Graphs, Dependency Resolution, and Verification Pipelines*
<!-- bilingual-en:end -->

构建图说明产物依赖，只有输入变化的节点重建。锁定依赖和运行时，CI 从干净环境执行 lint/test/build。缓存应是优化，删除缓存仍应能正确构建。
<!-- bilingual-en:start -->
A build graph describes artifact dependencies so that only nodes with changed inputs are rebuilt. Pin dependencies and runtimes, and have CI run lint, test, and build in a clean environment. A cache is an optimization; deleting it must not break correctness.
<!-- bilingual-en:end -->

构建系统回答“怎样从输入得到目标”，包管理器回答“依赖从哪里来及选哪个版本”，CI 回答“在受控环境中是否可重复通过”。把三者混为一个脚本会隐藏失败层。
<!-- bilingual-en:start -->
A build system answers how inputs produce targets, a package manager answers where dependencies come from and which versions are selected, and CI asks whether the process repeats in a controlled environment. Folding all three into an opaque script hides the failing layer.
<!-- bilingual-en:end -->

## Worked example：缓存不是输入
<!-- bilingual-en:start -->
*Worked Example: A Cache Is Not an Input*
<!-- bilingual-en:end -->

若 CI 只在缓存命中时通过，说明依赖或生成步骤未完整声明。验证方法是定期从空缓存构建；性能可以下降，但产物和测试结果不应改变。
<!-- bilingual-en:start -->
If CI passes only when its cache hits, a dependency or generation step is undeclared. Verify the workflow periodically with an empty cache; performance may decline, but artifacts and test results must not change.
<!-- bilingual-en:end -->

一个最小管线通常按顺序安装锁定依赖、生成/编译、运行静态检查与测试、打包产物。每阶段只消费显式产物，失败立即停止并保留日志。
<!-- bilingual-en:start -->
A minimal pipeline normally installs locked dependencies, generates or compiles, runs static checks and tests, and packages artifacts. Each stage consumes explicit outputs, stops on failure, and retains diagnostics.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 本机通过、CI 失败：比较运行时、锁文件、环境变量、工作目录和未跟踪生成物。
  <!-- bilingual-en:start -->
  Local build passes but CI fails: compare runtime versions, lockfiles, environment variables, working directory, and untracked generated files.
  <!-- bilingual-en:end -->
- 改一个文件却全部重建：检查依赖边是否过宽、时间戳生成物是否每次变化，以及目标是否声明真实输入。
  <!-- bilingual-en:start -->
  One file change rebuilds everything: inspect overly broad dependency edges, always-changing timestamped outputs, and whether targets declare their real inputs.
  <!-- bilingual-en:end -->
- 删除缓存后失败：把缓存中被错误依赖的内容变成正式输入或构建步骤，不要把缓存当持久存储。
  <!-- bilingual-en:start -->
  Clearing the cache breaks the build: turn the accidentally cached content into a declared input or build step rather than treating cache as durable storage.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### build graph 的节点和边分别表示什么？
<!-- bilingual-en:start -->
*What do nodes and edges represent in a build graph?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 节点是目标或中间产物，边表示一个目标依赖哪些输入，因此输入变化决定哪些节点必须重建。
> <!-- bilingual-en:start -->
> Nodes are targets or intermediate artifacts, and edges state which inputs a target depends on; changed inputs determine which nodes must be rebuilt.
> <!-- bilingual-en:end -->

### 为什么锁文件仍不能独自保证可复现？
<!-- bilingual-en:start -->
*Why can a lockfile not guarantee reproducibility by itself?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 工具链、操作系统、环境变量、外部下载和生成步骤也影响产物，必须一并声明或控制。
> <!-- bilingual-en:start -->
> Toolchains, operating systems, environment variables, external downloads, and generation steps also affect artifacts and must be declared or controlled.
> <!-- bilingual-en:end -->

### 如何判断缓存只是优化？
<!-- bilingual-en:start -->
*How can you verify that a cache is only an optimization?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 从空缓存运行仍得到正确产物并通过验证；差别只应是时间和资源消耗。
> <!-- bilingual-en:start -->
> A run from an empty cache still produces correct artifacts and passes validation; only time and resource use should differ.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/)：支持构建系统、依赖管理与自动化工作流的课程语境。
  <!-- bilingual-en:start -->
  The [official MIT Missing Semester course](https://missing.csail.mit.edu/2020/) supports the course context for build systems, dependency management, and automation.
  <!-- bilingual-en:end -->
- [CMake Buildsystem documentation](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html)：核验 target 与依赖图模型。
  <!-- bilingual-en:start -->
  The [CMake Buildsystem documentation](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html) verifies the target and dependency-graph model.
  <!-- bilingual-en:end -->
- [CMake Using Dependencies Guide](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html)：核验依赖发现与提供机制。
  <!-- bilingual-en:start -->
  The [CMake Using Dependencies Guide](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html) verifies dependency discovery and provision mechanisms.
  <!-- bilingual-en:end -->
