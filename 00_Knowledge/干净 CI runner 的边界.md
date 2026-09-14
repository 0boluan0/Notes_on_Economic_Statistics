---
aliases:
  - "干净 CI runner 能暴露未声明状态但不能证明构建已经密封"
  - Clean CI runner hermeticity boundary
  - 干净 runner 的验证边界
student_os: knowledge-atom
atom_id: CS-BUILD-017
atom_set: build-dependencies-ci
atom_type: guarantee-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[密封构建]]"
  - "[[CI 工作流结构]]"
related:
  - "[[本机与 CI 统一入口]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# 干净 CI runner 能暴露未声明状态但不能证明构建已经密封
<!-- bilingual-en:start -->
*A clean CI runner can expose undeclared state but cannot prove that a build is hermetic*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> GitHub-hosted job 从新的 runner image 启动，因此不会自然继承开发者上一次构建留下的工作目录与依赖缓存。若 CI 缺少生成物或包，这种新环境常能暴露“本机已有、规则未声明”的前置状态。
> <!-- bilingual-en:start -->
> A GitHub-hosted job starts from a fresh runner image, so it does not naturally inherit the developer's previous working directory or dependency cache. Missing generated files or packages can therefore expose prerequisites that existed locally but were never declared by the build.
> <!-- bilingual-en:end -->

然而 runner image 自带系统软件，workflow 也可能读取网络、时间、secrets 或浮动 action tag。一次干净运行通过，只说明这一个 runner 状态满足流程，不证明所有相关输入都已固定，也不证明不同环境会产生逐字节相同产物。
<!-- bilingual-en:start -->
However, a runner image includes system software, and a workflow may read the network, clock, secrets, or floating action tags. Passing once on a fresh runner shows only that this runner state satisfied the workflow; it proves neither that every relevant input is fixed nor that another environment will produce byte-identical artifacts.
<!-- bilingual-en:end -->

> [!question]- 自检
> CI 在 fresh runner 上通过一次，为什么不能直接给项目标记“hermetic and reproducible”？
>
> **答案：** fresh runner 仍有 image、网络、时间等环境输入，而且尚未通过独立重建与逐字节比较验证产物。

## 来源与核验

- [GitHub Actions Documentation, Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching)：核对 GitHub-hosted jobs 从 clean runner image 启动并需重新获取依赖。
- [Bazel Documentation, Hermeticity](https://bazel.build/basics/hermeticity)：核对宿主程序、时间戳与外部状态仍可成为 non-hermetic inputs。
- [Reproducible Builds, Definitions](https://reproducible-builds.org/docs/definition/)：核对可复现性需要指定产物的逐字节比较。
<!-- bilingual-en:start -->
- [GitHub Actions Documentation: Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching) was checked for GitHub-hosted jobs starting from clean runner images and reacquiring dependencies.
- [Bazel Documentation: Hermeticity](https://bazel.build/basics/hermeticity) was checked for host programs, timestamps, and external state as non-hermetic inputs.
- [Reproducible Builds: Definitions](https://reproducible-builds.org/docs/definition/) was checked for bit-for-bit comparison of specified artifacts.
<!-- bilingual-en:end -->
