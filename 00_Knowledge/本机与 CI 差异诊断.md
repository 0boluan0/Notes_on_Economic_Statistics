---
aliases:
  - "本机通过而 CI 失败时先逐项对齐提交、工具链与环境输入"
  - Local passes CI fails diagnostic
  - 本机 CI 差异诊断
student_os: knowledge-atom
atom_id: CS-BUILD-019
atom_set: build-dependencies-ci
atom_type: diagnostic-procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[锁文件与可复现构建]]"
  - "[[本机与 CI 统一入口]]"
  - "[[干净 CI runner 的边界]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# 本机通过而 CI 失败时先逐项对齐提交、工具链与环境输入
<!-- bilingual-en:start -->
*When a local build passes but CI fails, align the commit, toolchain, and environment inputs one dimension at a time*
<!-- bilingual-en:end -->

> [!summary] 原子诊断
> 先确认两边运行的是同一 commit/ref 与同一项目入口，再依次比较 runner OS/architecture、runtime 与 compiler 版本、manifest 和 lockfile、build flags 与环境变量、工作目录、untracked/generated files、外部服务和 cache 状态。每次只改变一个差异并重跑最小失败步骤。
> <!-- bilingual-en:start -->
> First confirm that both sides run the same commit or ref through the same project entry point. Then compare runner OS and architecture, runtime and compiler versions, manifest and lockfile, build flags and environment variables, working directory, untracked or generated files, external services, and cache state. Change one difference at a time and rerun the smallest failing step.
> <!-- bilingual-en:end -->

这个次序先排除“比较对象不同”，再查未声明环境。把本机已存在的 generated file 手工拷进 CI 只能遮住症状；若它是必要输入，就应由仓库提供或由声明过的步骤生成。
<!-- bilingual-en:start -->
This order first rules out comparing different revisions, then investigates undeclared environment. Manually copying a locally generated file into CI only masks the symptom. If it is required, the repository must supply it or a declared step must generate it.
<!-- bilingual-en:end -->

## 边界

本机/CI 差异常提示环境泄漏，但不是唯一根因；flaky test、资源限制或暂时外部服务故障也可能只在 CI 出现。需要保留失败日志和可重跑条件，而不是从“只在 CI 失败”直接下结论。
<!-- bilingual-en:start -->
A local/CI discrepancy often suggests environmental leakage, but it is not the only cause. Flaky tests, resource limits, or transient external-service failures may also appear only in CI. Preserve logs and reproducible conditions instead of inferring a cause merely from the location of the failure.
<!-- bilingual-en:end -->

> [!question]- 自检
> CI 报缺少生成文件，而本机仓库中该文件未跟踪但一直存在。长期修复是什么？
>
> **答案：** 声明并运行生成步骤，或把它作为正式输入纳入仓库；不要依赖本机遗留文件。

## 来源与核验

- [GitHub Actions Documentation, Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)：核对一次 run 绑定的 commit SHA/ref、runner 与 steps。
- [Bazel Documentation, Hermeticity](https://bazel.build/basics/hermeticity)：核对用不同机器、container 和 cache 行为发现宿主环境泄漏；正文的逐项对齐次序是基于这些可观测差异设计的诊断程序，不是 GitHub 强制流程。
- [GitHub Actions Documentation, Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching)：核对 clean runner 与 cache miss 的行为。
<!-- bilingual-en:start -->
- [GitHub Actions Documentation: Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows) was checked for the commit SHA/ref, runner, and steps associated with a run.
- [Bazel Documentation: Hermeticity](https://bazel.build/basics/hermeticity) was checked for using different machines, containers, and cache behaviour to expose host-environment leakage. The one-difference-at-a-time sequence in the text is a diagnostic procedure built from those observable differences, not a GitHub-mandated workflow.
- [GitHub Actions Documentation: Dependency caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/dependency-caching) was checked for clean-runner and cache-miss behaviour.
<!-- bilingual-en:end -->
