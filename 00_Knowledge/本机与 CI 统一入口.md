---
aliases:
  - "本机与 CI 应调用同一个项目级构建验证入口"
  - Shared local and CI build entry point
  - 本机 CI 同入口原则
student_os: knowledge-atom
atom_id: CS-BUILD-016
atom_set: build-dependencies-ci
atom_type: design-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[构建规则三要素]]"
  - "[[CI 工作流结构]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# 本机与 CI 应调用同一个项目级构建验证入口
<!-- bilingual-en:start -->
*Local development and CI should invoke the same project-level build and verification entry point*
<!-- bilingual-en:end -->

> [!summary] 原子原则
> 把生成、编译、静态检查和测试的真实依赖与项目级编排放在项目自己的构建入口中，例如 `make check` 或 package script；本机与 CI 都调用它。CI 配置主要负责触发、runner、凭证与结果上报，而不是复制一份逐渐漂移的项目逻辑。
> <!-- bilingual-en:start -->
> Put the real dependencies and project-level orchestration of generation, compilation, static checks, and tests behind a project-owned entry point such as `make check` or a package script, and invoke it both locally and in CI. CI configuration should mainly supply triggers, runners, credentials, and result reporting rather than duplicate project logic that can drift.
> <!-- bilingual-en:end -->

这样，“本机同一命令通过、CI 失败”就成为有信息量的环境差异；若本机脚本和 CI YAML 各自编排另一套步骤，失败也可能来自两份流程本身不一致。
<!-- bilingual-en:start -->
Then “the same command passes locally but fails in CI” becomes an informative environmental difference. If a local script and CI YAML independently orchestrate different steps, failure may instead come from divergence between the two procedures.
<!-- bilingual-en:end -->

## 边界

同入口不要求环境完全相同。CI 仍可运行 OS/version matrix、注入只在 CI 可用的 secrets，或在验证后执行独立发布 job；共同的是项目语义入口，不是每个外围配置字节。
<!-- bilingual-en:start -->
A shared entry point does not require identical environments. CI may still run an OS/version matrix, inject CI-only secrets, or perform a separate release job after verification. What is shared is the project-semantic entry point, not every byte of surrounding configuration.
<!-- bilingual-en:end -->

> [!question]- 自检
> 本机运行 `make check`，CI 却手写另一套编译和测试命令。最直接的结构性风险是什么？
>
> **答案：** 两套编排会漂移，使“本机通过、CI 失败”混入流程差异，难以只定位环境差异。

## 来源与核验

- [MIT Missing Semester, Metaprogramming exercises](https://missing.csail.mit.edu/2020/metaprogramming/#exercises)：课程用同一个 `make paper.pdf` 入口连接本机构建与 pre-commit 自动检查，并要求在 GitHub Actions 中运行项目检查。
- [GitHub Actions Documentation, Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)：核对 step 可运行项目脚本或 reusable action；“共享项目入口”是基于这些机制的工程设计原则，而非 GitHub 的强制语法。
<!-- bilingual-en:start -->
- [MIT Missing Semester: Metaprogramming exercises](https://missing.csail.mit.edu/2020/metaprogramming/#exercises) connects local building and pre-commit automation through the same `make paper.pdf` entry point and asks learners to run project checks in GitHub Actions.
- [GitHub Actions Documentation: Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows) was checked for steps running project scripts or reusable actions. The shared-entry recommendation is an engineering design principle built on those mechanisms, not mandatory GitHub syntax.
<!-- bilingual-en:end -->
