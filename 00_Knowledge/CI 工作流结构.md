---
aliases:
  - "CI 工作流把触发事件映射为 runner 上的 jobs 与 steps"
  - CI workflow event job step model
  - CI 工作流层级
student_os: knowledge-atom
atom_id: CS-BUILD-014
atom_set: build-dependencies-ci
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# CI 工作流把触发事件映射为 runner 上的 jobs 与 steps
<!-- bilingual-en:start -->
*A CI workflow maps trigger events to jobs and steps executed on runners*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> 以 GitHub Actions 为例，workflow 是仓库中的 YAML 自动化定义；event、手动调用或 schedule 触发一次 run。一个 workflow 含一个或多个 jobs，每个 job 在一台 runner 上执行一串 steps；step 可以运行脚本，也可以调用可复用 action。
> <!-- bilingual-en:start -->
> In GitHub Actions, a workflow is an automation definition stored as YAML in the repository. An event, manual invocation, or schedule triggers a run. A workflow contains one or more jobs, each job executes a sequence of steps on a runner, and a step either runs a script or invokes a reusable action.
> <!-- bilingual-en:end -->

四层回答不同问题：trigger 决定**何时**运行，job 决定**在哪个 runner 和相互依赖关系中**运行，step 决定**做什么动作**，日志与结论属于这一次具体 run。把它们区分开，才能判断“根本没触发”“job 没排到”“step 失败”或“检查通过”。
<!-- bilingual-en:start -->
The layers answer different questions: the trigger decides **when** to run, a job decides **on which runner and within what job dependency structure**, a step decides **what action to perform**, and logs plus conclusions belong to one concrete run. This separates “never triggered,” “job not scheduled,” “step failed,” and “checks passed.”
<!-- bilingual-en:end -->

## 边界

这些名称和 YAML 位置是 GitHub Actions 的实现模型，不是 CI 的普遍语法。其他服务可能叫 pipeline、stage、task 或 agent，但仍应明确触发、执行环境、工作单元与结果。
<!-- bilingual-en:start -->
These names and YAML locations belong to GitHub Actions rather than universal CI syntax. Other services may use pipeline, stage, task, or agent, but triggers, execution environments, work units, and results still need to be distinguished.
<!-- bilingual-en:end -->

> [!question]- 自检
> workflow 文件存在但 push 后完全没有 run，应该先查测试命令还是 `on:` 条件？
>
> **答案：** 先查 trigger，即 `on:` 是否匹配这次事件；测试命令只有 run 被触发后才会进入 step。

## 来源与核验

- [GitHub Actions Documentation, Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)：核对 workflow 文件、triggers、jobs、runners 与 steps。
- [MIT Missing Semester, Continuous integration systems](https://missing.csail.mit.edu/2020/metaprogramming/#continuous-integration-systems)：核对仓库事件触发受控机器执行命令并记录结果的课程模型。
<!-- bilingual-en:start -->
- [GitHub Actions Documentation: Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows) was checked for workflow files, triggers, jobs, runners, and steps.
- [MIT Missing Semester: Continuous integration systems](https://missing.csail.mit.edu/2020/metaprogramming/#continuous-integration-systems) was checked for the course model of repository events triggering commands on controlled machines and recording results.
<!-- bilingual-en:end -->
