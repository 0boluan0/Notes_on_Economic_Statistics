---
aliases:
  - "CI 通过自动构建和测试集成频繁变更而部署发布属于另一层"
  - Continuous integration versus deployment
  - CI 与部署发布边界
student_os: knowledge-atom
atom_id: CS-BUILD-015
atom_set: build-dependencies-ci
atom_type: systems-boundary
status: source-checked
mastery_state: unassessed
related:
  - "[[CI 工作流结构]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# CI 通过自动构建和测试集成频繁变更而部署发布属于另一层
<!-- bilingual-en:start -->
*CI integrates frequent changes through automated builds and tests, while deployment and release form another layer*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> continuous integration（持续集成）的核心是频繁把变更提交到共享仓库，并持续构建、测试或执行其他检查，尽早发现某次变更引入的问题。绿色 CI 只说明本次 run 中实际执行并计入结论的检查通过，不自动说明软件已经部署、发布或可安全投产。
> <!-- bilingual-en:start -->
> Continuous integration centres on frequent changes to a shared repository and continual builds, tests, or other checks that detect introduced problems early. Green CI means the checks actually executed and counted in that run's conclusion passed; it does not automatically mean the software was deployed, released, or proven safe for production.
> <!-- bilingual-en:end -->

同一个 GitHub Actions 平台也能执行 package、release 或 deploy workflow，但那是平台覆盖的更完整软件生命周期，不应倒推为所有 CI run 的固有后果。是否允许部署还可能取决于 approvals、environment rules 与额外验证。
<!-- bilingual-en:start -->
The same GitHub Actions platform can also run package, release, or deployment workflows, but those belong to the broader software lifecycle supported by the platform and are not inherent consequences of every CI run. Deployment may additionally depend on approvals, environment rules, and further verification.
<!-- bilingual-en:end -->

> [!question]- 自检
> pull request 的 lint 与 tests 全绿，能否据此填写“production deployment complete”？
>
> **答案：** 不能。它只证明已配置的 CI 检查通过；部署需要独立的执行与可验证结果。

## 来源与核验

- [GitHub Actions Documentation, Continuous integration](https://docs.github.com/en/actions/get-started/continuous-integration?learn=continuous_integration)：核对频繁提交、build/test 与 Actions 还能另做 deploy/package/release 的边界。
- [GitHub Actions Documentation, Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)：核对同一仓库可有承担不同任务的多个 workflows。
<!-- bilingual-en:start -->
- [GitHub Actions Documentation: Continuous integration](https://docs.github.com/en/actions/get-started/continuous-integration?learn=continuous_integration) was checked for frequent commits, build/test, and the separate use of Actions for deploy/package/release work.
- [GitHub Actions Documentation: Workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows) was checked for multiple workflows with different responsibilities in one repository.
<!-- bilingual-en:end -->
