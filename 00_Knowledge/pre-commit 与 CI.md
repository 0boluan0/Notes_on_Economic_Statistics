---
aliases:
  - "本地 pre-commit hook 提前拦截失败但不能替代共享 CI"
  - Local pre-commit hook versus shared CI
  - 本地 hook 与共享 CI
student_os: knowledge-atom
atom_id: CS-BUILD-022
atom_set: build-dependencies-ci
atom_type: guarantee-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[CI 工作流结构]]"
related:
  - "[[本机与 CI 统一入口]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# 本地 pre-commit hook 提前拦截失败但不能替代共享 CI
<!-- bilingual-en:start -->
*A local pre-commit hook catches failures early but cannot replace shared CI*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Git 的 `pre-commit` hook 在创建 commit 之前运行；脚本返回非零状态会中止这次 commit。因此它可以调用与 CI 相同的 `make check` 等入口，让开发者在 push 以前得到快速反馈。
> <!-- bilingual-en:start -->
> Git's `pre-commit` hook runs before a commit is created, and a nonzero exit aborts that commit. It can therefore invoke the same entry point as CI, such as `make check`, and give the developer fast feedback before a push.
> <!-- bilingual-en:end -->

但 client-side hook 默认位于本地 Git directory 的 hooks 路径，不会随 repository clone 自动复制，而且 `git commit --no-verify` 可以绕过 `pre-commit`。它是个人反馈层，不是团队已执行检查的共同证据；共享 CI 仍应对 push 或 pull request 对应的 commit 运行必要验证。
<!-- bilingual-en:start -->
However, a client-side hook normally lives in the hooks path of the local Git directory, is not copied automatically when a repository is cloned, and can be bypassed for `pre-commit` with `git commit --no-verify`. It is a personal feedback layer rather than shared evidence that checks ran; shared CI should still verify the commit associated with a push or pull request.
<!-- bilingual-en:end -->

## 边界

remote/server-side hook 可以在接收 push 时执行并拒绝更新，保证边界不同于 client-side hook；托管平台的 CI 与 branch protection 又有自己的配置。不能把“我装了本地 hook”写成“仓库政策已被所有协作者强制执行”。
<!-- bilingual-en:start -->
A remote or server-side hook can run while receiving a push and reject an update, giving it a different enforcement boundary from a client-side hook. Hosted CI and branch protection have their own configuration. “I installed a local hook” must not be reported as “repository policy is enforced for every collaborator.”
<!-- bilingual-en:end -->

> [!question]- 自检
> 你的 pre-commit hook 通过了，能否由此断言同事 clone 后的每次 commit 都执行了同一检查？
>
> **答案：** 不能。client-side hooks 不随 clone 自动复制且可以绕过；需要共享 CI 或明确的 remote enforcement 提供共同边界。

## 来源与核验

- [Git Documentation, githooks](https://git-scm.com/docs/githooks)：核对 hook 路径、可执行要求、`pre-commit` 的调用时机、非零退出与 `--no-verify`。
- [Pro Git, Git Hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)：核对 client-side hooks 不随 clone 自动复制及其不适合单独强制团队政策的边界。
- [GitHub Documentation, About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)：核对 branch protection 可把指定 status checks 设为合并前要求，因此共享执行与仓库强制边界取决于独立配置。
- [MIT Missing Semester, Metaprogramming exercises](https://missing.csail.mit.edu/2020/metaprogramming/#exercises)：核对用 pre-commit 调用 `make paper.pdf` 并拒绝不可构建 commit 的课程练习。
<!-- bilingual-en:start -->
- [Git Documentation: githooks](https://git-scm.com/docs/githooks) was checked for hook location, executability, `pre-commit` timing, nonzero exit, and `--no-verify`.
- [Pro Git: Git Hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) was checked for client-side hooks not being copied on clone and not by themselves enforcing team policy.
- [GitHub Documentation: About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) was checked for branch protection requiring selected status checks before merge, so shared execution and repository enforcement remain separately configured boundaries.
- [MIT Missing Semester: Metaprogramming exercises](https://missing.csail.mit.edu/2020/metaprogramming/#exercises) was checked for the exercise that calls `make paper.pdf` from pre-commit and rejects an unbuildable commit.
<!-- bilingual-en:end -->
