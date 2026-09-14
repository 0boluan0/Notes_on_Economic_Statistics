---
aliases:
  - "共享历史的撤销优先新增 revert 而不是移动公开引用"
  - Revert shared history instead of rewriting it
  - 共享提交的安全撤销
student_os: knowledge-atom
atom_id: CS-GIT-009
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git merge 与 rebase]]"
contrasts_with:
  - "[[Git reset 三种模式]]"
---

# 共享历史的撤销优先新增 revert 而不是移动公开引用

<!-- bilingual-en:start -->
*Prefer a new revert when undoing shared history instead of moving a published reference*
<!-- bilingual-en:end -->

> [!summary] 原子决策
> 错误 commit 已被推送并可能成为他人工作的祖先时，通常用 `git revert` 新增一个反向 commit。它改变当前内容却保留原 commit 与共同祖先关系，避免把协作者依赖的历史改写掉。
>
> <!-- bilingual-en:start -->
> When a faulty commit has been pushed and may already be an ancestor of other people's work, normally use `git revert` to add an inverse commit. It changes current content while preserving the original commit and shared ancestry, avoiding a rewrite of history on which collaborators depend.
> <!-- bilingual-en:end -->

revert 并不是让旧 commit 从历史消失。它读取目标 commit 引入的变化，尝试产生反向变化，默认把结果记录为新 commit；若使用 `--no-commit`，则先把结果留在 index 与工作区。若后续内容与原变化交织，仍可能冲突，需要人工判断。对 merge commit 做 revert 还必须指定保留哪一个父方向，并会影响以后再次合并该分支的含义。
<!-- bilingual-en:start -->
Revert does not make the old commit disappear. It derives an inverse of the selected commit's change and records the result as a new commit by default; with `--no-commit`, it first leaves the result in the index and working tree. If later work overlaps that change, conflicts may still require judgment. Reverting a merge also requires choosing the mainline parent and affects the meaning of later attempts to merge the same history.
<!-- bilingual-en:end -->

未共享的本地错误可以根据目标选择 amend、rebase 或 reset；“优先 revert”不是禁止本地整理，而是由共享边界触发的协作规则。判断前先问：这些 commit 是否已经被别人取得并作为祖先使用？
<!-- bilingual-en:start -->
For unshared local mistakes, amend, rebase, or reset may be appropriate depending on the desired state. “Prefer revert” is not a ban on local cleanup; it is a collaboration rule triggered by the sharing boundary. Ask first whether anyone else may already possess and build on these commits.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 已推送的错误提交为什么不能总用 `reset --hard` 后强推解决？
>
> **答案：** 那会移动公开引用并改写他人可能已依赖的祖先图；revert 追加可追踪反向提交，保留共同历史。

## 来源与核验

- Git 官方文档，[`git-revert`](https://git-scm.com/docs/git-revert)：核验 revert 通过新 commit 反转已有 commit 的变化。
- [*Pro Git: Advanced Merging — Undoing Merges*](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging.html)：核验共享仓库中 reset 重写历史的风险，以及 revert 保留历史的用法与 merge-revert 边界。
