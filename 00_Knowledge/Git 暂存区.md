---
aliases:
  - "Git 暂存区定义下一次提交而不是已保存历史"
  - The Git index defines the next commit
  - Git 暂存区语义
student_os: knowledge-atom
atom_id: CS-GIT-006
atom_type: state
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git 三棵状态]]"
---

# Git 暂存区定义下一次提交而不是已保存历史

<!-- bilingual-en:start -->
*The Git index defines the next commit; it is not saved history*
<!-- bilingual-en:end -->

> [!summary] 原子状态
> `git add` 把所选路径或片段的当前内容写入 index，形成下一次 `git commit` 将采用的候选快照。暂存不是提交：只保存在 index 的选择仍可被改写，也不会成为共享历史。
>
> <!-- bilingual-en:start -->
> `git add` writes the selected paths or hunks into the index, forming the candidate snapshot used by the next `git commit`. Staging is not committing: an index-only selection can still be replaced and is not shared history.
> <!-- bilingual-en:end -->

暂存区使同一工作区中的修改可以按逻辑拆成不同 commit。`git add -p` 可以只选一个文件中的某些片段；提交后，工作区未暂存的其余修改仍留在原处。正确问题不是“这个文件有没有被 add”，而是“index 中这个路径的内容到底是什么”。
<!-- bilingual-en:start -->
The index lets changes in one working tree be divided into coherent commits. `git add -p` may select only some hunks from a file; after committing them, the remaining unstaged edits stay in the working tree. The precise question is not merely whether the file was added, but what content the index currently holds for that path.
<!-- bilingual-en:end -->

未解决的三方合并是边界情况：同一路径可以在 index 中同时保留 stage 1（base）、2（ours）和 3（theirs）。这时 index 提供重建结果的输入，还不是一个可直接提交的单一候选版本；解决并 `git add` 后才回到普通的 stage 0 条目。
<!-- bilingual-en:start -->
An unresolved three-way merge is the boundary case: the index may hold stage 1 (base), 2 (ours), and 3 (theirs) entries for the same path. The index then supplies inputs for reconstructing the result rather than one directly committable candidate version; resolving and staging the path returns it to a normal stage-0 entry.
<!-- bilingual-en:end -->

提交前的最小检查因此是：用 `git status` 确认层次，再用 `git diff --cached` 审阅将被提交的内容。普通 `git diff` 只显示工作区相对 index 的差异，可能为空，却不表示没有暂存内容。
<!-- bilingual-en:start -->
A minimal pre-commit check is therefore to inspect the layers with `git status` and review the actual candidate snapshot with `git diff --cached`. Plain `git diff` shows the working tree relative to the index. It may be empty even when staged content exists.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> `git diff` 没有输出，能否推出下一次提交为空？
>
> **答案：** 不能。它只说明工作区与 index 相同；应使用 `git diff --cached` 比较 index 与 HEAD。

## 来源与核验

- Git 官方文档，[`git-diff`](https://git-scm.com/docs/git-diff)：核验普通 diff 与 `--cached` 的比较端点，以及 index 是下一次提交的 staging area。
- [*Pro Git: Undoing Things*](https://git-scm.com/book/en/v2/Git-Basics-Undoing-Things.html)：核验 commit 使用暂存区形成新快照。
