---
aliases:
  - "Git merge 保留分叉而 rebase 重写提交父关系"
  - Git merge versus rebase
  - Git 合并与变基
student_os: knowledge-atom
atom_id: CS-GIT-008
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git commit 快照与父指针]]"
  - "[[Git 分支引用]]"
---

# Git merge 保留分叉而 rebase 重写提交父关系

<!-- bilingual-en:start -->
*Git merge preserves divergence while rebase rewrites commit parentage*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 两条历史已经分叉时，merge 通常创建一个同时指向两个父节点的新 commit；rebase 需要把提交重放到新基点时，则产生内容相似但 ID 与父关系不同的新 commit。
>
> <!-- bilingual-en:start -->
> When two histories have diverged, merge normally creates a new commit with both tips as parents. When rebase must replay commits onto another base, it produces commits with similar content but different IDs and parent relationships.
> <!-- bilingual-en:end -->

两种操作可以得到相同的最终文件快照，却留下不同历史。merge 明确保留“工作曾并行发生、后来汇合”；默认 rebase 会省略待重放历史中的 merge commit，把改动排列成新的线性祖先链。`--rebase-merges` 可以尝试重建分支结构和 merge commit，但重建出的 commit 仍是新对象，并不保留原 commit 身份。选择哪种不是单纯审美问题，而要看项目是否需要保留分叉信息、提交是否已经共享，以及后续维护约定。
<!-- bilingual-en:start -->
The two operations can produce the same final file snapshot while leaving different histories. Merge explicitly records that work happened in parallel and later converged. By default, rebase omits merge commits from the commits being replayed and arranges the changes into a new linear ancestry. `--rebase-merges` can try to recreate branching structure and merge commits, but the recreated commits are still new objects and do not retain the original commit identities. The choice is not merely aesthetic; it depends on whether divergence should remain visible, whether the commits have been shared, and the project's maintenance convention.
<!-- bilingual-en:end -->

若两个分支没有真正分叉，merge 可能只做 fast-forward：移动当前分支引用而不创建 merge commit。这并不推翻模型，只说明当前端点本来就是另一端点的祖先，不需要新节点表达汇合。
<!-- bilingual-en:start -->
If the histories have not truly diverged, merge may fast-forward by moving the current branch reference without creating a merge commit. This does not contradict the model; one tip was already an ancestor of the other, so no new node was needed to represent convergence.
<!-- bilingual-en:end -->

`git merge --squash` 是另一个明确例外：它把仿佛合并后的内容放入 working tree 与 index，却不移动 HEAD，也不记录 `MERGE_HEAD`，因此随后产生的是普通单父 commit，不会保存另一分支的祖先关系。不能只看命令名就断言历史分叉已被保留。
<!-- bilingual-en:start -->
`git merge --squash` is another explicit exception: it prepares the working tree and index as if a merge had occurred, but it neither moves HEAD nor records `MERGE_HEAD`. A later commit is therefore an ordinary single-parent commit and does not preserve the other branch's ancestry. The command name alone does not prove that divergence was recorded.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> merge 与 rebase 可能得到相同最终文件，为什么 commit ID 仍不同？
>
> **答案：** commit ID 取决于内容、父节点和元数据。rebase 改变父关系并创建新 commit；merge 通过多父 commit 保留原祖先图。

## 来源与核验

- [*Pro Git: Basic Branching and Merging*](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging.html)：核验 fast-forward、三方合并及多父 merge commit。
- [*Pro Git: Rebasing*](https://git-scm.com/book/en/v2/Git-Branching-Rebasing.html)：核验 rebase 重放改动、创建新 commit，以及与 merge 可能具有相同最终快照但不同历史。
- Git 官方文档，[`git-merge`](https://git-scm.com/docs/git-merge)：核验 true merge、fast-forward 与 `--squash` 对父关系的不同结果。
- Git 官方文档，[`git-rebase`](https://git-scm.com/docs/git-rebase)：核验默认线性化与 `--rebase-merges` 重建分支结构的边界。
