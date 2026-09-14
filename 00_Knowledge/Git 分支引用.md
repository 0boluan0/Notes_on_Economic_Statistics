---
aliases:
  - "Git 分支只是指向 commit 的可移动引用"
  - A Git branch is a movable reference
  - Git 分支指针
student_os: knowledge-atom
atom_id: CS-GIT-003
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git 对象图]]"
---

# Git 分支只是指向 commit 的可移动引用

<!-- bilingual-en:start -->
*A Git branch is only a movable reference to a commit*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> 分支不是另一份项目目录，也不是一组独有 commit。它是一个名字，当前值指向某个 commit；在该分支上产生新 commit 时，这个引用向前移动。
>
> <!-- bilingual-en:start -->
> A branch is neither another project directory nor an exclusive collection of commits. It is a name whose current value points to a commit. Creating a new commit on that branch advances the reference.
> <!-- bilingual-en:end -->

两个分支可以同时指向同一个 commit；创建分支只是多建一个引用，不会复制历史。分叉发生在它们后来移动到不同 commit 时。删除一个分支也只是删除引用：只要它指向的 commit 仍能从别的 branch、tag 或引用到达，对象仍属于可达历史。
<!-- bilingual-en:start -->
Two branches may point to the same commit. Creating a branch adds a reference rather than copying history; divergence occurs only when the references later advance to different commits. Deleting a branch removes a reference. Its commits remain reachable when another branch, tag, or reference still reaches them.
<!-- bilingual-en:end -->

这也解释了“当前分支包含哪些 commit”的真正含义：从分支指向的 commit 沿父边可达的所有 commit，而不是某个单独存放的列表。比较两个分支，实质是在比较两个引用所指向的历史端点及其可达图。
<!-- bilingual-en:start -->
This explains what it means for a branch to “contain” commits: they are the commits reachable by following parent edges from the branch tip, not members of a separately stored list. Comparing branches means comparing the history endpoints named by two references and their reachable graphs.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 在同一个 commit 上创建两个新分支，会不会复制 commit 或文件？
>
> **答案：** 不会，只会增加两个都指向该 commit 的引用；对象图仍是同一份。

## 来源与核验

- [*Pro Git: Branches in a Nutshell*](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)：核验 branch 是指向 commit 的轻量可移动指针，以及新提交使当前分支前移。
- Git 官方文档，[`gitdatamodel`](https://git-scm.com/docs/gitdatamodel)：核验 branch、tag 与 remote-tracking branch 都属于 references。
