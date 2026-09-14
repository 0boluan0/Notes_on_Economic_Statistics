---
aliases:
  - "Git reset 的模式分别影响引用、暂存区与工作区"
  - Git reset modes
  - Git reset 三种主要模式
student_os: knowledge-atom
atom_id: CS-GIT-010
atom_type: state-transition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git 三棵状态]]"
contrasts_with:
  - "[[Git revert 与公开历史]]"
---

# Git reset 的模式分别影响引用、暂存区与工作区

<!-- bilingual-en:start -->
*Git reset modes affect the reference, index, and working tree differently*
<!-- bilingual-en:end -->

> [!summary] 原子状态转换
> 对 commit 执行 reset 时，`--soft` 只移动 HEAD 所指的分支；若 HEAD 已 detached，则移动 HEAD 自身。默认 `--mixed` 还把 index 改成目标 tree；`--hard` 再把工作区也覆盖成目标状态。越往后，潜在数据损失越大。
>
> <!-- bilingual-en:start -->
> When resetting to a commit, `--soft` moves the branch named by HEAD, or HEAD itself when it is detached. The default `--mixed` also makes the index match the target tree; `--hard` additionally overwrites the working tree. Each successive mode can discard more local state.
> <!-- bilingual-en:end -->

| 模式 | 移动 attached branch / detached HEAD | 重置 index | 覆盖 working tree |
|---|---:|---:|---:|
| `--soft` | 是 | 否 | 否 |
| `--mixed` | 是 | 是 | 否 |
| `--hard` | 是 | 是 | 是 |

这张表只覆盖三个主要模式。`--merge` 与 `--keep` 也会按文件状态选择性更新 index / working tree，并在会覆盖特定本地修改时中止；它们不是在上表末尾再增加两个机械步骤。
<!-- bilingual-en:start -->
The table covers only the three principal modes. `--merge` and `--keep` also update the index and working tree selectively and abort when particular local changes would be overwritten; they are not two further mechanical steps appended to the table.
<!-- bilingual-en:end -->

`git reset <path>` 是另一种形式：它更新指定路径在 index 中的版本，不移动 HEAD，也不改工作区，效果类似取消暂存。不能因为命令同叫 reset 就套用上表；先看调用是否给了 commit 模式还是 pathspec 模式。
<!-- bilingual-en:start -->
`git reset <path>` is a different form: it updates the selected paths in the index without moving HEAD or changing the working tree, and therefore behaves like unstaging. The table above must not be applied merely because the command is named reset. First determine whether the invocation is in commit mode or pathspec mode.
<!-- bilingual-en:end -->

执行前应把三层的当前内容和目标写出来，并用 `git status` / diff 检查。尤其是 `--hard`，它会覆盖跟踪文件的未提交修改，还可能覆盖路径中的未跟踪文件；不确定时先建可恢复引用或选择不破坏工作区的命令。
<!-- bilingual-en:start -->
Before running reset, write down the current and target states of all three layers and inspect them with status and diffs. `--hard` can overwrite uncommitted changes to tracked files and may overwrite untracked files in its way. When uncertain, create a recoverable reference or choose an operation that preserves the working tree.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 想移动本地分支到上一 commit，但保留 index 与工作区原样，应选哪个主要模式？
>
> **答案：** `git reset --soft HEAD^`；它移动分支端点但不改 index 或工作区。执行前仍应确认这些 commit 未共享。

## 来源与核验

- Git 官方文档，[`git-reset`](https://git-scm.com/docs/git-reset)：核验 commit 模式中 `--soft`、`--mixed`、`--hard` 对 HEAD、index 与 working tree 的不同影响，以及 pathspec 模式只更新 index。
