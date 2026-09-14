---
aliases:
  - "Git 工作区、暂存区与 HEAD 是三个不同状态"
  - Git working tree index and HEAD
  - Git 三棵树
student_os: knowledge-atom
atom_id: CS-GIT-005
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git HEAD 与分离状态]]"
---

# Git 工作区、暂存区与 HEAD 是三个不同状态

<!-- bilingual-en:start -->
*The Git working tree, index, and HEAD are three different states*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> 工作区是磁盘上正在编辑的文件；index（暂存区）保存下一次提交准备采用的版本；HEAD 表示当前检出历史的 commit。一个路径可以在这三层同时具有三个不同内容。
>
> <!-- bilingual-en:start -->
> The working tree contains the files being edited on disk. The index stores the versions prepared for the next commit. HEAD denotes the commit currently checked out. The same path may have three different contents across these layers at once.
> <!-- bilingual-en:end -->

这里把 HEAD 称作一“层”，实际是指 HEAD 所解析到的 commit tree；HEAD 本身只是引用，不保存文件。若 HEAD 指向尚无首次 commit 的 unborn branch，这一已提交端点还不存在。
<!-- bilingual-en:start -->
Calling HEAD a “layer” here means the tree of the commit to which HEAD resolves; HEAD itself is only a reference and stores no files. If HEAD names an unborn branch with no first commit, this committed endpoint does not yet exist.
<!-- bilingual-en:end -->

假设 HEAD 中 `report.md` 是 A。你把它改成 B 并执行 `git add`，index 记录 B；随后继续把工作区改成 C。此时：

- HEAD：A，上一份已提交状态；
- index：B，若立即提交将进入新快照的状态；
- working tree：C，仍有一部分尚未暂存的编辑。

<!-- bilingual-en:start -->
Suppose `report.md` is A in HEAD. You edit it to B and run `git add`, so the index records B; then you continue editing the working tree to C. HEAD remains A, the index contains B for the next snapshot, and the working tree contains the still-unstaged C.
<!-- bilingual-en:end -->

这套模型比“文件已修改/未修改”多一层，却能解释 `status`、`diff`、`add`、`restore`、`commit` 和不同 `reset` 模式为何看似做着相似但并不相同的事。执行会覆盖内容的命令前，先写清它读取哪一层、写入哪一层。
<!-- bilingual-en:start -->
This model is one layer richer than a simple modified/unmodified flag, but it explains why `status`, `diff`, `add`, `restore`, `commit`, and reset modes perform related yet distinct operations. Before running a content-overwriting command, state which layer it reads and which layer it writes.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 文件先修改并 `git add`，随后又修改一次。为什么它会同时出现在 staged 和 unstaged changes 中？
>
> **答案：** index 保存第一次修改后的版本，工作区保存第二次修改后的版本；两层都分别不同于自己的比较端点。

## 来源与核验

- Git 官方文档，[`git`](https://git-scm.com/docs/git) 的 Discussion：核验对象数据库、index、working tree 与引用的关系。
- Git 官方 [user manual](https://git-scm.com/docs/user-manual)：核验高层命令在 working tree、index 与对象数据库之间移动数据。
