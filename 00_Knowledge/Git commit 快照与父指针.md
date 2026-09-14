---
aliases:
  - "Git commit 保存快照与父指针而不是差异袋"
  - Git commits are snapshots, not diffs
  - Git 提交不是差异包
student_os: knowledge-atom
atom_id: CS-GIT-002
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git 对象图]]"
---

# Git commit 保存快照与父指针而不是差异袋

<!-- bilingual-en:start -->
*A Git commit stores a snapshot and parent links, not a bag of differences*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> commit 指向一次完整项目状态的根 tree，并记录父 commit。diff 是选定两个状态以后计算出的比较视图，不是 commit 的存储本体。
>
> <!-- bilingual-en:start -->
> A commit points to the root tree of a complete project state and records its parent commits. A diff is a comparison view computed after two states are chosen; it is not the stored essence of the commit.
> <!-- bilingual-en:end -->

若 commit C2 的父节点是 C1，`git show C2` 常显示“C2 相对 C1 改了什么”，于是容易误以为 C2 只保存这份补丁。实际上，C2 指向自己的完整 tree；显示的差异来自事后比较 C1 与 C2。给 C2 选择另一个比较端点，结果也会改变，而 C2 对象本身不变。
<!-- bilingual-en:start -->
If commit C2 has parent C1, `git show C2` commonly displays what changed from C1 to C2, which can make C2 look like a stored patch. In fact, C2 points to its own complete tree. The displayed difference is computed by comparing C1 and C2 afterward. Choosing another comparison endpoint changes the diff without changing the C2 object.
<!-- bilingual-en:end -->

“快照”也不意味着 Git 每次机械复制所有文件。没有变化的内容可以继续引用已有 blob 或 tree；模型上仍是完整状态，存储上则可复用对象并打包压缩。把逻辑模型和物理优化分开，既能正确理解历史，也不会误以为每次提交必然占用整个项目大小。
<!-- bilingual-en:start -->
“Snapshot” does not mean that Git mechanically copies every file on each commit. Unchanged content may keep referring to existing blobs or trees. The logical model is a complete state, while physical storage can reuse and pack objects. Separating those levels explains history without implying that every commit consumes the full project size.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 为什么同一个 commit 可以针对不同基准显示不同 diff？
>
> **答案：** commit 保存的是快照；diff 由调用者选择的两个端点计算。换基准会换比较结果，却不会修改 commit。

## 来源与核验

- [*Pro Git: What is Git?*](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)：核验 Git 以快照而非一串差异来组织数据，并复用未变化内容。
- [*Pro Git: Branches in a Nutshell*](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)：核验 commit 指向暂存内容的 tree、父 commit 与元数据。
