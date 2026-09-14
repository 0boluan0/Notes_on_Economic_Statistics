---
aliases:
  - "Git blob、tree 与 commit 组成不可变对象图"
  - Git object model
  - Git 对象模型
student_os: knowledge-atom
atom_id: CS-GIT-001
atom_type: model
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
---

# Git blob、tree 与 commit 组成不可变对象图

<!-- bilingual-en:start -->
*Git blobs, trees, and commits form an immutable object graph*
<!-- bilingual-en:end -->

> [!summary] 原子模型
> blob 保存文件内容，tree 把名称和模式连接到 blob 或子 tree，commit 再指向一个根 tree、父 commit 与作者等元数据。这三者构成项目快照历史的核心对象图。对象创建后不再原地修改；新状态产生新对象，并由对象 ID 连接成图。
>
> <!-- bilingual-en:start -->
> A blob stores file content, a tree connects names and modes to blobs or child trees, and a commit points to a root tree, parent commits, and metadata such as authorship. These three form the core object graph for project snapshots and history. Objects are not mutated in place after creation. A new state creates new objects linked into the graph by object IDs.
> <!-- bilingual-en:end -->

blob 本身不知道文件名；文件名属于 tree 的目录条目。tree 也不记录“上一版目录”，历史关系由 commit 的父指针表达。一个普通 commit 有一个父节点，初始 commit 没有父节点，merge commit 可以有多个父节点。沿父指针向后走，就得到该引用可达的项目历史。
<!-- bilingual-en:start -->
A blob does not know its filename; names belong to entries in a tree. A tree does not record the previous directory version either; historical relationships are represented by parent links between commits. An ordinary commit has one parent, an initial commit none, and a merge commit may have several. Following parent links yields the history reachable from a reference.
<!-- bilingual-en:end -->

Git 还有第四种对象：annotated tag object，它指向另一个对象并保存 tagger、日期和消息。lightweight tag 则只是直接指向对象的 ref。二者都不改变 blob → tree → commit 这条快照与历史主链。
<!-- bilingual-en:start -->
Git also has a fourth object type: an annotated tag object, which points to another object and stores a tagger, date, and message. A lightweight tag is instead a ref that points directly to an object. Neither changes the blob → tree → commit chain that represents snapshots and history.
<!-- bilingual-en:end -->

对象 ID 由对象类型和内容计算，所以同一内容可被多个快照复用，而内容变化会产生不同 ID。这使 Git 可以检查对象完整性，但内容寻址本身不等于远程备份，也不保证一个已经不可达的对象永久保留。
<!-- bilingual-en:start -->
An object ID is computed from the object's type and content, allowing identical content to be reused across snapshots while changed content receives another ID. This supports integrity checking, but content addressing is neither an off-device backup nor a promise that unreachable objects will be retained forever.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 文件名存在哪里？commit 又直接指向什么？
>
> **答案：** 文件名在 tree 条目中；commit 直接指向表示项目快照的根 tree，并记录父 commit 与元数据。

## 来源与核验

- Git 官方文档，[`gitdatamodel`](https://git-scm.com/docs/gitdatamodel)：核验 commit、tree、blob、tag 四类对象、不可变性及内容寻址。
- Git 官方文档，[`git`](https://git-scm.com/docs/git) 的 Discussion：交叉核验 blob、tree、commit 与父节点构成的对象图。
