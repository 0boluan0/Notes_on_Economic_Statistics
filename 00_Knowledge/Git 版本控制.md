---
aliases:
  - "Git Version Control"
  - "Git"
  - "Git版本控制"
status: source-checked
---

# Git 版本控制
<!-- bilingual-en:start -->
*Git Version Control*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用 Git 保存可解释的历史，用系统化调试定位根因，用构建/依赖工具复现结果，并在密码学威胁模型下保护数据。
> **具体锚点：** Git commit 是项目快照及父指针，不是“文件差异袋”；分支只是可移动的 commit 名称。
> **核心难点：** 工具解决不同层级：版本控制不等备份，测试不等安全，哈希不等加密。
> **为什么重要：** 可靠开发来自可回退、可复现、可验证和最小权限，而不是记命令。
> **继续：** 先建立 Git 数据模型和调试循环，再理解依赖/CI 与密码学原语的用途边界。
> <!-- bilingual-en:start -->
> **Problem addressed:** Use Git for explainable history, systematic debugging for root causes, build and dependency tools for reproducibility, and a cryptographic threat model for data protection.
> **Concrete anchor:** A Git commit is a project snapshot plus parent links, not a bag of file differences; a branch is merely a movable name for a commit.
> **Central difficulty:** These tools operate at different layers: version control is not backup, testing is not security, and hashing is not encryption.
> **Why it matters:** Reliable development comes from recoverability, reproducibility, verification, and least privilege rather than memorizing commands.
> **Continue with:** Establish the Git data model here, use [[测试、调试、异常与断言|testing and debugging]] for failures, then continue to [[构建、依赖与 CI|Builds, Dependencies, and CI]] and [[密码学原语与安全模型|Cryptographic Primitives and Security Models]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/) 与本地练习：支持 shell、Git、调试、构建与安全工作流。
> <!-- bilingual-en:start -->
> - The [official MIT Missing Semester course](https://missing.csail.mit.edu/2020/) and local exercises support shell, Git, debugging, build, and security workflows.
> <!-- bilingual-en:end -->

## Git 数据模型
<!-- bilingual-en:start -->
*The Git Data Model*
<!-- bilingual-en:end -->

blob 存文件内容，tree 存目录，commit 指向 tree、父 commit 和元数据；对象由内容寻址。branch 是指向 commit 的可移动引用，HEAD 表示当前检出位置。理解图比背 `checkout/reset` 更安全。
<!-- bilingual-en:start -->
A blob stores file content, a tree stores a directory, and a commit points to a tree, parent commits, and metadata; objects are content-addressed. A branch is a movable reference to a commit, and HEAD identifies the current checkout position. Understanding the graph is safer than memorizing `checkout` and `reset` recipes.
<!-- bilingual-en:end -->

提交图回答“哪些快照来自哪些父快照”。diff 是两个快照的比较结果，不是 commit 的存储本体；同一个 commit 可被多个分支或 tag 引用。
<!-- bilingual-en:start -->
The commit graph answers which snapshots descend from which parents. A diff is a comparison between snapshots, not the stored essence of a commit, and several branches or tags can refer to the same commit.
<!-- bilingual-en:end -->

## 工作区、暂存区与提交
<!-- bilingual-en:start -->
*Working Tree, Index, and Commit*
<!-- bilingual-en:end -->

工作区是当前文件，index/staging area 是下一提交快照，commit 是已保存历史。提交前看 status/diff，按逻辑范围 stage，写说明“为什么”。不要把 secrets 提交后只删除当前文件，历史仍保留。
<!-- bilingual-en:start -->
The working tree contains current files, the index or staging area describes the next snapshot, and a commit records saved history. Inspect status and diffs before committing, stage a coherent logical change, and explain why. Deleting a secret only from the current file does not remove it from history.
<!-- bilingual-en:end -->

分别比较 working tree↔index、index↔HEAD 和任意两个 commits，可准确回答修改处在哪一层。`git status` 是入口，diff 的比较端点必须说清。
<!-- bilingual-en:start -->
Comparing working tree to index, index to HEAD, or any two commits reveals the layer containing a change. `git status` is the entry point, and the endpoints of a diff must be explicit.
<!-- bilingual-en:end -->

## 分支、合并与回退
<!-- bilingual-en:start -->
*Branches, Merges, and Recovery*
<!-- bilingual-en:end -->

merge 保留两条历史，rebase 重写提交父关系。共享历史重写需协调。恢复优先新 commit/revert 等可追踪操作；reset 的不同模式会改变引用、index 或工作区，使用前解析精确影响。
<!-- bilingual-en:start -->
A merge preserves two lines of history, whereas a rebase rewrites parent relationships. Rewriting shared history requires coordination. Prefer traceable recovery such as a new fix or revert; different reset modes alter references, the index, or the working tree, so resolve their exact effects first.
<!-- bilingual-en:end -->

## Worked example：安全撤销已共享提交
<!-- bilingual-en:start -->
*Worked Example: Safely Undo a Shared Commit*
<!-- bilingual-en:end -->

若错误 commit 已推送并被他人基于它工作，`git revert <commit>` 创建一个反向变化的新 commit，保留原图和协作历史。用 rebase/reset 强行改远端会让他人的分支失去共同基点。
<!-- bilingual-en:start -->
If a faulty commit has been pushed and others have built on it, `git revert <commit>` creates a new commit containing the inverse change while preserving the graph and collaboration history. Forcing a rewritten remote with rebase or reset can remove the common base expected by collaborators.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 修改“消失”：依次看 branch、HEAD、status、stash 和 reflog，先定位引用与对象，再做恢复。
  <!-- bilingual-en:start -->
  A change “disappeared”: inspect the branch, HEAD, status, stash, and reflog in order; locate the reference and object before attempting recovery.
  <!-- bilingual-en:end -->
- 合并冲突：冲突标记只是两个分支无法自动合成的位置，必须根据目标行为重建内容并运行验证，而不是机械选一边。
  <!-- bilingual-en:start -->
  A merge conflicts: markers identify a location that could not be combined automatically; reconstruct the intended behavior and validate it rather than mechanically choosing one side.
  <!-- bilingual-en:end -->
- 不确定 reset 会删什么：停止，先在只读状态下说明它会移动哪个引用、改哪一层，再选择 revert 或备份分支。
  <!-- bilingual-en:start -->
  The effect of reset is unclear: stop and state which reference and layers it would change, then choose a revert or backup branch if appropriate.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### Git branch 本质是什么？
<!-- bilingual-en:start -->
*What is a Git branch fundamentally?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 一个可移动的名字/引用，指向某个 commit；提交后该引用前移。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It is a movable name or reference to a commit; making a new commit advances the current branch reference.
<!-- bilingual-en:end -->

### 为什么 commit 不是 diff？
<!-- bilingual-en:start -->
*Why is a commit not a diff?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> commit 指向完整 tree 和父节点；diff 是选择两个快照后计算出的比较视图。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A commit points to a complete tree and its parents; a diff is a comparison view computed after choosing two snapshots.
<!-- bilingual-en:end -->

### 已共享提交为什么通常用 revert 而非 reset？
<!-- bilingual-en:start -->
*Why is a shared commit normally undone with revert rather than reset?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> revert 追加可追踪的反向提交，不改他人已依赖的历史；reset 会移动引用并可能要求危险的强制推送。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Revert appends a traceable inverse commit without rewriting history on which others rely; reset moves a reference and may require a dangerous force push.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [MIT Missing Semester official course](https://missing.csail.mit.edu/2020/) 与本地练习：支持 shell、Git、调试、构建与安全工作流。
  <!-- bilingual-en:start -->
  The [official MIT Missing Semester course](https://missing.csail.mit.edu/2020/) and local exercises support shell, Git, debugging, build, and security workflows.
  <!-- bilingual-en:end -->
- [Git data model documentation](https://git-scm.com/docs/gitdatamodel)：核验 objects、references、index 与 reflog 的正式含义。
  <!-- bilingual-en:start -->
  The [Git data model documentation](https://git-scm.com/docs/gitdatamodel) verifies the formal meanings of objects, references, the index, and reflogs.
  <!-- bilingual-en:end -->
