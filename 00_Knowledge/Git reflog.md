---
aliases:
  - "Git reflog 记录本地引用移动但不是永久备份"
  - Git reflog is local temporary reference history
  - Git reflog 恢复边界
student_os: knowledge-atom
atom_id: CS-GIT-011
atom_type: recovery
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git 分支引用]]"
  - "[[Git HEAD 与分离状态]]"
---

# Git reflog 记录本地引用移动但不是永久备份

<!-- bilingual-en:start -->
*Git reflog records local reference movements but is not a permanent backup*
<!-- bilingual-en:end -->

> [!summary] 原子恢复机制
> 仓库为某个 ref 保留 reflog 时，它会记录 branch、HEAD 等引用曾从哪个对象移动到哪个对象。它常能找回 reset、rebase 或 detached HEAD 后暂时不可达的 commit，但记录会过期，而且不会随普通 push/pull 在仓库之间共享。
>
> <!-- bilingual-en:start -->
> When a repository keeps a reflog for a ref, it records how references such as branches and HEAD moved locally. It can often locate commits made temporarily unreachable by reset, rebase, or detached HEAD, but entries expire and are not shared between repositories by ordinary push or pull.
> <!-- bilingual-en:end -->

例如误把 `main` 从 C5 reset 到 C2 后，`git log` 从当前分支不再显示 C3–C5；`git reflog` 仍可能显示 reset 前的旧端点。确认旧 commit 后，应立即创建 branch 或 tag 让它重新可达，再继续修复，而不是长期依赖 `HEAD@{n}` 这种位置编号。
<!-- bilingual-en:start -->
For example, after accidentally resetting `main` from C5 to C2, ordinary log traversal from the branch no longer shows C3–C5. The reflog may still reveal the old tip. Once the desired commit is identified, create a branch or tag to make it reachable again instead of relying indefinitely on a positional expression such as `HEAD@{n}`.
<!-- bilingual-en:end -->

普通 non-bare 仓库默认启用常见 refs 的 reflog，而 bare 仓库的 `core.logAllRefUpdates` 默认关闭；具体 ref 也可能没有 reflog。默认过期策略通常是可达条目 90 天、从当前 tip 不可达的条目 30 天，并可被配置或维护提前改变；克隆出的另一台机器也没有你本地此前的引用移动。因此 reflog 是事故后的恢复线索，不是异地、永久或不可篡改的备份制度。
<!-- bilingual-en:start -->
A normal non-bare repository enables reflogs for common refs by default, whereas `core.logAllRefUpdates` defaults to false in a bare repository; a particular ref may therefore have no reflog. Default expiry is normally 90 days for entries in general and 30 days for entries unreachable from the current tip, and configuration or maintenance may change that. Another clone also lacks the previous movements of your local references. A reflog is therefore a recovery clue after an accident, not an off-device, permanent, or immutable backup system.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> reflog 找到误删分支的旧 tip 后，为什么还要立即新建分支？
>
> **答案：** reflog 条目会过期；新分支把该 commit 重新变成正常可达历史，避免恢复线索被清理。

## 来源与核验

- Git 官方文档，[`git-reflog`](https://git-scm.com/docs/git-reflog)：核验 reflog 记录本地引用端点更新、旧值语法、默认过期与清理机制。
- Git 官方文档，[`git-config`](https://git-scm.com/docs/git-config#Documentation/git-config.txt-corelogAllRefUpdates)：核验 non-bare 与 bare 仓库创建 reflog 的默认差异。
- Git 官方 [user manual](https://git-scm.com/docs/user-manual)：核验 reflog 与共享 commit 历史不同，只反映本地引用如何移动。
