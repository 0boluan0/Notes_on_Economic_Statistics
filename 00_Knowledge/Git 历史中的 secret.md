---
aliases:
  - "从当前文件删除 secret 不会清除 Git 历史"
  - Deleting a secret from the current file does not remove Git history
  - Git 历史中的秘密
student_os: knowledge-atom
atom_id: CS-GIT-013
atom_type: security-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git commit 快照与父指针]]"
---

# 从当前文件删除 secret 不会清除 Git 历史

<!-- bilingual-en:start -->
*Deleting a secret from the current file does not remove it from Git history*
<!-- bilingual-en:end -->

> [!summary] 原子安全边界
> 一个 secret 一旦进入 commit，后续“删除该行”的 commit 只生成新快照；旧 commit 仍保存原内容，并可能已被其他 clone、fork、缓存或制品取得。首要动作是吊销或轮换凭据，而不是只改文件。
>
> <!-- bilingual-en:start -->
> Once a secret enters a commit, a later commit that deletes the line only creates a new snapshot. The old commit still contains the value and may already exist in clones, forks, caches, or artifacts. The first response is to revoke or rotate the credential, not merely edit the file.
> <!-- bilingual-en:end -->

事故处理至少分三件事：

1. **停止继续暴露**：吊销、轮换或禁用 secret，并检查使用记录。
2. **修正当前版本**：从代码和配置中移除值，改用合适的 secret 管理方式，并加忽略/扫描规则防止重犯。
3. **按需要重写历史**：若必须从可达历史清除内容，使用专门的历史重写工具，更新所有相关引用，并协调所有 clone；旧对象或外部副本仍可能存在。

<!-- bilingual-en:start -->
Incident response separates three actions: revoke or rotate the credential and inspect its use; remove it from the current project and prevent recurrence; and, when required, rewrite every relevant reachable history with a suitable tool while coordinating all clones. Rewriting does not recall copies already obtained elsewhere.
<!-- bilingual-en:end -->

这里的关键不是背某一条历史清理命令。凭据一旦暴露就不能再假设保密；即使仓库历史被成功重写，也不能把轮换当成可选步骤。
<!-- bilingual-en:start -->
The key is not memorising one history-cleaning command. Once a credential is exposed, its secrecy can no longer be assumed. Even a successful history rewrite does not make rotation optional.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 已把 API key 从最新 commit 删除，为什么仍必须轮换它？
>
> **答案：** 旧 commit 和外部副本可能仍含该值；删除当前文件无法证明无人取得过凭据。

## 来源与核验

- [*Pro Git: Rewriting History — Removing a File from Every Commit*](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History.html)：核验从当前版本删除文件不会清除每个历史快照，以及历史清理需要重写相关 commit。
- GitHub 官方文档，[Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)：核验先吊销/轮换凭据、再清理历史与协调 clone 的事故顺序。
