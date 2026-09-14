---
aliases:
  - "Git 合并冲突要求重建目标内容并验证而不是机械选边"
  - Resolve Git conflicts by reconstructing intent
  - Git 合并冲突处理
student_os: knowledge-atom
atom_id: CS-GIT-012
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git merge 与 rebase]]"
---

# Git 合并冲突要求重建目标内容并验证而不是机械选边

<!-- bilingual-en:start -->
*A Git merge conflict requires reconstructing and validating the intended result, not mechanically choosing one side*
<!-- bilingual-en:end -->

> [!summary] 原子过程
> 冲突表示 Git 无法从共同祖先与两个端点自动确定合并结果。`ours`、`theirs` 和冲突标记是证据来源，不是两个必选答案；最终文件可以同时采用、重写或舍弃两边内容。
>
> <!-- bilingual-en:start -->
> A conflict means Git cannot automatically determine the merged result from the common ancestor and the two tips. `ours`, `theirs`, and conflict markers are evidence, not two compulsory answers. The final file may combine, rewrite, or discard content from either side.
> <!-- bilingual-en:end -->

一个可靠的处理顺序是：

1. 用 `git status` 列出所有未合并路径，并确认当前操作是 merge、rebase 还是其他三方合并。
2. 查看共同祖先、ours 与 theirs 各自想完成什么，不只看冲突的几行文本。
3. 按目标行为重写工作区文件，移除标记；必要时同时调整测试、配置或调用者。
4. 运行相关测试、构建或人工检查。
5. `git add` 标记该路径已解决，再用 status 和 staged diff 审阅整个结果，最后继续或完成操作。

<!-- bilingual-en:start -->
A reliable sequence is to identify every unmerged path and the operation in progress; inspect the common ancestor and the intent of both sides; reconstruct the desired working-tree content; run the relevant validation; then stage the resolved paths and review the complete staged result before continuing.
<!-- bilingual-en:end -->

在 rebase 中，`ours` / `theirs` 的直觉还可能与“我正在重放的提交”相反，所以机械接受一边尤其危险。命令标签随操作语境变化，产品或程序应满足的最终行为才是稳定标准。
<!-- bilingual-en:start -->
During rebase, the intuitive meaning of `ours` and `theirs` may even appear reversed relative to “the commit I am replaying,” making mechanical selection particularly risky. Labels vary with the operation; the intended final behaviour of the program or artifact is the stable criterion.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 冲突文件能成功编译，是否足以证明冲突已正确解决？
>
> **答案：** 不足。编译只排除一部分语法或类型错误；还要核对两边意图、相关测试、行为与整份 staged diff。

## 来源与核验

- [*Pro Git: Basic Branching and Merging*](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging.html)：核验三方合并、未合并路径、冲突标记、暂存解决结果与完成 merge 的流程。
- [*Pro Git: Advanced Merging*](https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging.html)：支持从三方版本与目标内容理解复杂冲突，而非机械接受一侧。
