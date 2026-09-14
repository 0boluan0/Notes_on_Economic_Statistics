---
aliases:
  - "git diff 的含义取决于比较的两个端点"
  - Git diff endpoints determine its meaning
  - Git diff 比较端点
student_os: knowledge-atom
atom_id: CS-GIT-007
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Git 版本控制.canvas]]"
requires:
  - "[[Git 三棵状态]]"
contrasts_with:
  - "[[Git 暂存区]]"
---

# git diff 的含义取决于比较的两个端点

<!-- bilingual-en:start -->
*The meaning of git diff depends on its two comparison endpoints*
<!-- bilingual-en:end -->

> [!summary] 原子过程
> diff 不是“当前所有修改”的同义词。先说明要比较哪两个状态，再选择命令；否则空输出和大段输出都可能被误读。
>
> <!-- bilingual-en:start -->
> A diff is not synonymous with “all current changes.” State the two states to compare before choosing the command; otherwise both empty and large outputs can be misread.
> <!-- bilingual-en:end -->

最常用的三个比较是：

| 问题 | 比较端点 | 常用命令 |
|---|---|---|
| 哪些编辑还没暂存？ | working tree ↔ index | `git diff` |
| 下一次提交会比当前 commit 多什么？ | index ↔ HEAD | `git diff --cached` |
| 本地全部跟踪修改相对当前 commit 是什么？ | working tree ↔ HEAD | `git diff HEAD` |

任意两个 commit、tree 或 blob 也可以比较。`git diff A..B` 与 `git diff A B` 同义；`git diff A...B` 则比较 `git merge-base A B` 选出的共同祖先与 B。这里的两点和三点都指定比较端点，不是遍历 commit range。若仓库还没有第一次 commit，HEAD 不存在；此时不指定 commit 的 `git diff --cached` 会显示全部 staged changes。
<!-- bilingual-en:start -->
The three everyday comparisons are working tree versus index (`git diff`), index versus HEAD (`git diff --cached`), and working tree versus HEAD (`git diff HEAD`). Arbitrary commits, trees, or blobs may also be compared. `git diff A..B` is synonymous with `git diff A B`, whereas `git diff A...B` compares the common ancestor selected by `git merge-base A B` with B. In this command both the two-dot and three-dot forms specify endpoints rather than a commit range. If the repository has no first commit and therefore no HEAD, `git diff --cached` with no commit displays all staged changes.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 想确认“现在执行 commit 会提交什么”，为什么不能只看普通 `git diff`？
>
> **答案：** 普通 diff 比较工作区与 index；真正候选提交是 index，通常应比较 index 与 HEAD，即查看 `git diff --cached`。unborn branch 尚无 HEAD 时，该命令显示全部 staged changes。

## 来源与核验

- Git 官方文档，[`git-diff`](https://git-scm.com/docs/git-diff)：核验 working tree、index、commit、tree 与 blob 之间各类 diff 的正式端点。
- Git 官方文档，[`git-diff-index`](https://git-scm.com/docs/git-diff-index)：交叉核验 cached 模式回答“已经标记为提交的内容与上一棵 tree 有何不同”。
