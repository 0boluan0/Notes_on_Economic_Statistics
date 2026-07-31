---
aliases:
  - The Missing Semester Lecture 6 Exercises
  - Missing Semester Version Control Exercises
tags:
  - computer-science
  - tools
  - exercises
  - git
  - version-control
  - the-missing-semester
---

# 第 6 讲 版本控制 (Git) 练习
<!-- bilingual-en:start -->
*Lecture 6: Version Control (Git) Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 6 讲 版本控制 (Git)]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/version-control/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//version-control-solution

## 练习清单
<!-- bilingual-en:start -->
*Exercise Checklist*
<!-- bilingual-en:end -->

1. 如果你之前没系统用过 Git：
   - 阅读 `Pro Git` 前几章，或
   - 完成 `Learn Git Branching`
   重点把命令和 Git 的数据模型联系起来理解。
<!-- bilingual-en:start -->
1. If you have not used Git systematically before, read the opening chapters of `Pro Git` or complete `Learn Git Branching`. Focus on connecting each command to Git's underlying data model.
<!-- bilingual-en:end -->
2. 克隆课程网站仓库：
   https://github.com/missing-semester-cn/missing-semester-cn.github.io.git
   然后完成：
   - 把版本历史可视化并探索
   - 找出最后修改 `README.md` 的人
   - 找出最后一次修改 `_config.yml` 中 `collections:` 行的提交信息
<!-- bilingual-en:start -->
2. Clone the course website repository from https://github.com/missing-semester-cn/missing-semester-cn.github.io.git. Visualize and explore its history, identify the last person to modify `README.md`, and find the commit message for the most recent change to the `collections:` line in `_config.yml`.
<!-- bilingual-en:end -->
3. 模拟一个常见错误：
   - 向仓库提交一个不该进 Git 的大文件或敏感文件
   - 再把它从历史中删除
   参考：https://help.github.com/articles/removing-sensitive-data-from-a-repository/
<!-- bilingual-en:start -->
3. Simulate a common mistake by committing a large or sensitive file that should not be tracked, then remove it from the repository's history. See https://help.github.com/articles/removing-sensitive-data-from-a-repository/.
<!-- bilingual-en:end -->
4. 克隆任意一个 GitHub 仓库，修改一些文件，然后：
   - 执行 `git stash`
   - 观察 `git log --all --oneline`
   - 再执行 `git stash pop`
   - 思考这个技巧什么时候有用
<!-- bilingual-en:start -->
4. Clone any GitHub repository and modify some files. Run `git stash`, inspect `git log --all --oneline`, then run `git stash pop`. Explain when this technique is useful.
<!-- bilingual-en:end -->
5. 在 `~/.gitconfig` 里创建一个 Git 别名，使 `git graph` 等价于：

```sh
git log --all --graph --decorate --oneline
```
<!-- bilingual-en:start -->
5. Add a Git alias to `~/.gitconfig` so that `git graph` runs the command shown above.
<!-- bilingual-en:end -->

6. 配置全局忽略文件：

```sh
git config --global core.excludesfile ~/.gitignore_global
```

   然后创建 `~/.gitignore_global`，忽略系统或编辑器临时文件，例如 `.DS_Store`。
<!-- bilingual-en:start -->
6. Configure the global ignore file with the command shown above. Then create `~/.gitignore_global` and add patterns for operating-system or editor-generated temporary files, such as `.DS_Store`.
<!-- bilingual-en:end -->
7. Fork 课程网站仓库，找到一个错别字或其他改进点，向 GitHub 提交一个 Pull Request。
<!-- bilingual-en:start -->
7. Fork the course website repository, find a typo or another improvement, and submit a pull request on GitHub.
<!-- bilingual-en:end -->
