---
aliases:
  - The Missing Semester Lecture 8 Exercises
  - Missing Semester Metaprogramming Exercises
tags:
  - computer-science
  - tools
  - exercises
  - build
  - ci
  - the-missing-semester
---

# 第 8 讲 元编程 练习
<!-- bilingual-en:start -->
*Lecture 8: Metaprogramming Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 8 讲 元编程]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/metaprogramming/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//metaprogramming-solution

## 练习清单
<!-- bilingual-en:start -->
*Exercise Checklist*
<!-- bilingual-en:end -->

1. 给课程里的 `Makefile` 添加一个 `clean` 目标：
   - 用于清理构建结果并让 `make` 可以重新构建
   - 将其设置成 phony target
   - 提示：`git ls-files` 可能有帮助
<!-- bilingual-en:start -->
1. Add a `clean` target to the course `Makefile`. It should remove build outputs so that `make` can rebuild them, and it should be declared a phony target. Hint: `git ls-files` may help.
<!-- bilingual-en:end -->
2. 学习 Rust/Cargo 的依赖版本要求语法，并分别为这些语法构造合理场景：
   - 尖号
   - 波浪号
   - 通配符
   - 比较运算
   - 多版本要求
<!-- bilingual-en:start -->
2. Learn the Rust/Cargo syntax for dependency version requirements and construct a sensible use case for each form: caret requirements, tilde requirements, wildcards, comparison operators, and multiple version constraints.
<!-- bilingual-en:end -->
3. 在任意 Git 仓库中写一个 `pre-commit` hook：
   - 提交前自动执行 `make paper.pdf`
   - 构建失败时拒绝提交
<!-- bilingual-en:start -->
3. Write a `pre-commit` hook in any Git repository that runs `make paper.pdf` before each commit and rejects the commit if the build fails.
<!-- bilingual-en:end -->
4. 用 GitHub Pages 发布任意一个自动部署页面，并添加一个 GitHub Action：
   - 对仓库中所有 shell 文件运行 `shellcheck`
<!-- bilingual-en:start -->
4. Publish any automatically deployed site with GitHub Pages and add a GitHub Action that runs `shellcheck` on every shell file in the repository.
<!-- bilingual-en:end -->
5. 自己构建一个 GitHub Action：
   - 对仓库中所有 `.md` 文件执行 `proselint` 或 `write-good`
   - 在仓库中启用它
   - 提交一个含错误的文件，确认 Action 真正生效
<!-- bilingual-en:start -->
5. Build your own GitHub Action to run `proselint` or `write-good` on every `.md` file. Enable it in the repository, commit a file containing a deliberate error, and confirm that the action detects it.
<!-- bilingual-en:end -->
