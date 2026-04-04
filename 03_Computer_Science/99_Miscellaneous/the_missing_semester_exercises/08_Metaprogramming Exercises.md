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

>[!note]
> 对应主笔记：[[the_missing_semester#第 8 讲 元编程]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/metaprogramming/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//metaprogramming-solution

## 练习清单

1. 给课程里的 `Makefile` 添加一个 `clean` 目标：
   - 用于清理构建结果并让 `make` 可以重新构建
   - 将其设置成 phony target
   - 提示：`git ls-files` 可能有帮助
2. 学习 Rust/Cargo 的依赖版本要求语法，并分别为这些语法构造合理场景：
   - 尖号
   - 波浪号
   - 通配符
   - 比较运算
   - 多版本要求
3. 在任意 Git 仓库中写一个 `pre-commit` hook：
   - 提交前自动执行 `make paper.pdf`
   - 构建失败时拒绝提交
4. 用 GitHub Pages 发布任意一个自动部署页面，并添加一个 GitHub Action：
   - 对仓库中所有 shell 文件运行 `shellcheck`
5. 自己构建一个 GitHub Action：
   - 对仓库中所有 `.md` 文件执行 `proselint` 或 `write-good`
   - 在仓库中启用它
   - 提交一个含错误的文件，确认 Action 真正生效
