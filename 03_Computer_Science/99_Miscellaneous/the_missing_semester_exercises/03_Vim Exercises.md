---
aliases:
  - The Missing Semester Lecture 3 Exercises
  - Missing Semester Vim Exercises
tags:
  - computer-science
  - tools
  - exercises
  - vim
  - editor
  - the-missing-semester
---

# 第 3 讲 编辑器 (Vim) 练习

>[!note]
> 对应主笔记：[[the_missing_semester#第 3 讲 编辑器 (Vim)]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/editors/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//editors-solution

## 练习清单

1. 完成 `vimtutor`。
   官方备注：在 80x24 终端窗口中体验最佳。
2. 下载官方提供的 `vimrc`，保存到 `~/.vimrc`。
   用 Vim 通读它，并观察新配置下 Vim 的外观与行为变化。
3. 安装并配置 `ctrlp.vim` 插件：
   - 创建插件目录：`mkdir -p ~/.vim/pack/vendor/start`
   - 克隆插件：`cd ~/.vim/pack/vendor/start && git clone https://github.com/ctrlpvim/ctrlp.vim`
   - 阅读插件文档
   - 在工程目录里打开 Vim，用 `:CtrlP` 启动文件定位
   - 修改 `~/.vimrc`，让 `Ctrl-P` 可以直接打开 CtrlP
4. 在自己的机器上，把课程里的 Vim 演示完整重做一遍。
5. 接下来一个月，所有文本编辑都尽量用 Vim 完成。
   每当觉得低效、或怀疑“一定有更好的方法”，就去搜索对应做法。
6. 在其他工具中启用 Vim 快捷键。
7. 继续自定义 `~/.vimrc`，并安装更多插件。
8. 高阶：使用 Vim 宏将 XML 转换为 JSON。
   官方示例文件：`/2020/files/example-data.xml`
