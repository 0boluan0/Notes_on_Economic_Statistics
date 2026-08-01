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
<!-- bilingual-en:start -->
*Lecture 3: Editor (Vim) Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 3 讲 编辑器 (Vim)]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/editors/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//editors-solution

## 练习清单
<!-- bilingual-en:start -->
*Exercise Checklist*
<!-- bilingual-en:end -->

1. 完成 `vimtutor`。
   官方备注：在 80x24 终端窗口中体验最佳。
<!-- bilingual-en:start -->

&nbsp;
**1.** Complete `vimtutor`. The official note recommends using an 80-by-24 terminal window for the best experience.<br>
<!-- bilingual-en:end -->
2. 下载官方提供的 `vimrc`，保存到 `~/.vimrc`。
   用 Vim 通读它，并观察新配置下 Vim 的外观与行为变化。
<!-- bilingual-en:start -->

&nbsp;
**2.** Download the provided `vimrc` and save it as `~/.vimrc`. Read through it in Vim and observe how the new configuration changes Vim's appearance and behavior.<br>
<!-- bilingual-en:end -->
3. 安装并配置 `ctrlp.vim` 插件：
   - 创建插件目录：`mkdir -p ~/.vim/pack/vendor/start`
   - 克隆插件：`cd ~/.vim/pack/vendor/start && git clone https://github.com/ctrlpvim/ctrlp.vim`
   - 阅读插件文档
   - 在工程目录里打开 Vim，用 `:CtrlP` 启动文件定位
   - 修改 `~/.vimrc`，让 `Ctrl-P` 可以直接打开 CtrlP
<!-- bilingual-en:start -->

&nbsp;
**3.** Install and configure the `ctrlp.vim` plugin. Create the plugin directory with `mkdir -p ~/.vim/pack/vendor/start`, clone it with `cd ~/.vim/pack/vendor/start && git clone https://github.com/ctrlpvim/ctrlp.vim`, and read its documentation. Open Vim inside a project, launch the file finder with `:CtrlP`, and update `~/.vimrc` so that `Ctrl-P` opens CtrlP directly.<br>
<!-- bilingual-en:end -->
4. 在自己的机器上，把课程里的 Vim 演示完整重做一遍。
<!-- bilingual-en:start -->

&nbsp;
**4.** Reproduce the complete Vim demonstration from the lecture on your own machine.<br>
<!-- bilingual-en:end -->
5. 接下来一个月，所有文本编辑都尽量用 Vim 完成。
   每当觉得低效、或怀疑“一定有更好的方法”，就去搜索对应做法。
<!-- bilingual-en:start -->

&nbsp;
**5.** For the next month, use Vim for as much text editing as possible. Whenever a task feels inefficient or you suspect that there must be a better method, look up the appropriate Vim technique.<br>
<!-- bilingual-en:end -->
6. 在其他工具中启用 Vim 快捷键。
<!-- bilingual-en:start -->

&nbsp;
**6.** Enable Vim-style keybindings in your other tools.<br>
<!-- bilingual-en:end -->
7. 继续自定义 `~/.vimrc`，并安装更多插件。
<!-- bilingual-en:start -->

&nbsp;
**7.** Continue customizing `~/.vimrc` and install additional plugins.<br>
<!-- bilingual-en:end -->
8. 高阶：使用 Vim 宏将 XML 转换为 JSON。
   官方示例文件：`/2020/files/example-data.xml`
<!-- bilingual-en:start -->

&nbsp;
**8.** Advanced: use a Vim macro to transform XML into JSON. The official example file is `/2020/files/example-data.xml`.<br>
<!-- bilingual-en:end -->
