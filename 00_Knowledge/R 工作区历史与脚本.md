---
student_os: knowledge-atom
atom_id: r-base-workspace-history-script
aliases:
  - "An R workspace stores objects while a script records an analysis to execute"
  - "R 工作区保存对象，脚本记录要执行的分析过程"
status: source-checked
---
<!-- bilingual-en:start -->
# An R workspace stores objects while a script records an analysis to execute
<!-- bilingual-en:end -->

R 工作区保存对象，脚本记录要执行的分析过程

<!-- bilingual-en:start -->
`save.image()` saves objects in `.GlobalEnv` for later restoration. Interactive history records entered commands and can include experiments or errors. A cleaned `.R` script records the intended analysis steps; it remains dependent on its data, packages, settings and any required random seed.
<!-- bilingual-en:end -->

`save.image()` 保存 `.GlobalEnv` 的对象，供以后恢复。交互历史记录输入的命令，也可能含尝试和错误。整理好的 `.R` 脚本记录预期分析步骤；复现仍依赖数据、包、设置以及需要时的随机种子。
<!-- bilingual-en:start -->
A workspace image does not reinstall packages or recreate their attachment state. `savehistory()` depends on the interactive front end and is not a general history facility for batch `Rscript`. `.RData` and `.Rhistory` are conventional filenames, not suffixes these functions universally require.
<!-- bilingual-en:end -->

工作区文件不会重新安装包，也不会重建包的附加状态。`savehistory()` 依赖交互前端，不是批处理 `Rscript` 的通用历史功能。`.RData`、`.Rhistory` 是惯用命名，并非这两个函数普遍强制要求的后缀。
<!-- bilingual-en:start -->
`source()` executes a script, including any file-writing commands it reaches. `echo = TRUE` displays expressions; `print.eval` controls result printing and defaults to `echo`. Use explicit `print()` when a script should print a result regardless of those display settings.
<!-- bilingual-en:end -->

`source()` 会执行脚本，也包括执行过程中遇到的写文件命令。`echo = TRUE` 显示表达式，`print.eval` 控制结果打印，默认跟随 `echo`。需要稳定打印结果时，在脚本中明确写 `print()`。

## 来源与核验

- [[01_Math/09_ST447_Data_Analysis_and_Statistical_Methods/2026_Course_Materials/02_Lecture_Notes/ST447 - Chapter 01 - Descriptive Data Analysis in R.pdf#page=26|Chapter 1, pp. 26–29]], saving and organizing analysis / 保存与组织分析。
- [R: save](https://stat.ethz.ch/R-manual/R-patched/library/base/html/save.html), [history](https://stat.ethz.ch/R-manual/R-patched/library/utils/html/history.html), [source](https://stat.ethz.ch/R-manual/R-patched/library/base/html/source.html): saved objects, front-end limits and execution / 对象、前端限制与执行。
