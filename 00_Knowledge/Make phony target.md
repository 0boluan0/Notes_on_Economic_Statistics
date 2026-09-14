---
aliases:
  - "GNU Make 的 phony target 表示动作而不是同名文件"
  - GNU Make phony target
  - Make 伪目标
student_os: knowledge-atom
atom_id: CS-BUILD-005
atom_set: build-dependencies-ci
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[构建规则三要素]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# GNU Make 的 phony target 表示动作而不是同名文件
<!-- bilingual-en:start -->
*A GNU Make phony target denotes an action rather than a file with the same name*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> `.PHONY: clean` 告诉 GNU Make，`clean` 是一个动作名而不是应由配方生成的文件。Make 在考虑这个目标时会无条件运行其配方，也不会因为目录里碰巧存在名为 `clean` 的文件而跳过它。
> <!-- bilingual-en:start -->
> `.PHONY: clean` tells GNU Make that `clean` names an action rather than a file produced by its recipe. Whenever Make considers this target, it runs the recipe unconditionally and does not skip it merely because a file named `clean` happens to exist.
> <!-- bilingual-en:end -->

```make
.PHONY: clean
clean:
	rm -f paper.pdf plot-*.png
```

`.PHONY` 还会跳过该目标的隐式规则搜索。它适合 `clean`、`test` 这类命令式入口，但不适合表示希望通过时间戳维护的真实产物。
<!-- bilingual-en:start -->
`.PHONY` also skips implicit-rule search for that target. It suits command-like entry points such as `clean` or `test`, but not real artifacts whose freshness should be tracked by timestamps.
<!-- bilingual-en:end -->

## 边界

phony target 通常不应成为真实文件目标的先决条件；否则每次考虑该文件时，phony 配方都会运行并可能迫使后续工作重复。phony target 可以有自己的先决条件，但“总会运行”只发生在它被请求或从请求目标可达时。
<!-- bilingual-en:start -->
A phony target should normally not be a prerequisite of a real file target, because its recipe will run every time that file is considered and may force repeated work. A phony target may have prerequisites of its own, but “always runs” applies only when it is requested or reachable from the requested goal.
<!-- bilingual-en:end -->

> [!question]- 自检
> 没写 `.PHONY: clean` 时，为什么新建一个空文件 `clean` 可能让 `make clean` 什么也不做？
>
> **答案：** Make 会把 `clean` 当作真实文件目标，并可能根据文件存在性判断它不需要更新。

## 来源与核验

- [GNU Make Manual, Phony Targets](https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html)：核对文件名冲突、无条件执行、隐式规则搜索与真实目标先决条件的边界。
- [MIT Missing Semester, Metaprogramming exercises](https://missing.csail.mit.edu/2020/metaprogramming/#exercises)：核对 `clean` 练习的课程语境。
<!-- bilingual-en:start -->
- [GNU Make Manual: Phony Targets](https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html) was checked for name collisions, unconditional execution, implicit-rule search, and the real-target prerequisite boundary.
- [MIT Missing Semester: Metaprogramming exercises](https://missing.csail.mit.edu/2020/metaprogramming/#exercises) was checked for the course context of the `clean` exercise.
<!-- bilingual-en:end -->
