---
aliases:
  - "GNU Make 用文件存在性与修改时间判断普通文件目标是否过期"
  - GNU Make timestamp freshness rule
  - Make 文件目标过期判断
student_os: knowledge-atom
atom_id: CS-BUILD-003
atom_set: build-dependencies-ci
atom_type: system-semantics
status: source-checked
mastery_state: unassessed
requires:
  - "[[构建规则三要素]]"
related:
  - "[[传递依赖递归构建]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# GNU Make 用文件存在性与修改时间判断普通文件目标是否过期
<!-- bilingual-en:start -->
*GNU Make decides whether an ordinary file target is out of date from existence and modification times*
<!-- bilingual-en:end -->

> [!summary] 原子语义
> 对普通文件目标，GNU Make 在目标不存在，或任一普通先决条件比目标更新时，认为目标过期并运行配方。若目标存在且不早于全部普通先决条件，Make 可以跳过配方。
> <!-- bilingual-en:start -->
> For an ordinary file target, GNU Make considers the target out of date when it does not exist or when any normal prerequisite is newer. If the target exists and is no older than every normal prerequisite, Make may skip the recipe.
> <!-- bilingual-en:end -->

这是一种基于文件修改时间的增量策略，不是内容哈希比较。仅仅 `touch` 一个输入可能触发重建；反过来，若内容改变却保留了较旧时间戳，Make 可能看不出变化。
<!-- bilingual-en:start -->
This is a modification-time-based incremental strategy, not a content-hash comparison. Merely touching an input can trigger a rebuild; conversely, a content change that retains an older timestamp can be invisible to Make.
<!-- bilingual-en:end -->

## 边界

这条规则描述 GNU Make 的普通文件目标，不应泛化成所有构建系统都只看时间戳。phony target、无配方规则、order-only prerequisite 以及采用内容寻址的其他构建系统有不同语义。
<!-- bilingual-en:start -->
This rule describes ordinary file targets in GNU Make and must not be generalized to every build system. Phony targets, rules without recipes, order-only prerequisites, and content-addressed build systems have different semantics.
<!-- bilingual-en:end -->

> [!question]- 自检
> 文件内容完全没变，但 `touch source.c` 后 `make` 重新编译，这能证明 Make 比较了内容吗？
>
> **答案：** 不能。恰恰是修改时间变新触发了普通先决条件的新旧判断。

## 来源与核验

- [GNU Make Manual, How Make Works](https://www.gnu.org/software/make/manual/html_node/How-Make-Works.html)：核对目标缺失或先决条件更新时的重建判断。
- [GNU Make Manual, Types of Prerequisites](https://www.gnu.org/software/make/manual/html_node/Prerequisite-Types.html)：核对普通先决条件参与过期判断、order-only prerequisite 不参与的边界。
<!-- bilingual-en:start -->
- [GNU Make Manual: How Make Works](https://www.gnu.org/software/make/manual/html_node/How-Make-Works.html) was checked for rebuilding when a target is absent or a prerequisite is newer.
- [GNU Make Manual: Types of Prerequisites](https://www.gnu.org/software/make/manual/html_node/Prerequisite-Types.html) was checked for normal prerequisites affecting freshness and the order-only boundary.
<!-- bilingual-en:end -->
