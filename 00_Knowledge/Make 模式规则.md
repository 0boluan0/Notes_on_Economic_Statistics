---
aliases:
  - "GNU Make 模式规则用同一 stem 实例化一族构建关系"
  - GNU Make pattern rule stem
  - Make 模式规则
student_os: knowledge-atom
atom_id: CS-BUILD-006
atom_set: build-dependencies-ci
atom_type: mechanism
status: source-checked
mastery_state: unassessed
requires:
  - "[[构建规则三要素]]"
part_of:
  - "[[构建、依赖与 CI.canvas]]"
---

# GNU Make 模式规则用同一 stem 实例化一族构建关系
<!-- bilingual-en:start -->
*A GNU Make pattern rule instantiates a family of build relationships with one shared stem*
<!-- bilingual-en:end -->

> [!summary] 原子机制
> GNU Make 的 pattern rule 在目标模式中使用一个 `%`。目标名与模式匹配出的非空子串叫 stem；先决条件模式里的 `%` 会替换成同一个 stem，从而由一条规则生成一族具体依赖关系。
> <!-- bilingual-en:start -->
> A GNU Make pattern rule uses one `%` in its target pattern. The nonempty substring matched in a concrete target name is the stem; `%` in prerequisite patterns is replaced by that same stem, instantiating a family of concrete dependency relationships from one rule.
> <!-- bilingual-en:end -->

```make
plot-%.png: %.dat plot.py
	./plot.py -i $*.dat -o $@
```

请求 `plot-sales.png` 时，stem 是 `sales`，对应先决条件为 `sales.dat` 与固定输入 `plot.py`。模式规则只有在匹配后的先决条件存在或能够生成时才可用。
<!-- bilingual-en:start -->
When `plot-sales.png` is requested, the stem is `sales`, giving prerequisites `sales.dat` and the fixed input `plot.py`. The pattern rule is applicable only when its instantiated prerequisites exist or can be made.
<!-- bilingual-en:end -->

## 边界

这里的 `%` 是 GNU Make 模式规则语法，不是 shell glob，也不能泛化为所有构建工具的模板语法。若多条模式规则都能匹配，Make 还要按自己的隐式规则选择过程挑选适用规则。
<!-- bilingual-en:start -->
This `%` belongs to GNU Make pattern-rule syntax; it is not a shell glob and must not be generalized to every build tool's templating syntax. If several pattern rules match, Make applies its own implicit-rule selection procedure.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对规则 `%.o: %.c headers.h` 请求 `parser.o` 时，stem 和两个先决条件是什么？
>
> **答案：** stem 是 `parser`；先决条件是 `parser.c` 与固定的 `headers.h`。

## 来源与核验

- [GNU Make Manual, Introduction to Pattern Rules](https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html)：核对 `%`、stem、先决条件替换与适用条件。
- [MIT Missing Semester, Metaprogramming](https://missing.csail.mit.edu/2020/metaprogramming/)：核对 `plot-%.png` 的课程例子。
<!-- bilingual-en:start -->
- [GNU Make Manual: Introduction to Pattern Rules](https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html) was checked for `%`, stems, prerequisite substitution, and applicability.
- [MIT Missing Semester: Metaprogramming](https://missing.csail.mit.edu/2020/metaprogramming/) was checked for the `plot-%.png` course example.
<!-- bilingual-en:end -->
