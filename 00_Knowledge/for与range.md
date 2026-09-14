---
aliases:
  - "for 逐项绑定 iterable 而 range 表示半开整数序列"
  - for binds iterable items while range represents a half-open integer sequence
student_os: knowledge-atom
atom_id: CS-PY-CORE-007
atom_set: python-core
atom_type: control-flow
status: source-checked
mastery_state: unassessed
part_of:
  - "[[计算模型、Python 表达式与控制流.canvas]]"
requires:
  - "[[名称绑定与赋值]]"
related:
  - "[[控制转移语句]]"
---

# for 逐项绑定 iterable 而 range 表示半开整数序列
<!-- bilingual-en:start -->
*`for` binds items from an iterable while `range` represents a half-open integer sequence*
<!-- bilingual-en:end -->

> [!summary] 原子控制流
> Python 先求值 `for target in expression` 的右侧一次，取得 iterable 并创建 iterator；随后每取到一个元素，就按赋值规则把它绑定到 target 并执行循环体，直到 iterator 耗尽。`for` 不是只能“把计数器加一”的循环。`range(start, stop, step)` 是不可变整数序列，沿 step 方向包含 start 而排除 stop。
> <!-- bilingual-en:start -->
> Python evaluates the expression in `for target in expression` once, obtains an iterable, and creates an iterator. Each item from that iterator is assigned to the target under ordinary assignment rules before the body runs; the loop ends when the iterator is exhausted. A `for` loop is not limited to incrementing a counter. `range(start, stop, step)` is an immutable integer sequence that includes `start` and excludes `stop` in the direction of `step`.
> <!-- bilingual-en:end -->

```python
for char in "cat":       # binds char to "c", then "a", then "t"
    print(char)

list(range(1, 6, 2))     # [1, 3, 5]
list(range(5, 0, -2))    # [5, 3, 1]
```

省略 start 时默认为 0，省略 step 时默认为 1；step 为 0 会抛出 `ValueError`。起点已经越过 stop 且 step 方向无法返回时，range 为空。range 只保存 start、stop、step 并按需计算元素，不等于先物化一个大列表。循环结束后 target 名称通常保留最后一次绑定；若 iterable 为空，循环可能从未给它赋值。
<!-- bilingual-en:start -->
An omitted start defaults to 0 and an omitted step to 1; a zero step raises `ValueError`. A range is empty when the starting point is already beyond the stop in a direction the step cannot reverse. A range stores its start, stop, and step and computes elements as needed rather than first materializing a large list. The target name normally keeps its final binding after the loop; an empty iterable may leave it never assigned by that loop.
<!-- bilingual-en:end -->

修改正在遍历的底层容器没有一种适用于所有 iterable 的统一结果，行为取决于具体 iterator。若算法需要增删元素，先选快照、索引策略或显式工作队列，而不要假定 `for` 会自动冻结原容器。
<!-- bilingual-en:start -->
Mutating an underlying container during iteration has no one result shared by every iterable; behavior depends on the concrete iterator. If an algorithm must add or remove elements, choose a snapshot, an index strategy, or an explicit work queue instead of assuming that `for` freezes the original container.
<!-- bilingual-en:end -->

## 来源与核验

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_lec03.pdf|MIT 6.100L Lecture 3 slides]] 与 [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-OCW-offline-site/static_resources/mit6_100l_f22_lec04.pdf|Lecture 4 slides]]：核对逐项绑定、range 边界与直接遍历字符串。
- [Python Language Reference: the for statement](https://docs.python.org/3/reference/compound_stmts.html#the-for-statement)：核对 iterable 只求值一次、iterator、target 绑定与耗尽语义。
- [Python Standard Library: ranges](https://docs.python.org/3/library/stdtypes.html#ranges)：核对正负 step、半开边界、空 range、零 step 与存储边界。
