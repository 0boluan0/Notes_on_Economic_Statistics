---
student_os: knowledge-atom
atom_id: CS-PY-TUPLE-SINGLE-001
aliases:
  - One-item tuple
  - Python singleton tuple
  - 单元素 tuple
status: source-checked
atom_type: syntax-rule
---

<!-- bilingual-en:start -->
# A one-item tuple needs a comma
<!-- bilingual-en:end -->

单元素元组需要逗号。

<!-- bilingual-en:start -->
A tuple is an ordered container. To put one value in a tuple, write a comma after it: `(3,)`. Parentheses alone group an expression, so `(3)` is still the integer `3`. The decimal point in `3.0`, rather than parentheses, makes that value a float.
<!-- bilingual-en:end -->

元组是有顺序的容器。只有一个值时，要在它后面加逗号：`(3,)`。单纯加括号只是把表达式括起来，所以 `(3)` 仍是整数 3；`3.0` 是浮点数，是因为数字的写法包含小数点。

```python
assert type((3)) is int
assert type((3,)) is tuple
assert type((3.0)) is float
assert len((3,)) == 1
assert () == tuple()          # empty tuple
```

<!-- bilingual-en:start -->
This matters when an operation expects a container: `len((3,))` works, but `len((3))` raises `TypeError` because an integer has no length. Writing `value = 3,` also creates a tuple; parentheses make the grouping clearer in a larger expression. How a tuple is constructed is separate from [[元组不可变边界|which nested objects may still change]].
<!-- bilingual-en:end -->

当操作要求容器时，这个逗号会直接影响结果：`len((3,))` 可以运行，`len((3))` 则会报 `TypeError`，因为整数没有长度。`value = 3,` 也会建立元组；括号让较长表达式里的分组更清楚。元组的构造语法与它内部对象能否改变，是两个需要分别理解的问题。

## 来源与核验

- [Python Tutorial — Tuples and Sequences](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences): one-item and empty-tuple construction; 单元素与空元组的构造规则。
- [[ST445 - Week 01 - Lecture.pdf#page=41|ST445 Lecture 1, pp. 41–42]]: course context for tuples; the comma example is a supplementary explanation / 元组的课程背景，逗号例子是补充说明。
