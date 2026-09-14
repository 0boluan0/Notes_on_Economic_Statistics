---
aliases:
  - "Python dict 保持插入顺序但仍按键哈希查找"
  - Python dictionaries preserve insertion order
  - Python dict 顺序语义
student_os: knowledge-atom
atom_id: CS-HASH-012
atom_type: language-semantics
status: source-checked
mastery_state: unassessed
part_of:
  - "[[哈希表.canvas]]"
requires:
  - "[[哈希相等性契约]]"
  - "[[哈希键语义稳定性]]"
---

# Python dict 保持插入顺序但仍按键哈希查找

<!-- bilingual-en:start -->
*A Python dict preserves insertion order while still using hashed keys for lookup*
<!-- bilingual-en:end -->

> [!summary] 原子语言语义
> Python 字典按键进入该字典的先后顺序迭代；给已有相等键赋新值只是更新，不算再次插入。这种顺序保证没有把字典变成排序映射，按键访问也仍要求键可哈希。
>
> <!-- bilingual-en:start -->
> A Python dictionary iterates in key-insertion order; assigning through an equal key that is already present is an update, not a new insertion. This ordering guarantee does not turn a dictionary into a sorted mapping, and key access still requires hashable keys.
> <!-- bilingual-en:end -->

对已有键重新赋值只替换对应值，不改变该键的位置；删除后再插入则把它放到迭代顺序末尾。例如：

```python
d = {"a": 1, "b": 2}
d["a"] = 3       # 顺序仍是 a, b
del d["a"]
d["a"] = 4       # 顺序变成 b, a
```

<!-- bilingual-en:start -->
Assigning a new value to an existing key replaces its value without moving it. Deleting and reinserting the key puts it at the end of iteration order, as the example shows.
<!-- bilingual-en:end -->

这项保证从 Python 3.7 起属于语言规范。只要构造与更新顺序相同，它就能提供可复现的迭代顺序；但序列化工具仍可能自行排序或改写输出。字典也不提供按键大小排序、按位置二分查找或任意“最旧优先”策略。若任务需要按键排序，仍要显式排序；若需要频繁按两种顺序索引，可能需要不同的数据结构。
<!-- bilingual-en:start -->
This has been a language guarantee since Python 3.7. When construction and updates occur in the same order, it provides reproducible iteration; a serializer may still sort or otherwise rewrite its output. A dictionary does not provide key sorting, binary search by position, or every possible age-based policy. Sort explicitly when key order is required, and choose another structure when the task needs multiple maintained orderings.
<!-- bilingual-en:end -->

字典相等性也不比较插入顺序：只要键值对相同，两个顺序不同的字典仍然相等。因此，“迭代有序”不能推出“顺序参与字典的值语义”。
<!-- bilingual-en:start -->
Dictionary equality also ignores insertion order: dictionaries with the same key-value pairs compare equal even when their iteration orders differ. Ordered iteration therefore does not make order part of a dictionary's value semantics.
<!-- bilingual-en:end -->

> [!question]- 回忆与辨别
> 更新 `d[k]` 的值与先删除再重新插入 `k`，对迭代顺序有何不同？
>
> **答案：** 直接更新不移动原位置；删除后重插会把键放到末尾。

## 来源与核验

- Python 标准库，[`dict` 映射类型](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)：核验键必须可哈希、字典保持插入顺序、更新不移动键，以及删除后重插进入末尾。
- Python 术语表，[`hashable`](https://docs.python.org/3/glossary.html#term-hashable) 与语言参考 [`object.__hash__`](https://docs.python.org/3/reference/datamodel.html#object.__hash__)：核验字典键的哈希值稳定性与相等契约。
