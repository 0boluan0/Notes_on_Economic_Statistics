---
student_os: knowledge-atom
atom_id: CS-PY-SET-OBS-001
aliases:
  - Sets and repeated observations
  - Distinct values versus observations
  - 集合不保留观测次数
status: source-checked
atom_type: distinction
---

<!-- bilingual-en:start -->
# A set keeps distinct values, not observation counts
<!-- bilingual-en:end -->

集合保留不同取值，不保留每个值出现了几次。

<!-- bilingual-en:start -->
A Python set answers questions such as “has this value appeared?” and “which values occur in one source but not another?”. It retains one representative of equal values and gives no positional-order guarantee. Converting observations to a set therefore changes the information available for analysis.
<!-- bilingual-en:end -->

Python 集合适合回答“这个值出现过吗”“哪些值在一份数据里有、另一份没有”。相等的值只保留一个代表，也不保证位置顺序。因此，把观测记录转成集合，会改变之后还能分析哪些信息。

```python
delays = [0, 0, 9]
distinct = set(delays)
assert len(delays) == 3
assert distinct == {0, 9}
assert sum(delays) / len(delays) == 3
assert sum(distinct) / len(distinct) == 4.5
```

<!-- bilingual-en:start -->
The first mean describes three buses; the second averages two distinct delay values. Two buses both arriving on time is not a duplicated observation merely because both values equal zero. Keep the list when repeated observations matter.
<!-- bilingual-en:end -->

第一个平均数描述三辆公交，第二个平均数只描述两种不同延误数值。两辆车都准点，不代表其中一条记录多余；如果关心观测次数，就要保留列表里的重复值。

<!-- bilingual-en:start -->
An empty set is `set()`; `{}` creates an empty dictionary. Set elements must be [[哈希键语义稳定性|hashable]], which rules out ordinary lists, dictionaries and mutable sets as elements. A tuple can be an element only when all its members are hashable. This restriction does not apply to list elements: a list can contain sets.
<!-- bilingual-en:end -->

空集合写 `set()`；`{}` 建立的是空字典。集合元素必须可哈希，所以不能直接放入普通 list、dict 或可变 set。tuple 只有每个成员都可哈希时，才能成为集合元素。列表没有这个限制，列表里可以放集合。

<!-- bilingual-en:start -->
For two sets, `A - B` selects values present in `A` and absent from `B`; it does not subtract numbers pair by pair. For example, `{"Central", "Museum"} - {"Central", "Park"}` is `{"Museum"}`. Set differences can diagnose mismatched names, but cannot establish matching observation counts.
<!-- bilingual-en:end -->

`A - B` 选出“在 A 中、但不在 B 中”的值，并不是逐项做数值减法。例如 `{"Central", "Museum"} - {"Central", "Park"}` 得到 `{"Museum"}`。集合差可以查站名是否一致，却不能证明各站的记录次数一致。

## 来源与核验

- [Python Tutorial — Sets](https://docs.python.org/3/tutorial/datastructures.html#sets): construction, uniqueness and difference / 集合构造、去重与差集。
- [Python Glossary — Hashable](https://docs.python.org/3/glossary.html#term-hashable): permitted elements and tuple conditions / 集合元素与元组的可哈希条件。
- [[ST445 - Week 01 - Lecture.pdf#page=41|ST445 Lecture 1, p. 41]]: course distinction between containers; bus means are a worked supplement / 容器的课程区分，公交平均数是补充例子。
