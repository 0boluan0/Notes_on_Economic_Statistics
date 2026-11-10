---
student_os: knowledge-atom
atom_id: CS-PY-UNPACK-001
aliases:
  - Python iterable unpacking
  - Python sequence unpacking
  - Python 解包赋值
status: source-checked
atom_type: operation-semantics
---

<!-- bilingual-en:start -->
# Unpacking assigns items in iteration order
<!-- bilingual-en:end -->

解包按迭代顺序把元素分配给变量。

<!-- bilingual-en:start -->
In `first, second = items`, Python takes the items produced by the right-hand iterable and assigns them to the names on the left. Without a starred target, there must be exactly as many items as names; too few or too many raises `ValueError`. This operation works with lists and other iterables, not only tuples.
<!-- bilingual-en:end -->

`first, second = items` 的意思是：依次取出右边提供的元素，交给左边的变量。不使用星号接收时，元素数量必须和变量数量相同，太少或太多都会报 `ValueError`。解包不只用于 tuple，也适用于 list 等可迭代对象。

```python
first, second = [10, 20]
assert (first, second) == (10, 20)

record = {"stop": "Central", "delay_min": 0}
first, second = record
assert (first, second) == ("stop", "delay_min")
stop, delay = record.values()
assert (stop, delay) == ("Central", 0)
```

<!-- bilingual-en:start -->
A dictionary supplies its keys by default. Its keys and values follow [[Python dict 顺序与查找|insertion order]], so the `.values()` example works for this dictionary's construction order. If the variable must correspond to a particular field regardless of construction order, use `stop = record["stop"]` explicitly.
<!-- bilingual-en:end -->

字典默认提供键；键和值都按插入顺序迭代。因此上面的 `.values()` 写法依赖这个字典的构造顺序。如果变量无论如何都必须接住站名，直接写 `stop = record["stop"]` 更明确。

<!-- bilingual-en:start -->
A set can also be unpacked if the counts match, but it does not promise a positional order. Do not use `origin, destination = some_set` when these names must have fixed roles. A starred target has a different count rule: in `first, *rest = items`, `rest` receives the remaining items as a list, and at least one item is required here.
<!-- bilingual-en:end -->

集合在数量匹配时也能解包，但没有位置顺序保证；如果起点和终点必须有固定含义，就不应写 `origin, destination = some_set`。带星号的写法另有数量规则：`first, *rest = items` 把其余元素装成列表交给 `rest`，这里至少要有一个元素供 `first` 接收。

## 来源与核验

- [Python Language Reference — Assignment statements](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements): iterable assignment, item counts and starred targets / 解包数量和星号接收规则。
- [Python Built-in Types — Dictionary view objects](https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects): key/value iteration order / 字典键值的迭代顺序。
- [Python Tutorial — Sets](https://docs.python.org/3/tutorial/datastructures.html#sets): lack of positional order / 集合没有位置顺序保证。
