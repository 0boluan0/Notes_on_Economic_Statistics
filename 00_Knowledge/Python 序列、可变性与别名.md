---
aliases:
  - "Python Sequences, Mutability and Aliasing"
  - "Python序列"
status: source-checked
---

# Python 序列、可变性与别名
<!-- bilingual-en:start -->
*Python Sequences, Mutability, and Aliasing*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 准确判断 tuple、list、string 和 dictionary 中哪些对象可变、哪些名称共享同一对象，以及修改会传播到哪里。
> **具体锚点：** `b = a` 不复制列表；`b.append(...)` 后 a 也看到变化，因为两个名称指向同一对象。
> **核心难点：** 容器不可变不等于其中对象不可变；浅拷贝只复制外层，嵌套元素仍可共享。
> **为什么重要：** 大量“莫名被修改”的 bug、默认参数陷阱和函数副作用都来自别名。
> **继续：** 画对象图而非把变量想成盒子；修改前决定共享、浅拷贝还是深拷贝。
> <!-- bilingual-en:start -->
> **Problem addressed:** Determine which tuple, list, string, and dictionary objects are mutable, which names share one object, and where a mutation will be observed.
> **Concrete anchor:** `b = a` does not copy a list; after `b.append(...)`, `a` also observes the change because both names refer to the same object.
> **Central difficulty:** An immutable container can still contain mutable objects, and a shallow copy duplicates only the outer container while nested elements remain shared.
> **Why it matters:** Many apparently mysterious mutations, default-argument bugs, and function side effects arise from aliasing.
> **Continue with:** Draw an object graph rather than imagining variables as boxes; decide deliberately among sharing, shallow copying, and deep copying before mutation.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
> <!-- bilingual-en:start -->
> - Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
> - [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
> <!-- bilingual-en:end -->

## 序列共同接口
<!-- bilingual-en:start -->
*The Common Sequence Interface*
<!-- bilingual-en:end -->

string、tuple、list 支持长度、索引、切片、迭代和成员测试，但元素/对象可变性不同。切片通常创建新外层序列；负索引和半开区间使边界组合方便。
<!-- bilingual-en:start -->
Strings, tuples, and lists support length, indexing, slicing, iteration, and membership tests, although their mutability differs. A slice normally creates a new outer sequence; negative indexes and half-open intervals make boundaries composable.
<!-- bilingual-en:end -->

## 可变与不可变
<!-- bilingual-en:start -->
*Mutable and Immutable Objects*
<!-- bilingual-en:end -->

字符串、tuple、数字不可原地改，操作返回新对象；list/dict/set 可原地修改。tuple 不可变只指其元素引用槽不能换，若元素是 list，内部 list 仍可变。
<!-- bilingual-en:start -->
Strings, tuples, and numbers cannot be changed in place, so operations produce new objects; lists, dictionaries, and sets can be mutated. Tuple immutability fixes its element-reference slots, but a list referenced by a tuple can still change internally.
<!-- bilingual-en:end -->

判断副作用时分别问：名称是否重新绑定、容器槽是否改变、槽指向的对象是否改变。这三种变化在嵌套结构中会产生不同的可见范围。
<!-- bilingual-en:start -->
When reasoning about side effects, ask separately whether a name was rebound, a container slot changed, or the object referenced by that slot mutated. These three changes have different visibility in nested structures.
<!-- bilingual-en:end -->

## aliasing
<!-- bilingual-en:start -->
*Aliasing*
<!-- bilingual-en:end -->

`b=a` 创建另一个名称绑定同一对象。身份用 `is` 判断，值相等用 `==`。函数收到 list 后原地修改会被调用者观察；是否允许应写在 contract。
<!-- bilingual-en:start -->
`b=a` creates another name for the same object. Use `is` for identity and `==` for value equality. If a function mutates a list argument in place, its caller observes the change; the contract should say whether this is allowed.
<!-- bilingual-en:end -->

## cloning
<!-- bilingual-en:start -->
*Cloning*
<!-- bilingual-en:end -->

`a[:]`、`list(a)` 或 `copy.copy` 生成浅拷贝，嵌套对象仍共享。`copy.deepcopy` 递归复制但成本高、对外部资源/自定义对象未必合适。最小需要的复制层级最好。
<!-- bilingual-en:start -->
`a[:]`, `list(a)`, and `copy.copy` create a shallow copy, leaving nested objects shared. `copy.deepcopy` copies recursively but is expensive and may be inappropriate for external resources or custom objects. Copy only as deeply as the contract requires.
<!-- bilingual-en:end -->

## Worked example：浅拷贝为何仍会联动
<!-- bilingual-en:start -->
*Worked Example: Why a Shallow Copy Still Shares Changes*
<!-- bilingual-en:end -->

下面两个外层列表身份不同，但都引用同一个内部列表。追加外层元素只影响副本；修改内部列表则两边都可见。
<!-- bilingual-en:start -->
The two outer lists below have different identities, but both refer to the same inner list. Appending to the outer copy affects only that copy; mutating the inner list is visible through both outer lists.
<!-- bilingual-en:end -->

```python
original = [[1, 2], [3, 4]]
shallow = original[:]

shallow.append([5, 6])
shallow[0].append(99)

print(original)  # [[1, 2, 99], [3, 4]]
print(shallow)   # [[1, 2, 99], [3, 4], [5, 6]]
```

对象图中应画两个外层节点、两个共享的内部节点和一个仅由 `shallow` 引用的新节点。仅比较打印值不容易看出这种共享结构。
<!-- bilingual-en:start -->
The object graph should contain two outer nodes, two shared inner nodes, and one new node referenced only by `shallow`. Comparing printed values alone makes this sharing structure easy to miss.
<!-- bilingual-en:end -->

## list comprehension
<!-- bilingual-en:start -->
*List Comprehensions*
<!-- bilingual-en:end -->

列表推导表达“遍历—过滤—映射”，适合无副作用构造。复杂嵌套或副作用用显式循环更清楚。生成器表达式惰性产生值，节省内存但只能按迭代消费。
<!-- bilingual-en:start -->
A list comprehension expresses traversal, filtering, and mapping, and works best for side-effect-free construction. Use an explicit loop for complex nesting or side effects. A generator expression produces values lazily, saving memory but yielding them only through iteration.
<!-- bilingual-en:end -->

## dictionary 与哈希
<!-- bilingual-en:start -->
*Dictionaries and Hashing*
<!-- bilingual-en:end -->

dict 把可哈希 key 映射到 value，平均查找近 O(1)。可哈希对象需哈希稳定且相等对象哈希相同，所以 list 不能作 key；tuple 仅在全部元素可哈希时可作 key。
<!-- bilingual-en:start -->
A dictionary maps hashable keys to values and provides average lookup near O(1). A hashable object needs a stable hash, and equal objects must have equal hashes, so a list cannot be a key; a tuple is hashable only when all of its elements are hashable.
<!-- bilingual-en:end -->

这里只关心 Python 对象能否安全作为 key；碰撞、负载因子和扩容机制集中在 [[哈希表]]，避免维护两份解释。
<!-- bilingual-en:start -->
This note asks only whether a Python object can safely serve as a key. Collision handling, load factors, and resizing live in [[哈希表|Hash Tables]] so that those mechanisms are not explained twice.
<!-- bilingual-en:end -->

## 安全迭代
<!-- bilingual-en:start -->
*Safe Iteration*
<!-- bilingual-en:end -->

遍历容器时结构修改可能跳过元素或报错；构造新容器、遍历副本或分两阶段修改。
<!-- bilingual-en:start -->
Structural mutation during iteration can skip elements or raise an error. Build a new container, iterate over a copy, or separate discovery and mutation into two phases.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 一个列表“自己变了”：搜索所有持有同一对象的别名和所有原地方法，如 `append`、`extend`、切片赋值与 `sort`。
  <!-- bilingual-en:start -->
  A list “changed by itself”: find every alias to the same object and every in-place operation, including `append`, `extend`, slice assignment, and `sort`.
  <!-- bilingual-en:end -->
- 拷贝后仍联动：逐层检查身份，确定共享发生在外层还是嵌套元素。
  <!-- bilingual-en:start -->
  Copies still change together: compare identity at each level to locate whether sharing occurs in the outer container or nested elements.
  <!-- bilingual-en:end -->
- dict key 报错：检查 key 是否可变，或自定义对象的 `__eq__` 与 `__hash__` 是否一致。
  <!-- bilingual-en:start -->
  A dictionary key fails: check whether the key is mutable and whether a custom object's `__eq__` and `__hash__` agree.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### `b=a` 和 `b=a[:]` 对列表有何差别？
<!-- bilingual-en:start -->
*What is the difference between `b=a` and `b=a[:]` for a list?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 前者共享同一列表；后者复制外层列表，但嵌套元素仍可能共享。
> <!-- bilingual-en:start -->
> The former shares one list; the latter copies the outer list, although nested elements may still be shared.
> <!-- bilingual-en:end -->

### tuple 为什么可能间接变化？
<!-- bilingual-en:start -->
*How can a tuple appear to change indirectly?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> tuple 的引用槽固定，但若槽指向可变对象，该对象内部可修改。
> <!-- bilingual-en:start -->
> A tuple's reference slots are fixed, but an object referenced by a slot can still be mutable internally.
> <!-- bilingual-en:end -->

### 为什么 list 不能作 dict key？
<!-- bilingual-en:start -->
*Why can a list not be used as a dictionary key?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> list 可变，内容变化会破坏哈希值稳定性和哈希表查找不变量。
> <!-- bilingual-en:start -->
> A list is mutable, so content changes would violate hash stability and the lookup invariant of the hash table.
> <!-- bilingual-en:end -->

### 一个函数应修改传入列表还是返回新列表，怎样决定？
<!-- bilingual-en:start -->
*How should a function decide between mutating an input list and returning a new one?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 由接口契约决定：若共享状态是明确目的，可原地修改并清楚命名；否则优先返回新对象，减少隐藏副作用。
> <!-- bilingual-en:start -->
> The interface contract decides. Mutate in place, with explicit naming, when shared state is intentional; otherwise prefer a new object to reduce hidden side effects.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- MIT 6.100L 本地官方 slides、transcripts、finger exercises 与 problem sets：支持每个概念、示例和课程范围。
  <!-- bilingual-en:start -->
  Local official MIT 6.100L slides, transcripts, finger exercises, and problem sets support the concepts, examples, and course scope.
  <!-- bilingual-en:end -->
- [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf]]：交叉核验 Python、算法、复杂度、OOP 与模拟。
  <!-- bilingual-en:start -->
  [[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Introduction to Computation and Programming Using Python]] cross-checks Python, algorithms, complexity, object-oriented programming, and simulation.
  <!-- bilingual-en:end -->
- [The Python Tutorial: Data Structures](https://docs.python.org/3/tutorial/datastructures.html)：复核序列、列表推导、tuple、set、dictionary 与遍历语义。
  <!-- bilingual-en:start -->
  [The Python Tutorial: Data Structures](https://docs.python.org/3/tutorial/datastructures.html) verifies sequences, list comprehensions, tuples, sets, dictionaries, and iteration semantics.
  <!-- bilingual-en:end -->
