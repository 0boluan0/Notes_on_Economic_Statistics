---
aliases:
  - MIT 6.100L Lecture 11
  - 6.100L L11
  - Aliasing and Cloning
tags:
  - computer-science
  - python
  - mit-6.100l
  - lecture-note
科目: Computer Science
course: MIT 6.100L Introduction to CS and Programming Using Python
lecture: 11
---

# Lecture 11: Aliasing and Cloning

> [!tip] Hint
> - 这节课不是离开 mutability，而是专门处理 mutability 最容易出事故的地方：删除元素。
> - 老师一上来先讲 copy，不是跑题，而是因为“删元素时遍历原列表”非常容易漏掉元素。
> - `L[:]` 这节课第一次被正式解释成 clone，而不是普通切片小技巧。
> - `remove`、`del`、`pop` 看起来都在删东西，但参数和返回值不一样，副作用也要分清。
> - `remove_all` 的错误写法是在说明：一边遍历一边删，循环看到的世界会变。
> - `L1_copy = L1` 不是 copy，而是 alias，这个坑是本讲的中心。
> - `hot = warm`、`chill = cool[:]` 这种短例子其实在训练你画内存图。
> - `sort()` vs `sorted()` 再次出现，是为了把 aliasing 和 mutation 一起看。
> - shallow copy 只复制最外层，deep copy 才会把嵌套结构也复制开。
> - 听完这节课，你应该能解释“为什么我明明只改了一个名字，另一个名字也变了”。
> <!-- bilingual-en:start -->
> - This lecture does not leave mutability behind. It focuses on one of its most failure-prone operations: deleting elements.
> - Copying appears at the start for a reason: iterating over a list while deleting from that same list can easily skip elements.
> - `L[:]` is formally introduced as a way to clone a list, not merely as a slicing trick.
> - `remove`, `del`, and `pop` all delete something, but they take different arguments, return different values, and have distinct side effects.
> - The incorrect version of `remove_all` demonstrates that the sequence seen by a loop changes when the loop deletes from it.
> - `L1_copy = L1` creates an alias, not a copy. That distinction is the center of the lecture.
> - Small examples such as `hot = warm` and `chill = cool[:]` train you to draw object-reference diagrams.
> - The contrast between `sort()` and `sorted()` returns because sorting makes mutation and aliasing interact visibly.
> - A shallow copy duplicates only the outer container; a deep copy separates the nested structure as well.
> - By the end, you should be able to explain why changing an object through one name can make it appear to change through another.
> <!-- bilingual-en:end -->

## Lecture flow

### 1. 这节课从“如何安全地删元素”开始
<!-- bilingual-en:start -->
*1. Beginning with the Safe Removal of Elements*
<!-- bilingual-en:end -->
Lecture 10 已经让你意识到 list 是 mutable。  
Lecture 11 则把问题推进一步：

- 如果我想删除元素怎么办
- 删除时为什么经常出 bug
- 我什么时候需要复制列表而不是直接改原列表
<!-- bilingual-en:start -->
Lecture 10 established that lists are mutable. Lecture 11 asks what follows: how elements should be removed, why deletion often causes bugs, and when a list should be copied rather than changed directly.
<!-- bilingual-en:end -->

所以这节课开头先讲 copy，不是偏题，而是因为一旦要 mutate 列表，就要开始考虑：

- 我现在改的是谁
- 我遍历的是谁
- 改动会不会反过来影响正在遍历的结构
<!-- bilingual-en:start -->
Copying therefore belongs at the beginning. Before mutating a list, you must know which object is being changed, which object is being traversed, and whether the change alters the structure currently driving the loop.
<!-- bilingual-en:end -->

### 2. 先讲 clone：`L[:]` 为什么重要
<!-- bilingual-en:start -->
*2. Starting with Cloning: Why `L[:]` Matters*
<!-- bilingual-en:end -->
老师最先正式介绍的是“复制整个列表”。

写法是：
<!-- bilingual-en:start -->
The instructor first introduces a full-list copy:
<!-- bilingual-en:end -->

```python
Lcopy = L[:]
```

课堂对这行代码的解释很直接：

- Python 在内存里创建一个新的 list object
- 把原列表顶层元素逐个复制进去
- 新旧列表此时是两个不同对象
<!-- bilingual-en:start -->
Python creates a new list object in memory, copies the original list's top-level elements into it, and leaves the old and new lists as distinct objects.
<!-- bilingual-en:end -->

这一点很关键，因为后面你会不断在两个策略之间切换：

- 直接改原列表
- 先复制一份，再在副本上遍历或修改
<!-- bilingual-en:start -->
This distinction supports two recurring strategies: mutate the original directly, or make a copy and traverse or modify the copy instead.
<!-- bilingual-en:end -->

> [!note]
> `L[:]` 在这节课里的身份不是“切片技巧”，而是最常用的 clone 写法。
> <!-- bilingual-en:start -->
> In this lecture, `L[:]` is not merely a slicing trick; it is the standard list-cloning idiom.
> <!-- bilingual-en:end -->

### 3. 删除操作先分清：`remove`、`del`、`pop`
<!-- bilingual-en:start -->
*3. Distinguishing the Deletion Operations: `remove`, `del`, and `pop`*
<!-- bilingual-en:end -->
老师接着把常见删除接口挨个拿出来。

它们都能“删掉东西”，但侧重点不同：

- `L.remove(e)`：按值删除，删除第一次出现的 `e`
- `del(L[i])`：按索引删除，不返回值
- `L.pop()` / `L.pop(i)`：按索引删除，并把删掉的元素返回
<!-- bilingual-en:start -->
The instructor compares the common deletion interfaces. `L.remove(e)` removes the first occurrence of a value; `del(L[i])` deletes by index and returns nothing; `L.pop()` or `L.pop(i)` deletes by index and returns the removed element.
<!-- bilingual-en:end -->

这一段看似 API 介绍，真正目的是训练你不要把所有删除操作混成一团。
<!-- bilingual-en:start -->
The purpose is not merely to catalogue APIs, but to keep distinct operations conceptually separate.
<!-- bilingual-en:end -->

例如：

- 你已经知道要删哪个值，适合 `remove`
- 你已经知道要删哪个位置，适合 `del` 或 `pop`
- 你还想保留被删掉的那个元素，才需要 `pop`
<!-- bilingual-en:start -->
Use `remove` when the value is known, `del` or `pop` when the position is known, and `pop` specifically when the removed value is also needed.
<!-- bilingual-en:end -->

### 4. `remove_all(L, e)`：为什么一边遍历一边删会错
<!-- bilingual-en:start -->
*4. `remove_all(L, e)`: Why Deleting While Iterating Goes Wrong*
<!-- bilingual-en:end -->
课程第一个关键例子是 `remove_all(L, e)`。

题目要求：

- 原地修改 `L`
- 去掉所有等于 `e` 的元素
- 不返回新列表
<!-- bilingual-en:start -->
The first key example, `remove_all(L, e)`, must mutate `L` in place, remove every element equal to `e`, and return no new list.
<!-- bilingual-en:end -->

错误写法通常像这样：
<!-- bilingual-en:start -->
A typical incorrect implementation is:
<!-- bilingual-en:end -->

```python
for elem in L:
    if elem == e:
        L.remove(e)
```

它的问题不是 `remove` 本身，而是：

- `for elem in L` 正在按当前列表状态前进
- 你又在循环内部改变这个列表
- 元素位置一旦左移，后续某些元素就可能被跳过
<!-- bilingual-en:start -->
The fault is not `remove` itself. The `for` loop advances through the current list while the loop body changes that same list. When deletion shifts later elements to the left, the iteration can skip them.
<!-- bilingual-en:end -->

老师在这里反复强调的不是“这个题怎么写”，而是一个一般规律：

> [!warning]
> 对 mutable sequence 来说，遍历和修改如果发生在同一对象上，循环的行为必须重新分析。
> <!-- bilingual-en:start -->
> Whenever traversal and mutation operate on the same mutable sequence, reanalyze what the loop will actually visit.
> <!-- bilingual-en:end -->

### 5. 更稳的写法：要么反复删，要么遍历副本
<!-- bilingual-en:start -->
*5. Safer Patterns: Repeated Deletion or Traversal over a Copy*
<!-- bilingual-en:end -->
课堂给出的比较安全的方式包括：
<!-- bilingual-en:start -->
One safer classroom pattern is:
<!-- bilingual-en:end -->

```python
while e in L:
    L.remove(e)
```

这时你没有一边 `for` 一边扫描一边删除，而是每轮重新检查当前列表状态。
<!-- bilingual-en:start -->
This loop rechecks the current list state on every iteration instead of advancing a `for` iterator while deleting.
<!-- bilingual-en:end -->

另一个思路则出现在后面的 `remove_dups(L1, L2)` 中：

- 遍历副本
- 修改原列表

这就需要 clone。
<!-- bilingual-en:start -->
The later `remove_dups(L1, L2)` example uses the other strategy: traverse a copy while mutating the original. That requires an actual clone.
<!-- bilingual-en:end -->

### 6. `remove_dups(L1, L2)`：别把 alias 当 copy
<!-- bilingual-en:start -->
*6. `remove_dups(L1, L2)`: Do Not Mistake an Alias for a Copy*
<!-- bilingual-en:end -->
老师随后抛出一个更能暴露问题的例子：  
如果 `L1` 中某个元素也在 `L2` 中，就把它从 `L1` 删除。
<!-- bilingual-en:start -->
The next example makes the distinction sharper: remove from `L1` every element that also appears in `L2`.
<!-- bilingual-en:end -->

错误版本里最经典的一行是：

```python
L1_copy = L1
```

这看起来像“复制了一份”，但实际上并没有。  
它只是让 `L1_copy` 成为同一个对象的另一个名字，也就是 **alias**。
<!-- bilingual-en:start -->
The line looks like a copy, but it only gives the same object another name. `L1_copy` is an **alias**.
<!-- bilingual-en:end -->

于是：

- 你以为自己在遍历副本
- 实际上你还是在遍历被修改的那个对象
<!-- bilingual-en:start -->
You may think you are traversing a copy, but you are still traversing the very object being modified.
<!-- bilingual-en:end -->

正确写法才是：
<!-- bilingual-en:start -->
An actual clone is required:
<!-- bilingual-en:end -->

```python
L1_copy = L1[:]
for e in L1_copy:
    if e in L2:
        L1.remove(e)
```

### 7. aliasing：两个名字指向同一个对象
<!-- bilingual-en:start -->
*7. Aliasing: Two Names Refer to the Same Object*
<!-- bilingual-en:end -->
到这里老师才正式把术语说出来：**aliasing**。

最典型例子是：
<!-- bilingual-en:start -->
The instructor now names the phenomenon **aliasing**, illustrated by:
<!-- bilingual-en:end -->

```python
warm = ['red', 'yellow', 'orange']
hot = warm
hot.append('pink')
```

如果你只从变量名表面看，会以为：

- `hot` 变了
- `warm` 不该变
<!-- bilingual-en:start -->
Looking only at the variable names might suggest that `hot` changes while `warm` does not.
<!-- bilingual-en:end -->

但课堂要你建立的对象视角是：

- `warm` 和 `hot` 指向同一个 list object
- append 改的是那个对象
- 所以两个名字看到的内容都会变
<!-- bilingual-en:start -->
The object model says otherwise: `warm` and `hot` refer to the same list object, and `append` mutates that object. Both names therefore reveal the updated contents.
<!-- bilingual-en:end -->

这就是“为什么我只改了一个变量，另一个变量也变了”的根源。
<!-- bilingual-en:start -->
That shared reference explains why a change made “through one variable” appears through the other as well.
<!-- bilingual-en:end -->

### 8. cloning：创建另一个独立对象
<!-- bilingual-en:start -->
*8. Cloning: Creating a Separate Object*
<!-- bilingual-en:end -->
为了和 aliasing 形成清楚对照，老师马上给出 cloning 例子：
<!-- bilingual-en:start -->
The instructor immediately contrasts aliasing with cloning:
<!-- bilingual-en:end -->

```python
cool = ['blue', 'green', 'grey']
chill = cool[:]
chill.append('black')
```

这时：

- `chill` 变了
- `cool` 不变
<!-- bilingual-en:start -->
Here `chill` changes while `cool` does not.
<!-- bilingual-en:end -->

因为这里不是两个名字指向同一个对象，而是先创建了一个新对象，再让 `chill` 指向它。
<!-- bilingual-en:start -->
The slice created a new object for `chill` rather than giving the original object a second name.
<!-- bilingual-en:end -->

> [!example]
> aliasing 和 cloning 的区别，表面上只差一个 `[:]`，但语义上是“共享对象”和“复制对象”的分界线。
> <!-- bilingual-en:start -->
> Aliasing and cloning may differ in the code only by `[:]`, but semantically that marks the boundary between sharing an object and copying it.
> <!-- bilingual-en:end -->

### 9. 再回头看 `sort()`：为什么它总和 aliasing 一起出问题
<!-- bilingual-en:start -->
*9. Returning to `sort()`: Why Sorting Interacts with Aliasing*
<!-- bilingual-en:end -->
老师接着用排序例子回收前两讲内容：
<!-- bilingual-en:start -->
The instructor reconnects the previous two lectures through sorting:
<!-- bilingual-en:end -->

```python
sortedwarm = warm.sort()
sortedcool = sorted(cool)
```

这两行一起看时，你会同时遇到两个坑：

- `sort()` 是原地修改
- 它返回 `None`
<!-- bilingual-en:start -->
The pair exposes two facts at once: `sort()` mutates in place, and it returns `None`.
<!-- bilingual-en:end -->

如果一个列表还有 alias，那么这个原地修改会沿着 alias 一起暴露出来。  
所以这节课里 `sort()` 再次出现，不只是复习 API，而是在把 mutation、aliasing、return value 三件事绑到一起看。
<!-- bilingual-en:start -->
If a list has aliases, that in-place mutation becomes visible through every alias. The example therefore unifies mutation, aliasing, and return-value semantics rather than merely reviewing an API.
<!-- bilingual-en:end -->

### 10. lists of lists：顶层对象和内层对象要分开想
<!-- bilingual-en:start -->
*10. Lists of Lists: Distinguishing Outer and Inner Objects*
<!-- bilingual-en:end -->
课程随后故意给出嵌套列表，比如：
<!-- bilingual-en:start -->
The lecture then introduces a nested list deliberately:
<!-- bilingual-en:end -->

```python
warm = ['yellow', 'orange']
hot = ['red']
brightcolors = [warm]
brightcolors.append(hot)
hot.append('pink')
```

这里最容易错的地方是没有分层思考：

- 顶层 list 是一个对象
- 其中每个子 list 也是对象
<!-- bilingual-en:start -->
The easy mistake is to ignore the levels: the outer list is one object, and each nested list is another object.
<!-- bilingual-en:end -->

所以当你修改 `hot` 时，`brightcolors` 里看到的那个子列表也会变，因为它保存的正是 `hot` 这个对象的引用。
<!-- bilingual-en:start -->
Mutating `hot` therefore changes the sublist observed through `brightcolors`, because the outer list stores a reference to the same `hot` object.
<!-- bilingual-en:end -->

这为后面的 shallow copy / deep copy 做了铺垫。
<!-- bilingual-en:start -->
This prepares the distinction between shallow and deep copies.
<!-- bilingual-en:end -->

### 11. shallow copy：只复制最外层壳
<!-- bilingual-en:start -->
*11. Shallow Copies: Duplicating Only the Outer Container*
<!-- bilingual-en:end -->
老师后半段引入 `copy.copy(old_list)`，并且强调：

- 它会创建一个新的顶层 list
- 但里面嵌套的子对象仍然共享
<!-- bilingual-en:start -->
`copy.copy(old_list)` creates a new outer list, but the nested objects inside it remain shared.
<!-- bilingual-en:end -->

所以如果：

- 你在顶层 append 一个新子列表
- 旧副本可能看不到
<!-- bilingual-en:start -->
Appending a new sublist at the outer level may therefore affect only one copy.
<!-- bilingual-en:end -->

但如果：

- 你修改某个共享子列表里的元素
- 新旧两个结构都可能一起变
<!-- bilingual-en:start -->
Mutating an element inside a shared sublist, however, may become visible in both structures.
<!-- bilingual-en:end -->

课堂这里真正想让你掌握的是“复制到了哪一层”。
<!-- bilingual-en:start -->
The real question is how many levels of the object graph have been copied.
<!-- bilingual-en:end -->

### 12. deep copy：连嵌套层级一起断开
<!-- bilingual-en:start -->
*12. Deep Copies: Separating the Nested Levels as Well*
<!-- bilingual-en:end -->
为了解决上面的共享问题，老师再引入：
<!-- bilingual-en:start -->
To eliminate that nested sharing, the instructor introduces:
<!-- bilingual-en:end -->

```python
copy.deepcopy(old_list)
```

这意味着：

- 顶层复制
- 内层也复制
- 再深一层也继续复制
<!-- bilingual-en:start -->
A deep copy duplicates the outer level and recursively duplicates the inner levels.
<!-- bilingual-en:end -->

所以之后你修改原结构里的嵌套元素，深拷贝出来的版本不会一起变。
<!-- bilingual-en:start -->
Later mutations of nested elements in the original structure therefore do not alter the deep-copied version.
<!-- bilingual-en:end -->

这部分不要求你死记库函数，而是要求你在看到嵌套结构时立刻问自己：

- 我需要的只是顶层独立吗
- 还是整个结构都要独立
<!-- bilingual-en:start -->
The point is not to memorize a library function, but to ask whether only the outer container must be independent or the entire nested structure must be independent.
<!-- bilingual-en:end -->

### 13. 这节课真正教的是“对象图”，不是列表技巧
<!-- bilingual-en:start -->
*13. The Real Subject Is the Object Graph, Not a Collection of List Tricks*
<!-- bilingual-en:end -->
Lecture 11 如果只记成“学了 copy / deepcopy / remove / pop”，会低估它的重要性。
<!-- bilingual-en:start -->
Remembering Lecture 11 only as a catalogue of `copy`, `deepcopy`, `remove`, and `pop` understates its importance.
<!-- bilingual-en:end -->

这节课实际在训练你把程序状态看成一个对象图：

- 名字指向对象
- 对象里还可能含有别的对象
- mutation 是对对象发生，不是对名字发生
- aliasing 会让多个名字共享一个对象
<!-- bilingual-en:start -->
The lecture teaches you to see program state as an object graph: names refer to objects, objects may contain references to other objects, mutation happens to objects rather than to names, and aliases let several names share one object.
<!-- bilingual-en:end -->

后面类、继承、复杂数据结构，全都建立在这种对象视角之上。
<!-- bilingual-en:start -->
Classes, inheritance, and more complex data structures all build on this object-centered view.
<!-- bilingual-en:end -->

## Exercise log

> [!example] Finger exercise 11
> 官方练习是 `remove_and_sort(Lin, k)`：
> - 原地删除前 `k` 个元素
> - 再把剩余元素按升序排序
> - 不返回任何值
> <!-- bilingual-en:start -->
> The official exercise is `remove_and_sort(Lin, k)`:
> - Delete the first `k` elements in place.
> - Sort the remaining elements in ascending order.
> - Return no value.
> <!-- bilingual-en:end -->

这题非常适合放在本讲末尾，因为它把本讲的三个关键词揉在一起了：

- mutation
- 删除操作
- 原地排序
<!-- bilingual-en:start -->
The exercise combines three themes from the lecture: mutation, deletion, and in-place sorting.
<!-- bilingual-en:end -->

官方解法先处理边界情况：
<!-- bilingual-en:start -->
The official solution first handles the boundary case:
<!-- bilingual-en:end -->

```python
if len(Lin) <= k:
    Lin.clear()
    return
```

然后再用 `del(Lin[0])` 连续删前面元素，最后 `Lin.sort()`。
<!-- bilingual-en:start -->
It then repeatedly deletes the first element with `del(Lin[0])` and finishes with `Lin.sort()`.
<!-- bilingual-en:end -->

这题的价值不只是“会不会删前 k 个元素”，而是你能不能稳定地区分：

- 修改列表本身
- 返回新列表
- `sort()` 的副作用
<!-- bilingual-en:start -->
The exercise tests more than removing the first `k` elements. It asks you to distinguish mutating a list from returning a new list and to understand the side effect of `sort()`.
<!-- bilingual-en:end -->

## Links to follow-up practice
- Slides: [[MIT 6.100L-slides/mit6_100l_lec11.pdf|Lecture 11 slides]]
- Lecture code: [[MIT 6.100L-lecture-code/mit6_100l_lec11_code.py|Lecture 11 code (py)]]
- Finger exercise: [[MIT 6.100L-finger-exercises/mit6_100l_ex11_sol.pdf|Lecture 11 finger exercise solution]]
- Transcript: [[MIT 6.100L-transcripts/mit6_100l_lec11_transcript.pdf|Lecture 11 transcript]]
- Recitation 6: [[MIT 6.100L-recitations/mit6_100l_rec06.zip|Recitation 06 materials]]
- Problem set milestone: none directly scheduled on this lecture
- Textbook: [[Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf|Guttag textbook]] (Ch 5.3-5.5)

## Review checklist
- [ ] 我能解释为什么 `L[:]` 在这节课里代表 clone。
- [ ] 我能区分 `remove`、`del`、`pop` 的适用场景和返回值。
- [ ] 我能说明为什么一边 `for` 遍历列表一边 `remove` 会漏元素。
- [ ] 我能解释 `L1_copy = L1` 为什么只是 alias，不是 copy。
- [ ] 我能画出 aliasing 和 cloning 的简单内存图。
- [ ] 我能解释 shallow copy 和 deep copy 的差别。
- [ ] 我能说明嵌套列表里“顶层复制”和“子对象共享”为什么是两回事。
- [ ] 我能把 `sort()`、`sorted()`、`clear()` 这些原地/非原地操作联系到对象视角上。
- [ ] 我能解释 finger exercise 11 为什么本质上在考 mutation 风格的 list 操作。
- [ ] 我能按课堂顺序复述：copy -> remove pitfalls -> aliasing -> cloning -> shallow/deep copy。
<!-- bilingual-en:start -->
- [ ] I can explain why `L[:]` represents cloning in this lecture.
- [ ] I can distinguish the appropriate uses and return values of `remove`, `del`, and `pop`.
- [ ] I can explain why removing elements from a list during a `for` loop may skip elements.
- [ ] I can explain why `L1_copy = L1` creates an alias rather than a copy.
- [ ] I can draw simple object-reference diagrams for aliasing and cloning.
- [ ] I can distinguish a shallow copy from a deep copy.
- [ ] I can explain why copying an outer list and sharing its child objects are separate issues.
- [ ] I can connect the in-place or non-mutating behavior of `sort()`, `sorted()`, and `clear()` to the object model.
- [ ] I can explain why finger exercise 11 tests mutation-oriented list operations.
- [ ] I can reconstruct the lecture sequence: copying -> deletion pitfalls -> aliasing -> cloning -> shallow and deep copies.
<!-- bilingual-en:end -->

> [!warning] Common mistakes
> - 把 `L1_copy = L1` 当成复制。
> - 在遍历列表的同时删除这个列表里的元素。
> - 只记 shallow copy / deep copy 名字，不去想“到底复制到哪一层”。
> - 看到变量名不同，就误以为对象也一定不同。
> <!-- bilingual-en:start -->
> - Treating `L1_copy = L1` as a copy.
> - Deleting elements from a list while iterating over that same list.
> - Memorizing the terms “shallow copy” and “deep copy” without asking which levels were actually duplicated.
> - Assuming that different variable names must refer to different objects.
> <!-- bilingual-en:end -->
