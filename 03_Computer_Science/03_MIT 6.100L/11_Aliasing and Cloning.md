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

## Lecture flow

### 1. 这节课从“如何安全地删元素”开始
Lecture 10 已经让你意识到 list 是 mutable。  
Lecture 11 则把问题推进一步：

- 如果我想删除元素怎么办
- 删除时为什么经常出 bug
- 我什么时候需要复制列表而不是直接改原列表

所以这节课开头先讲 copy，不是偏题，而是因为一旦要 mutate 列表，就要开始考虑：

- 我现在改的是谁
- 我遍历的是谁
- 改动会不会反过来影响正在遍历的结构

### 2. 先讲 clone：`L[:]` 为什么重要
老师最先正式介绍的是“复制整个列表”。

写法是：

```python
Lcopy = L[:]
```

课堂对这行代码的解释很直接：

- Python 在内存里创建一个新的 list object
- 把原列表顶层元素逐个复制进去
- 新旧列表此时是两个不同对象

这一点很关键，因为后面你会不断在两个策略之间切换：

- 直接改原列表
- 先复制一份，再在副本上遍历或修改

> [!note]
> `L[:]` 在这节课里的身份不是“切片技巧”，而是最常用的 clone 写法。

### 3. 删除操作先分清：`remove`、`del`、`pop`
老师接着把常见删除接口挨个拿出来。

它们都能“删掉东西”，但侧重点不同：

- `L.remove(e)`：按值删除，删除第一次出现的 `e`
- `del(L[i])`：按索引删除，不返回值
- `L.pop()` / `L.pop(i)`：按索引删除，并把删掉的元素返回

这一段看似 API 介绍，真正目的是训练你不要把所有删除操作混成一团。

例如：

- 你已经知道要删哪个值，适合 `remove`
- 你已经知道要删哪个位置，适合 `del` 或 `pop`
- 你还想保留被删掉的那个元素，才需要 `pop`

### 4. `remove_all(L, e)`：为什么一边遍历一边删会错
课程第一个关键例子是 `remove_all(L, e)`。

题目要求：

- 原地修改 `L`
- 去掉所有等于 `e` 的元素
- 不返回新列表

错误写法通常像这样：

```python
for elem in L:
    if elem == e:
        L.remove(e)
```

它的问题不是 `remove` 本身，而是：

- `for elem in L` 正在按当前列表状态前进
- 你又在循环内部改变这个列表
- 元素位置一旦左移，后续某些元素就可能被跳过

老师在这里反复强调的不是“这个题怎么写”，而是一个一般规律：

> [!warning]
> 对 mutable sequence 来说，遍历和修改如果发生在同一对象上，循环的行为必须重新分析。

### 5. 更稳的写法：要么反复删，要么遍历副本
课堂给出的比较安全的方式包括：

```python
while e in L:
    L.remove(e)
```

这时你没有一边 `for` 一边扫描一边删除，而是每轮重新检查当前列表状态。

另一个思路则出现在后面的 `remove_dups(L1, L2)` 中：

- 遍历副本
- 修改原列表

这就需要 clone。

### 6. `remove_dups(L1, L2)`：别把 alias 当 copy
老师随后抛出一个更能暴露问题的例子：  
如果 `L1` 中某个元素也在 `L2` 中，就把它从 `L1` 删除。

错误版本里最经典的一行是：

```python
L1_copy = L1
```

这看起来像“复制了一份”，但实际上并没有。  
它只是让 `L1_copy` 成为同一个对象的另一个名字，也就是 **alias**。

于是：

- 你以为自己在遍历副本
- 实际上你还是在遍历被修改的那个对象

正确写法才是：

```python
L1_copy = L1[:]
for e in L1_copy:
    if e in L2:
        L1.remove(e)
```

### 7. aliasing：两个名字指向同一个对象
到这里老师才正式把术语说出来：**aliasing**。

最典型例子是：

```python
warm = ['red', 'yellow', 'orange']
hot = warm
hot.append('pink')
```

如果你只从变量名表面看，会以为：

- `hot` 变了
- `warm` 不该变

但课堂要你建立的对象视角是：

- `warm` 和 `hot` 指向同一个 list object
- append 改的是那个对象
- 所以两个名字看到的内容都会变

这就是“为什么我只改了一个变量，另一个变量也变了”的根源。

### 8. cloning：创建另一个独立对象
为了和 aliasing 形成清楚对照，老师马上给出 cloning 例子：

```python
cool = ['blue', 'green', 'grey']
chill = cool[:]
chill.append('black')
```

这时：

- `chill` 变了
- `cool` 不变

因为这里不是两个名字指向同一个对象，而是先创建了一个新对象，再让 `chill` 指向它。

> [!example]
> aliasing 和 cloning 的区别，表面上只差一个 `[:]`，但语义上是“共享对象”和“复制对象”的分界线。

### 9. 再回头看 `sort()`：为什么它总和 aliasing 一起出问题
老师接着用排序例子回收前两讲内容：

```python
sortedwarm = warm.sort()
sortedcool = sorted(cool)
```

这两行一起看时，你会同时遇到两个坑：

- `sort()` 是原地修改
- 它返回 `None`

如果一个列表还有 alias，那么这个原地修改会沿着 alias 一起暴露出来。  
所以这节课里 `sort()` 再次出现，不只是复习 API，而是在把 mutation、aliasing、return value 三件事绑到一起看。

### 10. lists of lists：顶层对象和内层对象要分开想
课程随后故意给出嵌套列表，比如：

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

所以当你修改 `hot` 时，`brightcolors` 里看到的那个子列表也会变，因为它保存的正是 `hot` 这个对象的引用。

这为后面的 shallow copy / deep copy 做了铺垫。

### 11. shallow copy：只复制最外层壳
老师后半段引入 `copy.copy(old_list)`，并且强调：

- 它会创建一个新的顶层 list
- 但里面嵌套的子对象仍然共享

所以如果：

- 你在顶层 append 一个新子列表
- 旧副本可能看不到

但如果：

- 你修改某个共享子列表里的元素
- 新旧两个结构都可能一起变

课堂这里真正想让你掌握的是“复制到了哪一层”。

### 12. deep copy：连嵌套层级一起断开
为了解决上面的共享问题，老师再引入：

```python
copy.deepcopy(old_list)
```

这意味着：

- 顶层复制
- 内层也复制
- 再深一层也继续复制

所以之后你修改原结构里的嵌套元素，深拷贝出来的版本不会一起变。

这部分不要求你死记库函数，而是要求你在看到嵌套结构时立刻问自己：

- 我需要的只是顶层独立吗
- 还是整个结构都要独立

### 13. 这节课真正教的是“对象图”，不是列表技巧
Lecture 11 如果只记成“学了 copy / deepcopy / remove / pop”，会低估它的重要性。

这节课实际在训练你把程序状态看成一个对象图：

- 名字指向对象
- 对象里还可能含有别的对象
- mutation 是对对象发生，不是对名字发生
- aliasing 会让多个名字共享一个对象

后面类、继承、复杂数据结构，全都建立在这种对象视角之上。

## Exercise log

> [!example] Finger exercise 11
> 官方练习是 `remove_and_sort(Lin, k)`：
> - 原地删除前 `k` 个元素
> - 再把剩余元素按升序排序
> - 不返回任何值

这题非常适合放在本讲末尾，因为它把本讲的三个关键词揉在一起了：

- mutation
- 删除操作
- 原地排序

官方解法先处理边界情况：

```python
if len(Lin) <= k:
    Lin.clear()
    return
```

然后再用 `del(Lin[0])` 连续删前面元素，最后 `Lin.sort()`。

这题的价值不只是“会不会删前 k 个元素”，而是你能不能稳定地区分：

- 修改列表本身
- 返回新列表
- `sort()` 的副作用

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

> [!warning] Common mistakes
> - 把 `L1_copy = L1` 当成复制。
> - 在遍历列表的同时删除这个列表里的元素。
> - 只记 shallow copy / deep copy 名字，不去想“到底复制到哪一层”。
> - 看到变量名不同，就误以为对象也一定不同。
