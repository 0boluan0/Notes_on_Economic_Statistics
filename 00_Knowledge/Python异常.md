---
aliases:
  - "Python 异常是携带类型与上下文并转移正常控制流的对象"
  - "Python exception"
student_os: knowledge-atom
atom_id: CS-EXC-001
atom_type: definition
status: source-checked
part_of:
  - "[[测试、调试、异常与断言.canvas]]"
leads_to:
  - "[[异常传播]]"
  - "[[异常类型匹配]]"
  - "[[主动抛出异常]]"
---

# Python 异常是携带类型与上下文并转移正常控制流的对象

<!-- bilingual-en:start -->
*A Python exception is an object carrying a type and context that redirects normal control flow*
<!-- bilingual-en:end -->

Python 异常（exception）是异常类的实例；被抛出时，它中断当前正常执行路径，并把控制交给适用的异常处理机制。异常类型说明发生了哪类情况，实例可携带消息和其他信息，traceback 记录相关调用位置。它不是单纯打印出来的一行文字，也不意味着程序一定立即结束。

<!-- bilingual-en:start -->
A Python exception is an instance of an exception class. Raising it interrupts the current normal execution path and transfers control to the applicable exception-handling mechanism. Its type identifies the kind of condition, its instance can carry a message and other information, and its traceback records relevant call locations. It is neither merely printed text nor a guarantee that the program immediately stops.
<!-- bilingual-en:end -->

```python
try:
    int("cat")
except ValueError as exc:
    print(type(exc).__name__)  # ValueError
```

这里转换失败产生 `ValueError` 对象，`exc` 在处理器内引用它；打印只是处理器选择的响应。异常还可以表示预期的运行条件，而非程序设计错误。应按[[异常类型匹配|类型]]识别异常；不要把某版本的完整错误消息当作稳定接口。

<!-- bilingual-en:start -->
The failed conversion creates a `ValueError` object, referenced by `exc` inside the handler; printing is merely the chosen response. An exception can also signal an expected runtime condition rather than a programming defect. Identify it by its [[异常类型匹配|type]], not by treating a particular version's complete error message as a stable interface.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec13.pdf#page=3|MIT 6.100L Lecture 13，PDF pp. 3–4]]；[[03_Computer_Science/03_MIT 6.100L/Introduction to Computation and Programming Using Python, Revised - Guttag, John V..pdf#page=101|Guttag，印刷 pp. 84–85 / PDF pp. 101–102]]：核对运行时异常、类型名称与可处理性。
  <!-- bilingual-en:start -->
  These pages support runtime exceptions, named exception types, and the possibility of handling them.
  <!-- bilingual-en:end -->
- [Python Language Reference §4.3, Exceptions](https://docs.python.org/3/reference/executionmodel.html#exceptions)；[Built-in Exceptions](https://docs.python.org/3/library/exceptions.html)：核对异常实例、附加信息、traceback 与错误消息不属于稳定 API 的边界。
  <!-- bilingual-en:start -->
  These references support exception instances, attached information, tracebacks, and the boundary that error-message wording is not a stable API.
  <!-- bilingual-en:end -->
