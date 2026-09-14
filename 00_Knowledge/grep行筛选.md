---
aliases:
  - "grep行筛选按模式选择输入行并默认输出完整选中行"
  - "Line selection with grep"
student_os: knowledge-atom
atom_id: CS-TEXT-005
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[正则表达式]]"
leads_to:
  - "[[grep退出状态]]"
related:
  - "[[正则方言边界]]"
---

# grep行筛选按模式选择输入行并默认输出完整选中行

<!-- bilingual-en:start -->
*grep selects input lines by a pattern and normally outputs each complete selected line*
<!-- bilingual-en:end -->

`grep` 的常规文本用法是逐行检查输入，选择含有匹配内容的行，默认输出整行，而不是只输出匹配到的那段字符。未指定输入文件时，它读取标准输入；因此可以接收上一条命令的输出。

<!-- bilingual-en:start -->
In ordinary text processing, `grep` examines input line by line, selects lines containing a match, and normally outputs the complete line rather than just the matching substring. With no input file specified, it reads standard input and can therefore consume a preceding command's output.
<!-- bilingual-en:end -->

```bash
printf '%s\n' 'INFO ready' 'ERROR disk full' 'ERROR ERROR' | grep -F 'ERROR'
```

输出最后两行。常用选择应按任务区分：`-F` 搜索固定字符串，`-E` 采用扩展正则，`-x` 要求整行匹配，`-v` 反选不匹配的行，`-c` 统计选中行数。因此上述输入使用 `grep -c -F 'ERROR'` 得到 `2`，不是字符片段 `ERROR` 的出现次数 `3`。

<!-- bilingual-en:start -->
The last two lines are output. Choose options according to the task: `-F` searches fixed strings, `-E` selects extended regular expressions, `-x` requires a whole-line match, `-v` selects nonmatching lines, and `-c` counts selected lines. Thus `grep -c -F 'ERROR'` on this input yields `2`, not the `3` occurrences of the substring `ERROR`.
<!-- bilingual-en:end -->

模式由 Shell 作为参数传入，通常需要引用；用 `-e 'pattern'` 可以明确模式参数，尤其避免以 `-` 开头的模式被当作选项。无输出可能是没有选中行，也可能是错误，必须查看[[grep退出状态]]。二进制输入、跨行搜索及具体工具的扩展选项不属于这里的普通逐行文本契约。

<!-- bilingual-en:start -->
The Shell passes the pattern as an argument, which normally needs quoting. `-e 'pattern'` explicitly marks the pattern argument, particularly when it starts with `-`. No output can mean either no selected lines or an error; inspect [[grep退出状态|grep's exit status]]. Binary input, cross-line searching, and implementation-specific extensions are outside this ordinary line-oriented text contract.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [POSIX grep, Description and Options](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/grep.html)；[GNU grep, Matching Control](https://www.gnu.org/software/grep/manual/html_node/Matching-Control.html)：支持逐行选择、完整行输出、`-F/-E/-x/-v/-c/-e` 的职责及模式参数边界。
  <!-- bilingual-en:start -->
  These sources support line selection, complete-line output, the roles of `-F/-E/-x/-v/-c/-e`, and explicit pattern arguments.
  <!-- bilingual-en:end -->
