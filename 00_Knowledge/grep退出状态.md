---
aliases:
  - "grep退出状态区分选中行、未选中行与执行错误"
  - "grep exit status distinguishes selection, no selection, and errors"
student_os: knowledge-atom
atom_id: CS-TEXT-006
atom_type: boundary
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[grep行筛选]]"
  - "[[命令退出状态]]"
related:
  - "[[Shell重定向]]"
---

# grep退出状态区分选中行、未选中行与执行错误

<!-- bilingual-en:start -->
*grep exit status distinguishes selected lines, no selected lines, and execution errors*
<!-- bilingual-en:end -->

普通 `grep` 的退出状态 `0` 表示选中了行并成功完成规定输出，`1` 表示没有选中行，大于 `1` 表示错误；GNU `grep` 的错误状态通常是 `2`。所以 `1` 不应自动解释为“文件读取失败”，无输出也不能单独区分没有匹配与执行错误。

<!-- bilingual-en:start -->
For ordinary `grep`, exit status `0` means lines were selected and the prescribed output succeeded, `1` means no lines were selected, and a value greater than `1` indicates an error. GNU `grep` normally uses `2` for errors. Thus `1` should not automatically be interpreted as a file-read failure, and empty output alone cannot distinguish no match from an execution error.
<!-- bilingual-en:end -->

检查是否存在匹配时可用 `grep -q`，但它的契约更窄：找到选中行就可成功返回，即使还发生了输入错误。因此它适合回答“是否找到了匹配”，不能单独证明“所有输入都已完整读完且无错”。

<!-- bilingual-en:start -->
`grep -q` is useful for checking whether a match exists, but its contract is narrower: it can return success upon finding a selected line even when an input error also occurs. It answers whether a match was found; it does not by itself establish that all input was read completely without errors.
<!-- bilingual-en:end -->

```bash
if grep -q -F -e 'ERROR' ./app.log; then
    printf '%s\n' 'match found'
else
    grep_status=$?
    case "$grep_status" in
        1) printf '%s\n' 'no match' ;;
        *) printf '%s\n' 'search could not be completed' >&2 ;;
    esac
fi
```

这个分支对固定文本文件 `./app.log` 展示状态分流，不是完整错误恢复策略。若改用 [[Shell重定向|Shell 输入重定向]]而重定向失败，`grep` 根本不会运行，观察到的是 Shell 的失败状态，不能套用 `grep` 的 `1` 号语义。`$?` 还必须在别的命令覆盖它以前保存。

<!-- bilingual-en:start -->
This branch illustrates status dispatch for the fixed text file `./app.log`, not a complete recovery policy. If [[Shell重定向|Shell input redirection]] is used instead and fails, `grep` never runs: the observed failure status belongs to the Shell rather than to grep's no-selection case. Save `$?` before another command overwrites it.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [POSIX grep, Exit Status](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/grep.html)；[GNU grep, Exit Status](https://www.gnu.org/software/grep/manual/html_node/Exit-Status.html)：支持 `0/1/>1` 区分、GNU 的错误状态以及 `-q` 在有匹配时的错误例外；示例区分的是实际运行的 `grep` 的状态。
  <!-- bilingual-en:start -->
  These sources establish `0/1/>1`, GNU's error status, and the `-q` exception when a match is found. The example's grep-specific interpretation applies only when grep actually ran.
  <!-- bilingual-en:end -->
- 本机 GNU Bash 3.2 `bash(1)`，`EXIT STATUS`、`Special Parameters` 与 `REDIRECTION`：核对重定向失败属于 Shell 执行边界，以及 `$?` 应及时读取或保存。
  <!-- bilingual-en:start -->
  The installed GNU Bash 3.2 manual supports the Shell-level redirection-failure boundary and prompt reading or preservation of `$?`.
  <!-- bilingual-en:end -->
