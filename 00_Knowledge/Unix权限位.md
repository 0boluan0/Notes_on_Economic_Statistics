---
aliases:
  - "Unix权限位分别规定所有者、组和其他用户的读写执行权限"
  - "Unix permission bits"
student_os: knowledge-atom
atom_id: CS-SHELL-023
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[文件路径]]"
related:
  - "[[Shebang]]"
  - "[[Shell重定向]]"
---

# Unix权限位分别规定所有者、组和其他用户的读写执行权限

Unix 的基本权限位把所有者、所属组和其他用户三类访问者分别对应到 `r`、`w`、`x`。对普通文件，它们分别表示读内容、改内容和执行；对目录，含义变为读取目录项名称、修改目录项和搜索穿越目录。实际访问还受身份选择、ACL、文件系统限制或特权等条件影响，不能只看一个 `x` 就断言操作必定成功。
<!-- bilingual-en:start -->
Basic Unix permission bits assign `r`, `w`, and `x` separately to owner, group, and other access classes. For regular files these mean reading contents, modifying contents, and execution. For directories they concern reading entry names, modifying entries, and searching/traversing. Actual access also depends on identity selection, ACLs, filesystem restrictions, or privileges; one `x` bit does not guarantee success.
<!-- bilingual-en:end -->

例如 `rw-r-----` 表示所有者可读写、组可读、其他用户无这三种权限；数字形式为 `640`。每组三位分别按 4、2、1 相加，因此 `rwx` 为 7、`r-x` 为 5。三类位不是不加区分地相加后授予每个用户。
<!-- bilingual-en:start -->
For example, `rw-r-----` grants owner read/write, group read, and none of these permissions to others; its numeric form is `640`. Each triplet uses weights 4, 2, and 1, so `rwx` is 7 and `r-x` is 5. The three access classes are not indiscriminately combined for every user.
<!-- bilingual-en:end -->

目录的 `r` 与 `x` 不同：能列出名字不必意味着能访问这些名字指向的对象。创建或删除目录项通常需要父目录的写与搜索权限，还可能受 sticky bit 等限制；删除某文件并不简单等于“对该文件具有写权限”。给脚本增加执行位也不等于赋予它任意读写权限。
<!-- bilingual-en:start -->
Directory `r` and `x` differ: being able to list names need not allow access to the named objects. Creating or removing an entry normally needs write and search access to its parent directory and may face restrictions such as the sticky bit. Deletion is not simply permission to write the file itself. Making a script executable does not grant it arbitrary read/write access.
<!-- bilingual-en:end -->

## 来源与核验

- 本机 `chmod(1)`，`MODES`；`unlink(2)`，访问条件与错误：支持三类权限、数字权重、目录搜索和删除取决于父目录的规则。
- [Missing Semester 2020：The Shell](https://missing.csail.mit.edu/2020/course-shell/)，Navigating in the shell 的 permissions 段：支持课程中的文件与目录权限区别；[POSIX chmod](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/chmod.html) 提供标准接口对照。
<!-- bilingual-en:start -->
The local `chmod` and `unlink` manuals support permission classes, numeric weights, directory search, and parent-directory conditions for deletion. The course introduces file-versus-directory permissions; POSIX `chmod` provides the standard-interface reference.
<!-- bilingual-en:end -->
