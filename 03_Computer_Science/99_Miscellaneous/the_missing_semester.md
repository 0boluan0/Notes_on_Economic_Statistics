# The Missing Semester - Shell 基础操作

## 一、Shell 简介
Shell 是一个命令行解释器，它提供了用户与操作系统内核之间的交互接口。常见的 Shell 包括 Bash（Linux/macOS 默认）、Zsh、Fish 等。

---

## 二、文件与目录管理

### 1. `pwd` - 显示当前工作目录
**功能**：Print Working Directory，显示当前所在的目录路径。
>[!example] 示例
```bash
pwd
# 输出：/Users/yourname/Documents
```

### 2. `cd` - 切换工作目录
**功能**：Change Directory，切换到指定目录。
**常用用法**：
```bash
cd /path/to/directory  # 切换到指定绝对路径
cd relative/path       # 切换到指定相对路径
cd ~                   # 切换到用户主目录（家目录）
cd -                   # 切换到上一个工作目录
cd ..                  # 切换到上级目录
cd .                   # 保持当前目录不变（通常用于脚本）
```

### 3. `ls` - 列出目录内容
**功能**：List，列出目录中的文件和子目录。
**常用选项**：
```bash
ls                      # 列出当前目录可见文件
ls -l                   # 详细列表格式（权限、大小、修改时间等）
ls -a                   # 显示所有文件（包括隐藏文件，以.开头）
ls -la                  # 详细列出所有文件
ls -h                   # 以人类可读格式显示文件大小
ls -t                   # 按修改时间排序（最新的在最前）
ls -R                   # 递归列出目录内容
ls /path/to/directory   # 列出指定目录内容
```

### 4. `mkdir` - 创建目录
**功能**：Make Directory，创建新目录。
**常用选项**：
```bash
mkdir new_dir           # 创建单个目录
mkdir -p dir1/dir2/dir3 # 递归创建多层目录（如果父目录不存在）
mkdir -m 755 my_dir     # 创建目录并设置权限（755表示所有者读写执行，其他只读执行）
```

### 5. `rmdir` - 删除空目录
**功能**：Remove Directory，删除空目录（只能删除空目录）。
>[!example] 示例
```bash
rmdir empty_dir
```

### 6. `rm` - 删除文件/目录
**功能**：Remove，删除文件或目录。
**常用选项**：
```bash
rm file.txt             # 删除单个文件
rm -f file.txt          # 强制删除（不提示确认）
rm -r dir               # 递归删除目录及其内容
rm -rf dir              # 强制递归删除（危险操作，慎用）
```

### 7. `cp` - 复制文件/目录
**功能**：Copy，复制文件或目录。
**常用选项**：
```bash
cp source.txt dest.txt  # 复制文件
cp -r source_dir dest_dir # 递归复制目录
cp -i source.txt dest.txt # 覆盖前提示确认
cp -v source.txt dest.txt # 显示详细复制过程
```

### 8. `mv` - 移动/重命名文件/目录
**功能**：Move，移动文件/目录或重命名。
>[!example] 示例
```bash
mv file.txt new_name.txt # 重命名文件
mv file.txt /path/to/dir # 移动文件到指定目录
mv dir /path/to/destination # 移动目录
mv -i file.txt dest/    # 覆盖前提示确认
```

### 9. `chmod` - 修改文件权限
**功能**：Change Mode，修改文件或目录的访问权限。
**权限表示方法**：
- **数字表示法**：r=4, w=2, x=1（r:读, w:写, x:执行）
- **符号表示法**：u(用户), g(组), o(其他), a(所有)

**常用用法**：
```bash
chmod 755 file.sh       # 用户:rwx, 组:r-x, 其他:r-x（常用的执行权限）
chmod 644 file.txt      # 用户:rw-, 组:r--, 其他:r--（常用的文件权限）
chmod +x file.sh        # 给所有用户添加执行权限
chmod u+x file.sh       # 给用户添加执行权限
chmod g-w file.txt      # 给组移除写权限
chmod o-rwx file.txt    # 给其他用户移除所有权限
chmod -R 755 dir/       # 递归修改目录及其内容的权限
```

### 10. `chown` - 修改文件所有者
**功能**：Change Owner，修改文件或目录的所有者和组。
**常用用法**：
```bash
chown user file.txt     # 变更文件所有者为user
chown user:group file.txt # 变更文件所有者为user，所属组为group
chown -R user:group dir/ # 递归修改目录及其内容的所有权
```

---

## 三、文件内容查看与处理

### 1. `cat` - 查看文件内容
**功能**：Concatenate，连接并显示文件内容。
>[!example] 示例
```bash
cat file.txt            # 显示文件全部内容
cat -n file.txt         # 显示内容并添加行号
```

### 2. `head` - 查看文件开头部分
**功能**：显示文件的前几行（默认10行）。
>[!example] 示例
```bash
head file.txt           # 显示前10行
head -n 20 file.txt     # 显示前20行
```

### 3. `tail` - 查看文件结尾部分
**功能**：显示文件的后几行（默认10行）。
**常用选项**：
```bash
tail file.txt           # 显示后10行
tail -n 15 file.txt     # 显示后15行
tail -f log.txt         # 实时跟踪文件更新（常用语查看日志）
```

### 4. `sort` - 排序文件内容
**功能**：对文件内容进行排序。
**常用选项**：
```bash
sort file.txt           # 字典序排序
sort -n file.txt        # 数值排序
sort -r file.txt        # 倒序排序
sort -k 2 file.txt      # 按第2列排序（默认以空格分隔）
sort -t',' -k 3 file.csv # 以逗号分隔，按第3列排序
```

### 5. `grep` - 搜索文本
**功能**：Global Regular Expression Print，在文件中搜索匹配模式的文本。
**常用选项**：
```bash
grep "pattern" file.txt # 在文件中搜索字符串
grep -i "pattern" file.txt # 忽略大小写搜索
grep -r "pattern" dir/  # 递归搜索目录中的所有文件
grep -n "pattern" file.txt # 显示匹配行的行号
grep -v "pattern" file.txt # 显示不匹配的行
grep -A 2 "pattern" file.txt # 显示匹配行及其后2行
grep -B 2 "pattern" file.txt # 显示匹配行及其前2行
grep -C 2 "pattern" file.txt # 显示匹配行及其前后各2行
```

### 6. `sed` - 流编辑器
**功能**：Stream Editor，用于对文本进行编辑和转换（擅长替换操作）。
**常用用法**：
```bash
sed 's/old/new/' file.txt # 替换每行第一个匹配的old为new
sed 's/old/new/g' file.txt # 替换所有匹配的old为new
sed '10s/old/new/' file.txt # 替换第10行的old为new
sed -i.bak 's/old/new/g' file.txt # 直接修改文件（创建备份file.txt.bak）
```

### 7. `awk` - 文本处理工具
**功能**：强大的文本处理工具，擅长按列处理数据（逐行扫描、分割字段、处理数据）。
**基本语法**：
```bash
awk 'pattern { action }' file.txt
```

**常用用法**：
```bash
awk '{print $1}' file.txt         # 打印第一列（默认以空格/制表符分隔）
awk -F',' '{print $2}' data.csv   # 以逗号分隔，打印第二列
awk '{print $1, $3}' file.txt     # 打印第一列和第三列
awk '/pattern/ {print $0}' file.txt # 打印包含pattern的所有行（$0表示整行）
awk 'NR > 5 {print $0}' file.txt  # 打印第5行之后的所有行（NR是行号）
awk '{sum += $1} END {print sum}' file.txt # 计算第一列的总和
awk '{if ($2 > 100) print $1}' file.txt # 打印第二列大于100的行的第一列
```

**内置变量**：
- `$0`：整行内容
- `$1, $2, ...`：第1列、第2列等（默认以空格分隔）
- `NF`：当前行的字段数
- `NR`：当前行号
- `FS`：字段分隔符（默认是空格）
- `OFS`：输出字段分隔符（默认是空格）
- `RS`：记录分隔符（默认是换行）
- `ORS`：输出记录分隔符（默认是换行）
- `FILENAME`：当前文件名

**高级用法示例**：
```bash
# 使用多个分隔符（空格或逗号）
awk -F'[ ,]' '{print $1, $3}' data.txt

# 格式化输出（固定宽度）
awk '{printf "%-10s %5d\n", $1, $2}' data.txt

# 处理多个文件
awk 'FNR == 1 {print "---", FILENAME, "---"} {print}' file1.txt file2.txt

# 使用自定义函数
awk '
function max(a, b) {
    return (a > b) ? a : b
}
{print max($1, $2)}' data.txt

# 数组使用（统计词频）
awk '{for(i=1;i<=NF;i++) count[$i]++}
END {for(w in count) print w, count[w]}' text.txt
```

**实际场景示例**：
```bash
# 处理CSV文件 - 统计各分数段人数
awk -F',' 'BEGIN {
    print "Score Range\tCount"
    print "------------------"
    for(i=0;i<=100;i+=10)
        ranges[i] = 0
}
NR>1 {
    score = $3
    if(score >= 0 && score <= 100) {
        range = int(score / 10) * 10
        ranges[range]++
    }
}
END {
    for(r in ranges) {
        if(ranges[r] > 0) {
            printf "%3d-%3d\t%d\n", r, r+9, ranges[r]
        }
    }
}' students.csv

# 分析日志文件 - 统计访问量最高的IP
awk '{count[$1]++}
END {for(ip in count) print count[ip], ip}' access.log | sort -nr | head -10
```

---

## 四、系统与信息查询

### 1. `date` - 显示/设置日期时间
**功能**：显示当前系统日期和时间，或设置系统时间。
>[!example] 示例
```bash
date                    # 显示当前日期时间
date "+%Y-%m-%d %H:%M:%S" # 自定义格式显示：2023-10-05 14:30:00
date "+%A, %B %d, %Y"    # 显示完整格式：Thursday, October 05, 2023
date -d "2 days ago"     # 显示两天前的日期
date -d "next Monday"    # 显示下周一的日期
```

**常用格式化选项**：
- `%Y`：4位年份
- `%m`：月份（01-12）
- `%d`：日期（01-31）
- `%H`：小时（00-23）
- `%M`：分钟（00-59）
- `%S`：秒（00-59）
- `%A`：完整星期名
- `%B`：完整月份名
- `%j`：年积日（001-366）

### 2. `echo` - 输出文本
**功能**：输出字符串或变量值。
>[!example] 示例
```bash
echo "Hello World"      # 输出字符串
echo $PATH              # 输出环境变量值
echo -e "Line1\nLine2"  # 解析转义字符（换行）
echo -n "No newline"    # 不输出尾随换行符
echo "Current dir: $(pwd)" # 命令替换
echo "User: $USER, Host: $HOSTNAME" # 输出多个变量
```

### 3. `man` - 查看命令手册
**功能**：Manual，显示命令的详细帮助信息（手册页）。
>[!example] 示例
```bash
man ls                  # 查看ls命令的手册
man 5 passwd            # 查看passwd配置文件的手册（第5部分）
man -k "search term"    # 搜索包含特定关键词的手册
man -f command          # 查看命令的简短描述
```

**手册页章节**：
- 1：用户命令
- 2：系统调用
- 3：库函数
- 4：特殊文件和设备
- 5：配置文件格式
- 6：游戏
- 7：杂项
- 8：系统管理命令和守护进程

### 4. `which` - 查找命令路径
**功能**：显示命令的完整路径（根据PATH环境变量）。
>[!example] 示例
```bash
which python3           # 显示python3命令的路径
which -a ls             # 显示所有同名命令的路径
```

### 5. `whoami` - 显示当前用户
**功能**：显示当前登录用户的用户名。
>[!example] 示例
```bash
whoami
```

### 6. `id` - 显示用户和组信息
**功能**：显示当前用户的UID、GID和所属组信息。
>[!example] 示例
```bash
id                      # 显示完整信息
id -u                   # 只显示UID
id -g                   # 只显示GID
id -G                   # 显示所有所属组的GID
id -nG                  # 显示所有所属组的名称
```

### 7. `who` - 显示当前登录用户
**功能**：显示当前登录系统的用户信息。
>[!example] 示例
```bash
who                     # 显示用户和登录时间
who -u                  # 显示用户和空闲时间
who -b                  # 显示系统启动时间
```

### 8. `uptime` - 显示系统运行时间
**功能**：显示系统已运行的时间和平均负载。
>[!example] 示例
```bash
uptime                  # 显示运行时间和负载
```

---

## 五、核心概念详细解释

### 1. 文件权限系统
**Unix/Linux权限模型**：每个文件/目录都有三组权限（用户、组、其他）和三种权限类型（读、写、执行）。

**权限表示方法**：
```
权限位：rwxr-xr-x
         |   |   |
         |   |   +-- 其他用户权限
         |   +------ 组权限
         +---------- 用户权限

数字表示：r=4, w=2, x=1
rwx = 4+2+1 = 7
r-x = 4+0+1 = 5
r-- = 4+0+0 = 4

常用权限组合：
- 755：用户rwx，组r-x，其他r-x（执行文件常用）
- 644：用户rw-，组r--，其他r--（普通文件常用）
- 700：用户rwx，组和其他无权限（敏感文件）
- 600：用户rw-，组和其他无权限（敏感配置文件）
```

**特殊权限位**：
- **SetUID（SUID，4000）**：执行程序时以文件所有者身份运行
- **SetGID（SGID，2000）**：执行程序时以文件所属组身份运行；目录中的新文件继承组
- **Sticky（1000）**：目录中的文件只能由所有者删除（例如/tmp目录）

>[!example] 示例
```bash
chmod 4755 program     # 设置SUID权限
chmod 2775 directory   # 设置SGID权限
chmod 1777 tempdir     # 设置Sticky权限
```

### 2. 环境变量
>[!note] 定义
> 环境变量是Shell会话中可用的动态值，影响程序的行为。

**常用环境变量**：
```bash
PATH    # 命令搜索路径（用:分隔）
HOME    # 用户主目录路径
PWD     # 当前工作目录
SHELL   # 当前使用的Shell
USER    # 当前用户名
HOSTNAME # 主机名
LANG    # 语言和地区设置
TERM    # 终端类型
PS1     # 命令提示符格式
```

**操作环境变量**：
```bash
echo $PATH              # 查看PATH变量
export PATH=$PATH:/new/directory # 添加新路径到PATH
PATH=/new/directory:$PATH       # 前置新路径
export MY_VAR="value"   # 声明并导出变量
unset MY_VAR            # 删除变量
env                     # 列出所有环境变量
printenv                # 列出所有环境变量（更详细）
```

### 3. 进程管理基础
**进程状态**：
- R（运行）：正在执行或在就绪队列中
- S（睡眠）：等待资源（可中断）
- D（磁盘睡眠）：等待磁盘I/O（不可中断）
- T（停止）：暂停执行
- Z（僵尸）：进程已结束但父进程未回收

**常用命令**：
```bash
ps aux                  # 列出所有进程
ps -ef                  # 列出所有进程（BSD风格）
top                     # 实时显示进程资源使用情况
htop                    # 增强版top（需安装）
pkill -f "process name" # 根据名称杀死进程
kill PID                # 发送信号给进程（默认SIGTERM）
kill -9 PID             # 强制杀死进程（SIGKILL）
jobs                    # 列出后台任务
fg %1                   # 前台运行任务1
bg %1                   # 后台继续运行任务1
```

---

## 六、高级操作与概念

### 1. 管道（Pipe）- `|`
**功能**：将一个命令的输出作为另一个命令的输入。
>[!example] 示例
```bash
ls -l | grep ".txt"     # 列出所有.txt文件
cat file.txt | sort | head -5 # 对文件内容排序并显示前5行
```

### 2. 重定向
**常用操作符**：
```bash
command > file.txt      # 将输出重定向到文件（覆盖）
command >> file.txt     # 将输出重定向到文件（追加）
command < file.txt      # 从文件读取输入
command 2> error.txt    # 将错误输出重定向到文件
command &> output.txt   # 将标准输出和错误输出都重定向到文件
```

### 3. 通配符
**常用通配符**：
```bash
*                       # 匹配任意字符序列（0个或多个）
?                       # 匹配任意单个字符
[abc]                   # 匹配a、b或c
[!abc]                  # 匹配除了a、b、c之外的字符
[0-9]                   # 匹配任意数字
```

>[!example] 示例
```bash
ls *.txt                # 列出所有.txt文件
ls file?.txt            # 列出file1.txt、file2.txt等
ls [a-c]*.py            # 列出以a、b或c开头的.py文件
```

### 4. 环境变量
**常用环境变量**：
- `PATH`：命令搜索路径
- `HOME`：用户主目录
- `PWD`：当前工作目录
- `SHELL`：当前使用的Shell
- `USER`：当前用户名

**操作环境变量**：
```bash
echo $PATH              # 查看PATH变量
export PATH=$PATH:/new/directory # 添加新路径到PATH
```

---

## 六、查找文件与目录

### 1. `find` - 查找文件/目录
**功能**：在文件系统中查找符合条件的文件或目录。
**常用用法**：
```bash
find /path/to/search -name "*.txt" # 按文件名查找.txt文件
find . -type f -name "*.py"       # 在当前目录查找.py文件（-type f表示文件）
find . -type d -name "test*"      # 在当前目录查找以test开头的目录（-type d表示目录）
find . -mtime -7                  # 查找7天内修改过的文件
find . -size +100k                # 查找大于100KB的文件
find . -perm 755                  # 查找权限为755的文件
find . -name "*.tmp" -delete      # 查找并删除所有.tmp文件
```

**与其他命令配合使用**：
```bash
find . -name "*.txt" -exec grep "pattern" {} + # 在所有.txt文件中搜索pattern
```

---

## 七、常用快捷键
| 快捷键       | 功能                     |
|--------------|--------------------------|
| `Ctrl + C`   | 中断当前命令             |
| `Ctrl + Z`   | 暂停当前命令（可使用fg恢复） |
| `Ctrl + D`   | 退出Shell会话（EOF）     |
| `Ctrl + L`   | 清屏                     |
| `Ctrl + A`   | 移动到行首               |
| `Ctrl + E`   | 移动到行尾               |
| `Ctrl + K`   | 剪切从光标到行尾的内容   |
| `Ctrl + U`   | 剪切从光标到行首的内容   |
| `Ctrl + R`   | 搜索历史命令             |
| `Tab`        | 自动补全命令或路径       |

---

## 八、常见问题与解决方案

### 1. 文件权限问题
**问题**：Permission denied（权限被拒绝）
**解决方案**：
```bash
# 检查文件权限
ls -l file.txt
# 给用户添加执行权限
chmod +x script.sh
# 给用户添加读写权限
chmod u+rw file.txt
# 递归修改目录权限
chmod -R 755 directory/
```

**问题**：无法删除文件：Operation not permitted
**解决方案**：
```bash
# 检查是否有特殊权限位
ls -l file.txt
# 如果有i（immutable）属性，先去除
chattr -i file.txt
# 再尝试删除
rm file.txt
```

### 2. 文件处理问题
**问题**：文件内容显示乱码（编码问题）
**解决方案**：
```bash
# 检查文件编码（需安装file命令）
file -i file.txt
# 转换编码（需安装iconv）
iconv -f GBK -t UTF-8 file.txt > newfile.txt
```

**问题**：处理大文件时内存不足
**解决方案**：
```bash
# 使用逐行处理的命令
awk '{processing}' large_file.txt
# 或者使用split分割文件
split -l 10000 large_file.txt part_
# 处理完后合并
cat part_* > merged_file.txt
```

### 3. 系统性能问题
**问题**：系统运行缓慢，CPU或内存使用率高
**解决方案**：
```bash
# 查看进程资源使用
top
htop  # 更直观
# 查找CPU使用率最高的进程
ps aux --sort=-%cpu | head -10
# 查找内存使用率最高的进程
ps aux --sort=-%mem | head -10
# 检查磁盘空间
df -h
# 检查磁盘使用情况
du -sh /path/to/directory
```

### 4. 网络问题
**问题**：无法连接到远程主机
**解决方案**：
```bash
# 检查网络连通性
ping example.com
# 检查DNS解析
nslookup example.com
# 检查端口连通性
telnet example.com 80
nc -zv example.com 80  # 使用nc（netcat）
# 查看路由表
route -n
```

### 5. 命令使用问题
**问题**：找不到命令：Command not found
**解决方案**：
```bash
# 检查命令是否安装
which command_name
# 检查PATH变量
echo $PATH
# 如果命令在非标准路径，添加到PATH
export PATH=$PATH:/path/to/command
# 或者直接使用完整路径
/path/to/command
```

**问题**：命令输出过长难以阅读
**解决方案**：
```bash
# 使用管道和分页工具
command | less
command | more
# 保存到文件并查看
command > output.txt
cat output.txt
# 或在输出中搜索关键词
command | grep "pattern"
```

---

## 九、Shell脚本基础

### 1. 脚本基本结构
```bash
#!/bin/bash
# 这是一个简单的Shell脚本

# 定义变量
NAME="World"

# 输出文本
echo "Hello, $NAME!"

# 条件判断
if [ "$NAME" == "World" ]; then
    echo "Welcome to the world!"
else
    echo "Welcome, $NAME!"
fi

# 循环
echo "Counting from 1 to 5:"
for i in {1..5}; do
    echo "Number: $i"
done

# 函数
greet() {
    local NAME=$1
    echo "Hello, $NAME!"
}

# 调用函数
greet "Alice"
greet "Bob"
```

### 2. 脚本执行方式
```bash
# 给脚本添加执行权限
chmod +x script.sh
# 执行脚本
./script.sh
# 或者使用bash解释器
bash script.sh
```

### 3. 常见脚本用途
```bash
# 自动备份脚本
#!/bin/bash
BACKUP_DIR="/backup"
SOURCE_DIR="/home/user/documents"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz $SOURCE_DIR

# 监控系统资源脚本
#!/bin/bash
CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d. -f1)
MEM=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}')

echo "CPU Usage: ${CPU}%"
echo "Memory Usage: ${MEM}"

# 发送邮件通知（需安装mailx）
if [ $CPU -gt 80 ]; then
    echo "High CPU usage: ${CPU}%" | mailx -s "Warning: High CPU Usage" admin@example.com
fi
```

---

## 十、Shell语法详细介绍

### 1. 变量

#### 变量定义与赋值
```bash
# 基本变量定义
name="John"
age=30

# 变量名规则：只能包含字母、数字和下划线，不能以数字开头
# 错误示例：2name="John" 或 name@john="John"

# 变量引用
echo "My name is $name"
echo "I am ${age} years old"  # 推荐使用${}格式，避免歧义
```

#### 变量类型
```bash
# 字符串变量
str1="Hello World"
str2='Single quote string'  # 单引号内的内容原样输出

# 数值变量
num1=10
num2=20

# 命令替换
current_dir=$(pwd)
# 或使用反引号
current_dir=`pwd`

# 变量运算
sum=$((num1 + num2))
echo "Sum: $sum"  # 输出：Sum: 30
```

#### 只读变量
```bash
readonly PI=3.14159
PI=3.14  # 会报错：PI: readonly variable
```

#### 删除变量
```bash
unset name
echo $name  # 输出空值
```

### 2. 字符串操作

#### 字符串长度
```bash
str="Hello World"
echo "Length: ${#str}"  # 输出：11
```

#### 字符串截取
```bash
str="abcdefghijklmnopqrstuvwxyz"
echo "${str:0:5}"    # 从第0位开始，取5个字符：abcde
echo "${str:5}"      # 从第5位开始，取剩余所有字符：fghijklmnopqrstuvwxyz
echo "${str: -5}"    # 取最后5个字符：vwxyz
echo "${str:(-5)}"   # 同上，取最后5个字符：vwxyz
```

#### 字符串替换
```bash
str="Hello World"
echo "${str/World/Universe}"    # 替换第一个匹配的World：Hello Universe
echo "${str//l/L}"              # 替换所有的l为L：HeLLo WorLD
echo "${str/#Hello/Hi}"         # 替换开头的Hello：Hi World
echo "${str/%World/Earth}"      # 替换结尾的World：Hello Earth
```

#### 字符串大小写转换
```bash
str="Hello World"
echo "${str^^}"   # 全部大写：HELLO WORLD
echo "${str,,}"   # 全部小写：hello world
```

### 3. 数组

#### 数组定义与操作
```bash
# 数组定义
arr=("apple" "banana" "orange" "grape")

# 数组长度
echo "Array length: ${#arr[@]}"  # 输出：4

# 访问数组元素
echo "First element: ${arr[0]}"       # 输出：apple
echo "Second element: ${arr[1]}"      # 输出：banana

# 添加元素
arr[4]="watermelon"
arr+=("pineapple")  # 追加元素

# 删除元素
unset arr[2]
echo "${arr[@]}"  # 输出：apple banana grape watermelon pineapple

# 数组切片
echo "${arr[@]:1:3}"  # 从索引1开始，取3个元素：banana grape watermelon
```

#### 关联数组（键值对）
```bash
# 声明关联数组
declare -A colors
colors["red"]="#FF0000"
colors["green"]="#00FF00"
colors["blue"]="#0000FF"

# 访问关联数组
echo "Red color: ${colors["red"]}"  # 输出：#FF0000

# 遍历关联数组
for key in "${!colors[@]}"; do
    echo "Key: $key, Value: ${colors[$key]}"
done
```

### 4. 条件判断

#### `if` 语句结构
`if` 语句是Shell条件判断的基础语法，使用 `fi` 作为结束标记：
```bash
# 基本语法
if [ condition ]; then
    # 条件成立时执行的命令
elif [ another_condition ]; then
    # 另一个条件成立时执行的命令
else
    # 所有条件都不成立时执行的命令
fi
```

#### 文件判断
```bash
# 判断文件是否存在
if [ -e "file.txt" ]; then
    echo "File exists"
else
    echo "File not exists"
fi

# 判断是否为普通文件
if [ -f "file.txt" ]; then
    echo "It's a regular file"
fi

# 判断是否为目录
if [ -d "dir" ]; then
    echo "It's a directory"
fi

# 判断文件是否可读
if [ -r "file.txt" ]; then
    echo "File is readable"
fi

# 判断文件是否可写
if [ -w "file.txt" ]; then
    echo "File is writable"
fi

# 判断文件是否可执行
if [ -x "script.sh" ]; then
    echo "File is executable"
fi
```

#### 数值比较
```bash
num1=10
num2=20

# 等于
if [ $num1 -eq $num2 ]; then
    echo "Equal"
fi

# 不等于
if [ $num1 -ne $num2 ]; then
    echo "Not equal"
fi

# 小于
if [ $num1 -lt $num2 ]; then
    echo "Less than"
fi

# 小于等于
if [ $num1 -le $num2 ]; then
    echo "Less than or equal"
fi

# 大于
if [ $num1 -gt $num2 ]; then
    echo "Greater than"
fi

# 大于等于
if [ $num1 -ge $num2 ]; then
    echo "Greater than or equal"
fi
```

#### 字符串比较
```bash
str1="hello"
str2="world"

# 字符串相等
if [ "$str1" = "$str2" ]; then
    echo "Strings are equal"
fi

# 字符串不相等
if [ "$str1" != "$str2" ]; then
    echo "Strings are not equal"
fi

# 字符串长度为0
if [ -z "$empty_str" ]; then
    echo "String is empty"
fi

# 字符串长度不为0
if [ -n "$str1" ]; then
    echo "String is not empty"
fi
```

### 5. 循环结构

#### for循环
```bash
# 遍历数组
fruits=("apple" "banana" "orange")
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

# 遍历数字范围
for ((i=1; i<=5; i++)); do
    echo "Number: $i"
done

# 遍历文件
for file in *.txt; do
    echo "File: $file"
done
```

#### while循环
```bash
# 基本while循环
count=1
while [ $count -le 5 ]; do
    echo "Count: $count"
    ((count++))
done

# 读取文件内容
while read -r line; do
    echo "Line: $line"
done < file.txt

# 无限循环
while true; do
    echo "This is an infinite loop"
    sleep 1
done
```

#### until循环
```bash
count=1
until [ $count -gt 5 ]; do
    echo "Count: $count"
    ((count++))
done
```

### 6. 函数

#### 函数定义与调用
```bash
# 基本函数定义
greet() {
    echo "Hello, world!"
}

# 调用函数
greet  # 输出：Hello, world!

# 带参数的函数
greet_person() {
    local name=$1
    echo "Hello, $name!"
}

greet_person "John"  # 输出：Hello, John!
greet_person "Jane"  # 输出：Hello, Jane!
```

#### 函数返回值
```bash
# 使用return返回值
add() {
    local num1=$1
    local num2=$2
    local sum=$((num1 + num2))
    return $sum
}

add 10 20
echo "Sum: $?"  # 输出：Sum: 30

# 使用输出返回值
multiply() {
    local num1=$1
    local num2=$2
    echo $((num1 * num2))
}

result=$(multiply 10 20)
echo "Product: $result"  # 输出：Product: 200
```

#### 函数变量作用域
```bash
# 全局变量
global_var="global"

show_var() {
    # 局部变量
    local local_var="local"
    echo "Local variable: $local_var"
    echo "Global variable: $global_var"
}

show_var
echo "Global variable outside: $global_var"
# echo "Local variable outside: $local_var"  # 会报错：local_var: not found
```

---

## 十一、高级Shell编程技巧

### 1. 输入输出处理

#### 读取用户输入
```bash
# 简单输入
echo -n "Enter your name: "
read name
echo "Hello, $name!"

# 带默认值的输入
read -p "Enter your age: " age
age=${age:-18}  # 如果没有输入，默认值为18
echo "Your age is $age"

# 隐藏输入（密码输入）
read -p "Enter your password: " -s password
echo
echo "Password entered: $password"
```

#### 格式化输出
```bash
# 使用printf
printf "Name: %-10s Age: %3d\n" "John" 30
printf "Price: $%.2f\n" 19.99
printf "Date: %04d-%02d-%02d\n" 2023 10 5
```

### 2. 错误处理

#### 错误检查
```bash
# 检查命令执行是否成功
if ls /nonexistent >/dev/null 2>&1; then
    echo "Command succeeded"
else
    echo "Command failed"
    echo "Exit code: $?"
fi

# 强制脚本在错误时退出
set -e  # 启用错误检查
command1  # 命令失败时脚本会立即退出
command2
```

#### 错误陷阱
```bash
# 设置错误陷阱
cleanup() {
    echo "Script interrupted, cleaning up..."
    # 执行清理操作
}

trap cleanup INT TERM EXIT

# 主程序
echo "Running..."
sleep 10
```

### 3. 正则表达式

#### 基本正则表达式
```bash
# 使用grep进行正则匹配
grep -E "^[0-9]+$" file.txt  # 匹配纯数字行
grep -E "^[a-zA-Z]+$" file.txt  # 匹配纯字母行
grep -E "^[a-zA-Z0-9_]+$" file.txt  # 匹配字母、数字、下划线
```

#### 正则表达式替换
```bash
# 使用sed进行正则替换
sed 's/[0-9]\+/NUMBER/g' file.txt  # 将所有数字替换为NUMBER
sed 's/^ *//g' file.txt  # 去除行首空格
sed 's/ *$//g' file.txt  # 去除行尾空格
```
