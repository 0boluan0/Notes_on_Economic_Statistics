
# 0_Scratch

computer use binary to represent numbers.

a byte contain 8 bits

基本就是一些感知上的东西,,简单理解input output 和 algorithm . 看了一些用scratch写的小程序.意义不大.

# 1_C

the code we write is called source code
computer only recognise  machine code :the binary 0-1 list. so there is a system to transcript the source code to the machine code .
the device we use to achieve this is called complier

3 dimensions(方面) to analyse you code:
1. correctness
2. design :often run faster when the code is well designed 
3. style: is your code easy to read


## write your first C code

### create a file 

in terminal , type: code hello.c

### 编译(compile)文件

also in terminal~~  ,~~ type `make hello`~~    ,no .c.~~（直接运行 `make hello`，无需写扩展名 `.c`）

### run the file

./<filename(no c, [[Just-identified|just]] file name)>
eg : ./hello
### print

```c
printf("hello,world\n");
```

f means :formatted ,按照指定格式输出
\n change the line 

C force that if you have a string , it should be in one line, changing the line by yourself is not allowed.
