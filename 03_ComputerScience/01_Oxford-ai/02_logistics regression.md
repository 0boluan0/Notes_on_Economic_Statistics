
实际上是一个classification

使用regression不能解决二分类问题

在linear regression的基础上将输出过一遍sigmoid函数,归到0-1之间.

softmax非常的巧妙,所有人的和是1,所以只要将对应的组拉到足够大,那就可以将整个组变成一堆0和一个1.