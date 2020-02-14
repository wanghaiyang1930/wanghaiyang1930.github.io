Golang-Channel-point.md

### 并发模型

- 多线程
```
多线程并发模型是在一个应用程序中同时存在多个执行流，多个执行流通过共享内存、信号量、互斥量等方式进行通信，CPU在多个执行流间进行上下文切换，从而达到并发执行，提高CPU利用率。
```

- CSP
```
CSP并发模型的将程序的执行和通信划分开来（Process和Channel），Process代表了任务执行单元，Channel用来在多个执行单元之间进行数据交互。Process内部不存在并发，所有由通信带来的并发问题都被压缩在Channel中，使得并发聚合在一起，得到了约束与同步，因此竞争聚焦在Channel上。Go就是基于这种并发模型实现的，Go在线程的基础上实现了这一套并发模型（MPG），线程之上虚拟出了协程的概念，一个协程代表一个Process，但在操作系统级别调度的基本单位依然是线程，只是Go自己实现了一个调度器，用来管理协程的调度，M（Machine）代表一个内核线程，P（Process）代表一个调度器，G(Goroutine)代表一个协程，其本质是内核线程和用户态线程成了多对多的关系。
```

### 并发示例
```
package main

import "fmt"
import "sync"

func main() {
	wg := new(sync.WaitGroup)
	wg.Add(1)

	go func() {
		defer wg.Done()
		fmt.Println("Hello world inner")
	} ()

	fmt.Println("Hello world outer")
	wg.Wait()
}
```

### channel 
```
c := make(chan int)
or
c := make(chan int, 10)

c <- 1           // Write data
data, ok := <- c // Read data

close(c)

for v := range c { // 遍历操作

}
```

### 关于 Close()

```
The close built-in function closes a channel, which must be either bidirectional or send-only. It should be executed only by the sender, never the receiver, and has the effect of shutting down the channel after the last sent value is received. After the last value has been received from a closed channel c, any receive from c will succeed without
blocking, returning the zero value for the channel element. The form x, ok := <-c will also set ok to false for a closed channel.
```

- 循环死锁
```
buf1 := make(chan int)
buf2 := make(chan int)

func read() {
	<- buf1
	buf2 <- 1
}

func write() {
	buf1 <- 1
	close(buf1) // 如果不关闭buf1，read和write就会出现互相等待的死锁现象
}

func main() {
	go read()
	go write()
	<- buf2
	close(buf2)
}
```

### Channel 的坑

```
func main(){
    c := make(chan int,10)
    //从一个永远都不可能有值的通道中读取数据，会发生死锁，因为会阻塞主程序的执行
    <- c
}

func main(){
    c := make(chan int,10)
    //往一个永远都不吭你又消费者的通道中写入数据，会发生死锁, 因为会阻塞主程序的执行
    c <- 1
}
```

```
func main(){
    
    c1 := make(chan int)
    c2 := make(chan int)
    go func(){
    	// 相互死锁
        c1 <- 1
        c2 <- 2
    }()

    // 相互死锁
    <- c2
    <- c1
}
```

### 经典应用

```
//select基本用法，用来监听和channel有关的IO操作，当 IO 操作发生时，触发相应的动作
select {
    case <- c1:
    // 如果c1成功读到数据，则进行该case处理语句
    case c2 <- 1:
    // 如果成功向c2写入数据，则进行该case处理语句
    default:
    // 如果上面都没有成功，则进入default处理流程
}
```

```
package main

func queryUserById(id int)chan string{
    c := make(chan string)
    go func(){
        c <- "姓名"+strconv.Itoa(id)
    }()
    return c
}

func main(){
    //多个不依赖的服务可以并发执行，三个协程同时并发查询，缩小执行时间，本来一次查询需要1秒，顺序执行就得3秒，
    //现在并发执行总共1秒就执行完成
    name1 := queryUserById(1)
    name2 := queryUserById(2)
    name3 := queryUserById(3)

    //从通道中获取执行结果
    <- name1
    <- name2
    <- name3
}
```

```
package main

import "fmt"
import "strconv"

func queryUserById(id int)chan string{
    c := make(chan string)
    fmt.Println("fisrt line: 11", id)
    go func(){
    	fmt.Println("fisrt line: 12", id)
        c <- "Name："+strconv.Itoa(id) 
        // 即：在没有执行读之前，这里阻塞，不会执行13，13 与 21、22、23顺序不定，但是13绝对不会出现在（“------”）之前，
        // 也就是说不会出现在 := <- c1/2/3 之前，没有读，写就会受到阻塞
        fmt.Println("fisrt line: 13", id)
    }()
    fmt.Println("fisrt line: 14", id)
    return c
}

func main(){    
    c1, c2, c3 := queryUserById(1), queryUserById(2), queryUserById(3)
    c := make(chan string)
    // select 监听通道合并多个通道的值到一个通道，开一个goroutine监视各个信道数据输出并收集数据到信道c
    go func() { 
        for {
            // 监视c1, c2, c3的流出，并全部流入信道c
            fmt.Println("--------------------")
            select {
               case v1 := <- c1:  
                   fmt.Println("fisrt line: 21")      
                   c <- v1
               case v2 := <- c2:   
                   fmt.Println("fisrt line: 22")     
                   c <- v2
               case v3 := <- c3:  
                   fmt.Println("fisrt line: 23")     
                   c <- v3
            }
        }
    }()
    // 阻塞主线，取出信道c的数据
    for i := 0; i < 3; i++ {
         // 从打印来看我们的数据输出并不是严格的顺序
        fmt.Println(<-c) 
    }
}
```

```
func main() {

    c, quit := make(chan int), make(chan int)
    go func() {
        c <- 2  // 添加数据
        quit <- 1 // 发送完成信号
    } ()
    for is_quit := false; !is_quit; {
        // 监视信道c的数据流出
        select { 
            case v := <-c: fmt.Printf("received %d from c", v)
            case <-quit: is_quit = true 
            // quit信道有输出，关闭for循环
        }
    }
}
```