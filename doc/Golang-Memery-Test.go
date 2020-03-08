package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
)

// Document: https://blog.cloudflare.com/recycling-memory-buffers-in-go/

const LENGTH = 5000000;

func makeBuffer() []byte {
	return make([]byte, rand.Intn(LENGTH) + LENGTH)
}

func main() {
	pool := make([][]byte, 20)

	var m runtime.MemStats

	makes := 0
	for {
		buf := makeBuffer()
		makes += 1

		i := rand.Intn(len(pool))
		pool[i] = buf

		time.Sleep(time.Second)

		bytes := 0

		for i := 0; i < len(pool); i++ {
			if pool[i] != nil {
				bytes += len(pool[i])
			}
		}

		runtime.ReadMemStats(&m)
		fmt.Printf("%d, %d, %d, %d, %d, %d\n",
			m.HeapSys, bytes, m.HeapAlloc, m.HeapIdle, m.HeapReleased, makes)

		// HeapSys: the numnber of bytes the program has asked the operating system for
		// HeapAlloc: the number of bytes currently allocated in the heap;
		// HeapIdle: the number of bytes in the heap that are unused;
		// HeapReleased: the number of bytes returned to the operating system;
	}
}