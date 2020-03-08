
### Macro

在 C99 中 为实现32位与64位的程序跨平台，重定义了如下类型，同时也定义了 UINTPTR_MAX（也可以用来判断32/64位平台）;


```
#include <stdint.h> // C99

 
/* Types for `void *' pointers. */
#if __WORDSIZE == 64
	# ifndef __intptr_t_defined
	typedef long int intptr_t;
	# define __intptr_t_defined
	# endif
	typedef unsigned long int uintptr_t;
#else
	# ifndef __intptr_t_defined
	typedef int intptr_t;
	# define __intptr_t_defined
	# endif
	typedef unsigned int uintptr_t;
#endif

/* Values to test for integral types holding `void *' pointer.  */
# if __WORDSIZE == 64
#  define INTPTR_MIN            (-9223372036854775807L-1)
#  define INTPTR_MAX            (9223372036854775807L)
#  define UINTPTR_MAX           (18446744073709551615UL) // uintptr_t 类型对象的最大值
# else
#  define INTPTR_MIN            (-2147483647-1)
#  define INTPTR_MAX            (2147483647)
#  define UINTPTR_MAX           (4294967295U) 			 // uintmax_t 类型对象的最大值
# endif

```

### __builtin_expect

long __builtin_expect(long EXP, long C);

告知编译器表达式EXP的取值很可能是C，指令的返回值依然为EXP的取值，如果编译时加上-freorder-blocks选项（在-O2时会启用），编译时会重新安排代码生成顺序，可以提高处理器Pipeline处理的效率（减少了分支跳转）。

在 EXP 处经常使用 !!(x) 网上有说这是在归一化（1/0），但没有找到官方解释。

[Reference](https://www.cnblogs.com/pengdonglin137/articles/3808631.html)

- Google
```
#ifndef GOOGLE_PREDICT_TRUE
#ifdef __GNUC__ 
//__GNUC__ >= 3
// Provided at least since GCC 3.0.
#define GOOGLE_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define GOOGLE_PREDICT_TRUE(x) (x)
#endif
#endif

#ifndef GOOGLE_PREDICT_FALSE
#ifdef __GNUC__
// __GNUC__ >= 3
// Provided at least since GCC 3.0.
#define GOOGLE_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#else
#define GOOGLE_PREDICT_FALSE(x) (x)
#endif
```

- Redis
```
#if __GNUC__ >= 3
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
```