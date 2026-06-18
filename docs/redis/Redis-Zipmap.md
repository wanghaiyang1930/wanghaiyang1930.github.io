# Zipmap

## Brief

```
/* String -> String Map data structure optimized for size.
 * This file implements a data structure mapping strings to other strings
 * implementing an O(n) lookup data structure designed to be very memory
 * efficient.
 *
 * The Redis Hash type uses this data structure for hashes composed of a small
 * number of elements, to switch to a hash table once a given number of
 * elements is reached.
 *
 * Given that many times Redis Hashes are used to represent objects composed
 * of few fields, this is a very big win in terms of used memory.
 */
```
> 切换条件

Hash 在到达一定元素数据后，数据结构会由 Zipmap 切换为 HashTable，条件是：

## Structure

> zmlen：
占用 1 Btye，表示 Zipmap 中元素（键值对）的个数，如果其：== 254，那么这个字段就不在有效，想要获取元素个数，需要遍历整个 Zipmap（这个设定很奇葩啊？难道不能像 len 一样用变长策略吗？）

> len

Variable 1 or 4 Bytes, it is the length of the following string (key or value). 

变长存储，如果长度 < 254 占用 1 Byte，如果 = 254 则占用 5 个字节（首字节存储 254，后面4字节存储长度）。

（哈哈哈不要问如果是255是什么情况！！！！！！）

> free

1 Byte，当 value 更新后，长度有可能会缩短，此时就会产生一个空闲空间，Redis 规定这个不得大于 4 Bytes，如果大于 4 Bytes，就会 realloc 压缩空间，依次来保证一个较高的空间占用率

> end

1 Byte，value = 255。

> example

Pesudo data: <zmlen><len>"foo"<len><free>"bar"<len>"hello"<len><free>"world"
Memory:      "\x02\x03foo\x03\x00bar\x05hello\x05\x00world\xff"

### zipmapLookupRaw

Q：为什么一开始：p = zm + 1 ？？？为啥，顶噶？
A：因为 Zipmap 的第一个字段（键值对/元素个数）并不采用变长存储，只保持 1 Byte 大小，当 zmlen >= 254，该字段即失效。

### zipmapDel

实现思路：主要依赖于 memmove 整体将内存向前移动，补齐由于删除数据产生的空缺，然后 realloc 收缩空间，够直接！！！

### zipmapSet

实现思路：先 realloc 开辟足够的内存，然后 memove 移动后续元素，腾出空间更新 value，如果是新 key，则在最后直接追加，策略上简单粗暴。

### Summarization

虽然在实现思路上似乎对待内存简单粗暴，极其容易产生内存碎片，但是 Redis 在上层控制了 Zipmap 的元素个数，因此该数据结构即保持了简洁，也在限制范围内保持了高效。