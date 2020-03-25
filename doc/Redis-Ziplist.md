# Ziplist

Redis 中压缩列表并没有使用显示结构存储，而是采用原生内存进行存储与管理。

## 结构

|8 Bytes|1 Bytes|
|-------|-------|
|Header| Tail|

压缩列表初始长度：9 Bytes（8 Bytes：Header、1Byte：Tail）；
|4 Bytes|4 Bytes    |2 Bytes|1 Byte|
|-------|-----------|—————-—|------|
|Length |Tail Offset|Number |Tail  |
|11     |10         |0      |255   |

Header：（ZIPLIST_HEADER_SIZE）
-	Total length：Total length of a ziplist(Header + Entry + Tail);
-	Tail offset：The last entry offset in ziplist；
-	Element count：0

Header：Two 32 bit integers for the total bytes count and last item offset
Number：One 16 bit integer for the number of items field

Length：表示列表内存总长度；
Tail Offset：表示列表最后一个元素偏移量；
Number：表示列表中元素个数；

```
unsigned char* p = ziplist;
ZIPLIST_ENTRY_END  = p + *((uint32_t*)(zl)) - 1;   // 列表序列尾部（最后一个元素尾）
ZIPLIST_ENTRY_HEAD = p + size_of(uint32_t)*2 + 16; // 列表序列头部（第一个元素头）
ZIPLIST_ENTRY_TAIL = p
```
缺点：
1. 每次操作都设计内存重新分配？

问题：
1. 为什么选择：245 而不是 255？因为 255 已经作为 Ziplist 的结尾标志符，所以这里选择 254.

## zlentry

```
/* We use this function to receive information about a ziplist entry.
 * Note that this is not how the data is actually encoded, is just what we
 * get filled by a function in order to operate more easily. */
typedef struct zlentry {
    unsigned int prevrawlensize; /* Bytes used to encode the previous entry len*/
    unsigned int prevrawlen;     /* Previous entry len. */
    unsigned int lensize;        /* Bytes used to encode this entry type/len.
                                    For example strings have a 1, 2 or 5 bytes
                                    header. Integers always use a single byte.*/
    unsigned int len;            /* Bytes used to represent the actual entry.
                                    For strings this is just the string length
                                    while for integers it is 1, 2, 3, 4, 8 or
                                    0 (for 4 bit immediate) depending on the
                                    number range. */
    unsigned int headersize;     /* prevrawlensize + lensize. */
    unsigned char encoding;      /* Set to ZIP_STR_* or ZIP_INT_* depending on
                                    the entry encoding. However for 4 bits
                                    immediate integers this can assume a range
                                    of values and must be range-checked. */
    unsigned char *p;            /* Pointer to the very start of the entry, that
                                    is, this points to prev-entry-len field. */
} zlentry;
```

- 说明
- prevrawlensize：if p[0] < 254，so prevrawlensize = 1；
                  if p[0] = 254, so prevrawlensize = 5;
- prevrawlen: if prevrawlensize == 1, so prevrawlen = p[0];
              if prevrawlensize == 5, so prevrawlen = *((uint32_t*)(p+1));
- lensize: 与encoding混合存储；
- len: 与encoding混合存储；
- headersize： 实际序列化存储中并不存储该项；
- encoding: 编码信息和长度信息存储在一起；字符串存储：
            整数存储：

## 编码

Ziplist 会尝试将每个列表中元素进行数字化处理，这样可以节省表示数字的字符串的存储空间。

限制条件：只有长度小于 32 Bytes 的字符串才有数字化的资格。

|Macro|十六进制|十进制|二进制|
|:--------------|----|     ------|---------|
|ZIP_INT_IMM_MIN|0xf1|241        |1111 0001|
|ZIP_INT_IMM_MAX|0xfd|253        |1111 1101|
|INT8_MIN       |    |-128       ||
|INT8_MAX       |    | 127       ||
|INT16_MIN      |    |-32768     ||
|INT16_MAX      |    | 32767     ||
|INT24_MIN      |    |-8388608   ||
|INT24_MAX      |    | 8388607   ||
|INT32_MIN      |    |-2147483648||
|INT32_MAX      |    | 2147483647||

```
if (value >= 0 && value <= 12) {
	*encoding = ZIP_INT_IMM_MIN + value;
} else if (value >= INT8_MIN && value <= INT8_MAX) {
	*encoding = ZIP_INT_8B;
} else if (value >= INT16_MIN && value <= INT16_MAX) {
	*encoding = ZIP_INT_16B;
} else if (value >= INT24_MIN && value <= INT24_MAX) {
	*encoding = ZIP_INT_24B;
} else if (value >= INT32_MIN && value <= INT32_MAX) {
	*encoding = ZIP_INT_32B;
} else {
	*encoding = ZIP_INT_64B;
}
```

## 压缩存储

| 十六进制 | 十进制 |       二进制      |
|:--------|:-----:|------------------:|
|0X30     |48     |0011 0000|
|0X3F     |63     |0011 1111|
|0XC0     |192    |1100 0000|
|0X3FFF   |16383  |0011 1111 1111 1111|

### 字符串压缩存储

如果数据被编码为字符串进行存储，则编码字段可能会与数据长度字段混合存储，混合字段长度范围：[1、2、5]。

- if len <= 0X3F,   p[0] = ZIP_STR_06B | len；
- if len <= 0X3FFF, p[0] = ZIP_STR_14B | ((len >> 8) & 0x3f);
                    p[1] = len & 0xff;
- if len > 0X3FFFF, p[0] = ZIP_STR_32B;
                    p[1] = (len >> 24) & 0xff;
                    p[2] = (len >> 16) & 0xff;
            		p[3] = (len >> 8) & 0xff;
            		p[4] = len & 0xff;

- #define ZIP_STR_MASK 0xc0    // Binary: 1100 0000
- #define ZIP_STR_06B (0 << 6) // Binary: 0000 00000
- #define ZIP_STR_14B (1 << 6) // Binary: 0100 00000
- #define ZIP_STR_32B (2 << 6) // Binary: 1000 00000


### 数字化压缩存储

如果数据被压缩为整数进行存储，则编码字段独占以为，且数字编码本身表示了长度，因此不需要额外存储数据长度信息。

- p[0]   = encoding // 首位存储字符串编码格式；
- p[1-4] = len      // 后四位存储字符串长度；

- #define ZIP_INT_MASK 0x30         // Binary: 0011 0000
- #define ZIP_INT_16B (0xc0 | 0<<4) // Binary: 1100 0000
- #define ZIP_INT_32B (0xc0 | 1<<4) // Binary: 1101 0000
- #define ZIP_INT_64B (0xc0 | 2<<4) // Binary: 1110 0000
- #define ZIP_INT_24B (0xc0 | 3<<4) // Binary: 1111 0000
- #define ZIP_INT_8B   0xfe         // Binary: 1111 1110

## 操作细节

- 删除元素：如果在尾部删除元素，则不需要操作其他数据，如果在中间删除元素，则需要移动（MemMove）后续元素。
- 插入元素：内存不够，需要重新分配（Realloc）内存。



