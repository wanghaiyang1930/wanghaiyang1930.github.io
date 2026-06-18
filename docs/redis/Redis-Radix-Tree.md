# Radix Tree

基数树：主要用于将 Long int（ID） 与指针进行映射（如：IDR），建立索引稀疏数组，存储效率高，查询快速。

《The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases》ART在内存数据库方面的应用。

Trie Tree 可以看作特化（基为26）的 Raidx Tree，一般用于将字符串映射到对象。
#

- 1 Bit 判断，树的高度过高，非叶子结点过多；
- N Bit 判断，树节点的字节点槽过多，增大节点的体积；
- 推荐判断位：2 Bit / 4 Bit

Trie Tree、Radix Tree在某些应用场景可以帮助我们节省内存使用空间，但也有明显的局限性，比如这类树结构无法适用于所有的数据类型，目前来看主要适用于能够用string字符等可作为表达式查询key的场景。

## Radix Tree

### Node

```
typedef struct raxNode {
    uint32_t iskey:1;     /* Does this node contain a key? */
    uint32_t isnull:1;    /* Associated value is NULL (don't store it). */
    uint32_t iscompr:1;   /* Node is compressed. */
    uint32_t size:29;     /* Number of children, or compressed string len. */
    unsigned char data[]; /* Reference */
} raxNode;
```
> An empty node:

Length: |                            4 Bytes                            |
Memory: |0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|
Value:  |0|0|0|                            0                            |

Node 节点分为压缩节点与非压缩节点，压缩节点只有一个指针（Children有可能多个），非压缩节点则包含多个指针。

### Radix

```
typedef struct rax {
    raxNode *head;
    uint64_t numele;
    uint64_t numnodes;
} rax;
```

整体上来看，多个节点之间采用树状形式进行存储（Node挂载），而在Node之内则采用连续内存进行存储布局（自定义格式）。

### data



### raxNodeCurrentLength

> 每个节点都是一块连续的内存，其长度由下面几个部分组成：
> Length = sizeof(raxNode) + size + Padding(size) + n*sizeof(raxNode*) + N*sizeof(void*)；

- size：每个字符站用一个Byte，这样无论是压缩节点、还是未压缩节点占用的公共前缀的空间是一样的；
- padding：字节对齐，data数据的空间按照void*的大小对齐，有可能是 8 Bytes or 4 Bytes （sizeof(void*)）；
- n：如果是节点被压缩：只存储一个字节点的指针，如果节点未被压缩：则存储每个字节点的指针（节点个数：size）；
- N：如果该节点是一个Key，则 N=1，否则 N=0，有些Key没有Value，那么就不会存储Value的指针（节约空间）；

### raxAddChild

### raxCompressNode

> 压缩节点：
- Alloc 一个空的字节点；
- Realloc 一个调整长度后的节点；
- 进行数据填充、SetData、挂载空的字节点。

### raxGenericInsert

1. 通过前缀匹配来寻找Key在基数树中的位置，压缩节点采用前缀匹配、非压缩节点采用二叉树查找；
2. 如果正好匹配到完整的节点上，则就地更新数据；
3. 如果正好匹配到一个压缩节点的中间，则需要将压缩节点拆解开；

### example

> A new radix tree

rax->head pointer（树中只有一个头节点，其内存布局如下所示：）

Length: |                            4 Bytes                            |
Memory: |0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|0 1 2 3 4 5 6 7|
Value:  |0|0|0|                            0                            |




