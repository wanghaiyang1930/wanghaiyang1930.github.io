# Quicklist

## Linkedlist

优点：
- 支持双向操作（push pop）；
- 复杂度低，实现简单；

缺点：
- 存储消耗大，需要额外存储两个节点指针（Prev、Next）；
- 存储不连续，每个节点单独开辟内存，容易产生内存碎片；

## Ziplist

优点：
- 连续内存，存储效率高；

缺点：
- 操作复杂，插入和删除需要频繁申请和释放内存，当列表较长时，有可能导致大量的数据拷贝；

## Quicklist

结合 Linkedlist 与 Ziplist，整体上采用 Linkedlist 存储，而每个 Linkedlist 的 Node 又是一个 Ziplist。

### QuicklistNode

```
/* Node, quicklist, and Iterator are the only data structures used currently. */

/* quicklistNode is a 32 byte struct describing a ziplist for a quicklist.
 * We use bit fields keep the quicklistNode at 32 bytes.
 * count: 16 bits, max 65536 (max zl bytes is 65k, so max count actually < 32k).
 * encoding: 2 bits, RAW=1, LZF=2.
 * container: 2 bits, NONE=1, ZIPLIST=2.
 * recompress: 1 bit, bool, true if node is temporarry decompressed for usage.
 * attempted_compress: 1 bit, boolean, used for verifying during testing.
 * extra: 10 bits, free for future use; pads out the remainder of 32 bits */
typedef struct quicklistNode {
    struct quicklistNode *prev;
    struct quicklistNode *next;
    unsigned char *zl;
    unsigned int sz;             /* ziplist size in bytes */
    unsigned int count : 16;     /* count of items in ziplist */
    unsigned int encoding : 2;   /* RAW==1 or LZF==2 */
    unsigned int container : 2;  /* NONE==1 or ZIPLIST==2 */
    unsigned int recompress : 1; /* was this node previous compressed? */
    unsigned int attempted_compress : 1; /* node can't compress; too small */
    unsigned int extra : 10; /* more bits to steal for future usage */
} quicklistNode;
```

Encoding：Quicklist 的数据项采用两种存储方式：
- RAW：一个Ziplist结构；
- LZF：一个4+N的结构，存储压缩字节流与其长度；

Encoding 如何生效：

### Quicklist

```
/* quicklist is a 40 byte struct (on 64-bit systems) describing a quicklist.
 * 'count' is the number of total entries.
 * 'len' is the number of quicklist nodes.
 * 'compress' is: -1 if compression disabled, otherwise it's the number
 *                of quicklistNodes to leave uncompressed at ends of quicklist.
 * 'fill' is the user-requested (or default) fill factor.
 * 'bookmakrs are an optional feature that is used by realloc this struct,
 *      so that they don't consume memory when not used. */
typedef struct quicklist {
    quicklistNode *head;
    quicklistNode *tail;
    unsigned long count;        /* total count of all entries in all ziplists */
    unsigned long len;          /* number of quicklistNodes */
    int fill : QL_FILL_BITS;              /* fill factor for individual nodes */
    unsigned int compress : QL_COMP_BITS; /* depth of end nodes not to compress;0=off */
    unsigned int bookmark_count: QL_BM_BITS;
    quicklistBookmark bookmarks[];
} quicklist;

```

疑问：
- bookmark_count 只有 4 Bits，最大值127，能够吗？

### quicklistLZF

```
/* quicklistLZF is a 4+N byte struct holding 'sz' followed by 'compressed'.
 * 'sz' is byte length of 'compressed' field.
 * 'compressed' is LZF data with total (compressed) length 'sz'
 * NOTE: uncompressed length is stored in quicklistNode->sz.
 * When quicklistNode->zl is compressed, node->zl points to a quicklistLZF */
typedef struct quicklistLZF {
    unsigned int sz; /* LZF size in bytes*/
    char compressed[];
} quicklistLZF;
```

在 QuicklistNode 中，如果存储压缩数据，则：QuicklistNode.sz >= size_of(quicklistLZF) + quicklistLZF.sz；

压缩存储是指：压缩 Ziplist，然后将其当作一个 LZF 进行存储，如果某个 QuicklistNode 采用压缩存储，那么关于其内部数据在操作（查找、更新、删除）过程中会频繁解压缩（考验压缩解压缩效率）。

触发压缩的条件：
- Quicklist 开启压缩开关：compress != 0; 
- And
- Quicklist 中被压缩的 Node 个数少于总个数的一半；（不准确）
- QuicklistNode 中数据长度超过 MIN_COMPRESS_BYTES（48）字节；

### quicklistBookmark

```
/* Bookmarks are padded with realloc at the end of of the quicklist struct.
 * They should only be used for very big lists if thousands of nodes were the
 * excess memory usage is negligible, and there's a real need to iterate on them
 * in portions.
 * When not used, they don't add any memory overhead, but when used and then
 * deleted, some overhead remains (to avoid resonance).
 * The number of bookmarks used should be kept to minimum since it also adds
 * overhead on node deletion (searching for a bookmark to update). */
typedef struct quicklistBookmark {
    quicklistNode *node;
    char *name;
} quicklistBookmark;
```

Note:
- 其中name表示字符串，而不是字节流，在使用时采用strcmp进行等比较；

Question：
- nmae 中存的是啥？