# Dict

##

当哈希表的冲突率过高时链表会很长，这时查询效率就会变低，所以有必要进行哈希表扩展，而如果哈希表存放的键值对很少的时候把size设的很大，又会浪费内存，这时就有必要进行哈希表收缩。这里扩展和收缩的过程，其实就是rehash的过程。