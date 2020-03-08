# List

### List

Redis List 底层采用 QuickList 实现，QuickList 并非简单的双向链表，其优化前驱后继指针的内存空间，同时防止过多的不连续内存碎片产生。

### Operation

- LPUSH Key value [value] : Push into a list;
- LPUSHX key value : Push if the list is exist;
- LPOP Key : Pop an element at head;
- LREM key count value : Complex
- LSET key index value :
- LRTIM key start stop :
- LRANGE key start stop :

- LLEN key
- LINDEX key index

- BLPOP key [key] timeout
- BRPOP key [key] timeout
- BRPOPLPUSH source destination timeout
- RPOOPLPUSH source destination

- LINSERT key BEFORE|AFTER privo value

- RPUSH key value [value] :
- RPUSHX key value :
- RPOP