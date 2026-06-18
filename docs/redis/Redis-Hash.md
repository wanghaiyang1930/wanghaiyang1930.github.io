
Redis Hash 结构采用两种数据结构：
	1）Zip List；
	2）Hash Table；
Redis Hash 存储结构何时切换：
1）第一次创建一个 Hash 结构是指定：Zip List；
2）如果添加 Field 或 Value 长度大于 64 Bytes；
3）如果 Zip List 中存储 Entry 大于 512 个；

- Redis Hash 存储结构如何切换：
1. 创建一个 Hash Table 结构对象；
2. 遍历 Zip List，使用 Iterator；
3. 将每个 Key Value Obj 转换成 SDS 结构， 然后 Add 到 HT 对象中；
4. 释放迭代器 Iterator 与 Zip List，完成转换；

HSCAN [Document](http://doc.redisfans.com/key/scan.html)
