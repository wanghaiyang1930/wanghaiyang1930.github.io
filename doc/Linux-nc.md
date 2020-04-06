
# nc

It can open TCP connections, send UDP packets, listen on arbitrary TCP and UDP ports, do port scanning, and deal with both IPv4 and IPv6. 

总之，能干的事情不少。

## 监听 UDP 端口

- 监听 UDP 端口：1234，默认ID：0.0.0.0

`nc -u -l 1234`


- 监听 UDP 端口：1234，监听地址：127.0.0.1（本地有效）
`nc -u -l 127.0.0.1 1234`


## 建立一个 UDP 链接

- 与 host.example.com 53 建立一个UDP链接
`nc -u host.example.com 53`


## 监听 TCP 端口

- 监听：0.0.0.0:8080
`nc -l 8080`


- 监听：127.0.0.1:8080
`nc -l 127.0.0.1 8080`


## 建立一个 TCP 链接

- 建立与127.0.0.1:8080的TCP链接
`nc 127.0.0.1 8080`

- 使用本地端口：31337 链接 host.example.com:42 且5秒内超时
`nc -p 31337 -w 5 host.example.com 42`


## 端口扫描：TCP

- 扫描 TCP 20～100 端口（2秒超时）
`nc -v -w 2 home -z 21-100`

## 端口扫描：UDP

- 扫描 UDP 20～100 端口（2秒超时）
`nc -u -v -w 2 ip -z 21-100`