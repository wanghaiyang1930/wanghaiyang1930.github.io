
# Linux VPN Instruction

---

## Mihomo

```TEXT
Download [mihomo-linux-amd64-v3-v1.19.27.gz](https://github.com/MetaCubeX/mihomo/releases/download/v1.19.27/mihomo-linux-amd64-v3-v1.19.27.gz). 
Another address [Minhomo](https://github.com/MetaCubeX/mihomo/releases).

gunzip mihomo-linux-amd64-v3-v1.19.27.gz
```

## geoip

```TEXT
Download [geoip.metadb](https://github.com/MetaCubeX/meta-rules-dat/releases/tag/latest).
```

## Service
```BASH
nohup ./mihomo -f ./config.yaml &
```

## Terminal
```TEXT
Add proxy to .bashrc.
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export ALL_PROXY="socks5://127.0.0.1:7891"
```

## Chrome
```BASH
google-chrome --proxy-server="http://127.0.0.1:7890"
```

## Test

```BASH
curl ipinfo.io
http://127.0.0.1:9090/proxies/
```
