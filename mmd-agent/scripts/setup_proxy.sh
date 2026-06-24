#!/usr/bin/env bash
# 持久化代理配置：在所有 terminal/screen 启动后 source 即可
export HTTP_PROXY_VAL="http://YOUR_HTTP_PROXY:PORT"
export HTTPS_PROXY_VAL="http://YOUR_HTTP_PROXY:PORT"
export NO_PROXY_VAL="10.0.0.0/8,100.96.0.0/12,.your-internal-domain.local,.your-internal-domain.local,.your-internal-domain.local"
export http_proxy="$HTTP_PROXY_VAL"
export https_proxy="$HTTPS_PROXY_VAL"
export no_proxy="$NO_PROXY_VAL"
export HTTP_PROXY="$HTTP_PROXY_VAL"
export HTTPS_PROXY="$HTTPS_PROXY_VAL"
export NO_PROXY="$NO_PROXY_VAL"
