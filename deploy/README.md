# letsencrypt certificates

```
sudo certbot certonly -d rti.livetok.io --server https://acme-v02.api.letsencrypt.org/directory
sudo chmod 0755 /etc/letsencrypt/{live,archive}
sudo setcap CAP_NET_BIND_SERVICE=+eip /home/gustavogb/.local/share/uv/python/cpython-3.13.5-linux-x86_64-gnu/bin/python3.13
```
