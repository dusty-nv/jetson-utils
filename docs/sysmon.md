# System Monitoring

Various system monitors, log viewers, and reporting utilities like top with additional information or utilization metrics.

```
sudo apt install -y \
    nvtop htop ctop \
    bmon tcptrack iftop \
    tmux terminator screen
```

#### nvidia-smi

`nvidia-smi` is included with JetPack 6 with limited fields reported.

```
watch -n 1 nvidia-smi -l 1
nvidia-smi pmon -i 0
```

#### btop (with GPU)

This is as used on x86 (not yet tested on Jetson)

```
git clone https://github.com/aristocratos/btop

docker run --gpus all -it --rm \
  -v $PWD/btop:/btop \
  --workdir /btop \
  nvcr.io/nvidia/pytorch:24.10-py3 \
    make
    
sudo chown $USER btop/bin/btop && \
sudo chmod +x btop/bin/btop && \
sudo cp btop/bin/btop /usr/local/bin
```

#### ctop

* https://github.com/bcicen/ctop
* https://github.com/jesseduffield/lazydocker

```
sudo wget https://github.com/bcicen/ctop/releases/download/v0.7.7/ctop-0.7.7-linux-amd64 -O /usr/local/bin/ctop
sudo chmod +x /usr/local/bin/ctop
```

#### tmux

The [`tmux-run`](/scripts/tmux-run) from jetson-utils will launch the given commands in parallel and in their own terminal panel using tmux / terminator:

```
tmux-run 'htop' 'nvtop' 'bmon' 'ctop'
```

Each individual command to run should be contained in quotes independently.

* https://tmuxcheatsheet.com/
* works over SSH (`Ctrl+B` is command mode)

#### goaccess

* For live viewing of server access logs
* https://goaccess.io/get-started

```
sudo apt install -y goaccess
goaccess -c /var/log/nginx/access.log
```

The log format to select for nginx is `NCSA Combined Log Format`

```
goaccess access.log -o /var/www/html/report.html --log-format=COMBINED --real-time-html
```