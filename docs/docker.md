# Docker

Various commands for managing docker images.

### Disk Cleanup

``` bash
sudo docker system prune  # use -a to prune all (not just dangling)
sudo docker rmi -f $(sudo docker images | grep "<none>" | awk "{print \$3}")
```

* https://forums.docker.com/t/command-to-remove-all-unused-images/20/5
* https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes

### Remove Images by Filter

``` bash
sudo docker rmi -f $(sudo docker images --filter=reference="*CONTAINER*:*TAG*" -q)
sudo docker rmi -f $(sudo docker images --filter=since=CONTAINER:TAG -q)
sudo docker rmi -f $(sudo docker images --filter=before=CONTAINER:TAG -q)
```

* https://stackoverflow.com/a/47265229
* https://docs.docker.com/engine/reference/commandline/images/#filter

### Change Default Runtime

Edit `/etc/docker/daemon.json` → add `"default-runtime": "nvidia"` 

* https://github.com/dusty-nv/jetson-containers#docker-default-runtime

### Change Data Root

Edit `/etc/docker/daemon.json` → add `"data-root": "/path/to/data"`

``` json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia",
    "data-root": "/mnt/nvme/docker"
}
```

Edit `/lib/systemd/system/docker.service` → add `Environment=DOCKER_TMPDIR=/path/to/tmp`

```
[Service]
Type=notify
# the default is not to use systemd for cgroups because the delegate issues still
# exists and systemd currently does not support the cgroup feature set required
# for containers run by docker
Environment=DOCKER_TMPDIR=/mnt/nvme/docker/tmp
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutSec=0
RestartSec=2
Restart=always
``` bash

* https://www.ibm.com/docs/en/z-logdata-analytics/5.1.0?topic=compose-relocating-docker-root-directory
* https://github.com/spotify/docker-client/issues/1028#issuecomment-392803461

### Saving Container Image

``` bash
sudo docker save myimage:latest | gzip > myimage_latest.tar.gz
```

### Load Container Image

``` bash
docker load > myimage_latest.tar.gz
```

### Daemon Status

``` bash
sudo systemctl status docker.service
sudo journalctl -fu docker.service   # view dockerd logfile
sudo docker info
```

### Docker Compose

`docker-compose` is deprecated in leiu of `docker compose` (the later should be used moving forward)

```
sudo apt install -y docker-compose-v2
```

```
docker compose up --build
```

The previous standalone `docker-compose` could be installed like this:

```
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
sudo chmod +x /usr/local/bin/docker-compose && \
docker-compose --version
```

#### Execute on Remote Hosts

* https://www.docker.com/blog/how-to-deploy-on-remote-docker-hosts-with-docker-compose/

```
DOCKER_HOST=“ssh://user@remotehost” docker-compose up -d
```

