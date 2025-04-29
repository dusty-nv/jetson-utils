# NFS

Setup NFS file-sharing server for mounting directories from other LAN devices running Linux (not Windows unlike Samba/SMB)

NFS is unencrypted so is only recommended for LAN use unless a VPN is also used, but may have higher performance for large file transfers.  Other options include [Samba](samba.md) (SMB) for Windows compatability and [SSHFS](#sshfs).

* https://ubuntu.com/server/docs/network-file-system-nfs

### Install NFS Driver

```
sudo apt install -y nfs-kernel-server && \
sudo systemctl start nfs-kernel-server.service
```

### Export Network Shares

Add the following to `/etc/exports` (replace `/mnt/nvme` with the server path you want to share)

```
/mnt/nvme *(rw,sync,no_subtree_check,no_root_squash)
```

(other than the exported directory itself, symbolic links on the server can't be followed by clients)

```
sudo exportfs -a && \
systemctl status nfs-server
```

### Mount from Client (Linux)

These commands will install the NFS client on other Linux devices and mount the server's NFS share, appearing like other local hard drives attached to the client's system, but over the network. 

```
sudo apt install -y nfs-common && \
sudo mkdir /mnt/nfs && \
sudo mount $NFS_HOST:/mnt/nvme /mnt/nfs
```

Replace `$NFS_HOST` with the hostname of the machine exporting the NFS share or `export NFS_HOST=my-jetson` first.

### Mount Google Drive (rclone)

[`rclone`](https://github.com/rclone/rclone) is a recommended utility for mounting Google Drives (GDrive) from Linux:  https://rclone.org/drive/

[`roundsync`](https://github.com/newhinton/Round-Sync) is a frontend available for Linux x86, ARM, and Android:  https://roundsync.com/

### SSHFS

Unlike NFS, SSHFS includes authentication and encryption as it is built upon SSH.  As such, it is easy to install or comes pre-configured on many Linux instances.  

* https://www.reddit.com/r/linuxadmin/comments/17ur9vb/why_use_sshfs_over_nfs/

```
sudo apt-get install sshfs
sudo mkdir -p /mnt/sshfs
sudo sshfs -o allow_other,default_permissions SSH_USER@SSH_HOST:/mnt/nvme /mnt/sshfs
```

Replace `SSH_USER` and `SSH_HOST` with the login and hostname of the server.