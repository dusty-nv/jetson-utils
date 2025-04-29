# Samba (SMB)

These notes follow the installation of Samba server on Jetson that implements SMB/CIFS network file-sharing protocol and can be mounted from other clients running Linux, Windows, and Mac.

* https://phoenixnap.com/kb/ubuntu-samba
* https://linuxconfig.org/how-to-configure-samba-server-share-on-ubuntu-20-04-focal-fossa-linux
* https://computingforgeeks.com/install-and-configure-samba-server-share-on-ubuntu/

#### Install Samba

```bash
$ sudo apt-get install samba
$ samba -V
$ systemctl status smbd
```

Add user to samba password list (or substitute a different Linux username for Samba access)

```bash
$ sudo smbpasswd -a my_username
```

#### Configure Shares

Append this to the end of `/etc/samba/smb.conf`, where:

* `my_share` is what the mounted path in URL will be for clients
* `path` is the local path on your system to share
* `valid users` should replace `my_username` with `USER` from above

```bash
$ sudo nano /etc/samba/smb.conf
```
```
[my_share]
comment = Samba share directory
path = /mnt/nvme/share
read only = no
writable = yes
browseable = yes
guest ok = no
valid users = @my_username
```

And add this to the top under `[global]` (https://unix.stackexchange.com/a/103418)

```
[global]
   map archive = no
```

Validate the changes and restart

```bash
$ testparm
$ sudo systemctl restart smbd
```

#### Mount from Windows

```
\\$HOSTNAME.local\my_share
Username = WORKGROUP\my_username
```

#### Mount from Linux CLI

```bash
$ mount -t cifs -o username=my_username //JETSON_IP/my_share /mnt/jetson

# add mount to /etc/fstab
//JETSON_IP/my_share /mnt/jetson cifs credentials=/.sambacreds 0 0

cat /.sambacreds
username=my_username
password=password
domain=WORKGROUP

df -hT | grep cifs
```