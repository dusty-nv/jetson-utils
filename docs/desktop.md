# Desktop

Tips for desktop or remote desktop configuration or installing desktop GUI applications.

### Desktop Launcher

How do I create an icon on the Ubuntu desktop that launches an application?

* https://www.reddit.com/r/Ubuntu/comments/xyyb40/comment/irjcn0r/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button

```
[Desktop Entry]
   Encoding=UTF-8
   Version=1.0
   Type=Application
   Terminal=false
   Exec=/opt/chrome-linux/chrome
   Name=Chromium    
   Icon=/opt/chrome-linux/product_logo_48.png
```

Create this file as `~/.local/share/applications/chrome.desktop`

### Firefox (no snap)

* https://www.omgubuntu.co.uk/2022/04/how-to-install-firefox-deb-apt-ubuntu-22-04

```
sudo snap remove firefox && \
sudo apt remove -y firefox && \
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | sudo tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null && \
echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | sudo tee -a /etc/apt/sources.list.d/mozilla.list > /dev/null && \
echo '
Package: *
Pin: origin packages.mozilla.org
Pin-Priority: 1000

Package: firefox*
Pin: release o=Ubuntu
Pin-Priority: -1' | sudo tee /etc/apt/preferences.d/mozilla && \
sudo apt update && \
sudo apt install -y --no-install-recommends firefox 
```

### Chromium (no snap)

This is for x86 only:

```
cd /opt && \
sudo wget https://download-chromium.appspot.com/dl/Linux_x64?type=snapshots -O chrome-linux.zip && \
sudo unzip chrome-linux.zip && \
sudo rm chrome-linux.zip && \
sudo ln -s /opt/chrome-linux/chrome /usr/local/bin/chrome
```

Then start with `chrome-linux/chrome`

### NoMachine

* NoMachine - https://www.nomachine.com/documents
* [Linux x86_64 DEB Downloads](https://downloads.nomachine.com/download/?id=1)
* [Linux aarch64 DEB Downloads](https://downloads.nomachine.com/linux/?distro=Arm&id=30)

Depending on latest version for x86 or ARM:

```
wget https://download.nomachine.com/download/8.14/Linux/nomachine_8.14.2_1_amd64.deb && \
sudo dpkg -i nomachine_8.14.2_1_amd64.deb
```

```
wget https://download.nomachine.com/download/8.16/Arm/nomachine_8.16.1_1_arm64.deb && \
sudo dpkg -i nomachine_8.16.1_1_arm64.deb
```

Once running, you should see the port it exposes to connect to from clients:

```
NoMachine was configured to run the following services:
NX> 700 NX service on port: 4000
```

