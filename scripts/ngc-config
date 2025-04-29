#!/usr/bin/env bash
# Download/install the NGC CLI tools
# https://ngc.nvidia.com/setup/installers/cli
set -e

if [ "$1" == "install" ]; then
  INSTALL_DIR="${2:-/opt/nvidia/ngc}"
  BIN_DIR="${3:-/usr/local/bin}"
  
  if [ "$(uname -m)" == "aarch64" ]; then
    NGC_URL="https://ngc.nvidia.com/downloads/ngccli_arm64.zip"
  else
    NGC_URL="https://ngc.nvidia.com/downloads/ngccli_linux.zip"
  fi

  printf "\nInstalling NGC CLI under:  $INSTALL_DIR\n"
  printf "Downloading NGC CLI from:  $NGC_URL\n\n"

  rm -rf $INSTALL_DIR || true
  mkdir -p $INSTALL_DIR
  cd $INSTALL_DIR

  wget --content-disposition $NGC_URL
  unzip *.zip
  rm *.zip
  
  chmod u+x ngc-cli/ngc
  find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5
  ln -sf $(pwd)/ngc-cli/ngc $BIN_DIR/ngc

  printf "\nInstalled NGC CLI under: $INSTALL_DIR"
  printf "\nInstalled NGC CLI under: $BIN_DIR/ngc\n\n"

  ngc --version
else
  printf "\nInvalid command $1 (valid options are:  'install')\n\n"
  exit 1
fi
