#!/usr/bin/env bash
# This script installs all the files in this directory
# to /usr/local/bin by default. These scripts should all have
# the +x executable bit set (`chmod +x script.sh`) and start
# with the shebang sequence like (`#!/usr/bin/env bash`)
#
# When installing to a system-wide directory like by default,
# it typically requires sudo privileges to run this install.sh
#
# TODO add dev mode like `pip install -e` using symlinks instead
# of copying the scripts for quicker editing during development.
SOURCE_DIR="$(dirname "$(readlink -f "$0")")"
INSTALL_DIR="${1:-/usr/local/bin}"

set -e
cd $SOURCE_DIR 

# *.{sh,py}
for file in *; do

  if [ $file == "install.sh" ]; then
    continue
  fi

  if [ ! -f "$file" ]; then
    continue
  fi

  src="$SOURCE_DIR/$file"
  dst="$INSTALL_DIR/${file%.*}"

  printf "Installing $src -> $dst\n"

  cp $src $dst
  chmod +x $dst
done
