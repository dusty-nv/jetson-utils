#!/usr/bin/env bash
# Persistently sets the mouse speed to the given DPI (default to 320)
# by creating udev rule under `/etc/udev/hwdb.d/50-mouse-dpi.hwdb`
#
#  https://askubuntu.com/a/11429964
#  https://github.com/systemd/systemd/blob/main/hwdb.d/70-mouse.hwdb
#
# This works on recent Ubuntu distros (22.04/24.04) whereas 
# changing it via xinput/gsettings does not have an effect.
#
# Run this with sudo priveleges:  `sudo mouse-speed <DPI>`
# Where DPI defaults to 320 (generally needed for 4K displays)
set -e

MOUSE_DPI="${1:-320}"
UDEV_RULE="/etc/udev/hwdb.d/50-mouse-dpi.hwdb"

printf "\nCreating $UDEV_RULE (MOUSE_DPI=$MOUSE_DPI)\n\n"
mkdir -p $(dirname $UDEV_RULE)

echo "mouse:*:name:*:*
  MOUSE_DPI=$MOUSE_DPI
" > $UDEV_RULE

cat $UDEV_RULE

set -x
systemd-hwdb update
udevadm trigger