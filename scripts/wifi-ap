#!/usr/bin/env bash
# Enables WiFi AP (Access Point)
set -e

IFACE="wlan0"      # set with -i or --iface
CONN="WIFI_AP"     # set with -c or --conn
SSID="MY_WIFI_AP"  # set with -s or --ssid
PASS="MY_PASSKEY"  # set with -p or --pass (required for mode 'up')
MODE="up"          # set with first positional arg (default 'up')

# https://medium.com/@Drew_Stokes/bash-argument-parsing-54f3b81a6a8f
while (( "$#" )); do
  case "$1" in
    -i|--iface|--interface)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        IFACE=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -c|--conn|--connection)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        CONN=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -s|--ssid)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        SSID=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -p|--pass|--password|--wpa|--key)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        PASS=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      MODE="$1"
      shift
      ;;
  esac
done

function wifi_up() {
  # https://forums.developer.nvidia.com/t/creating-wifi-ap-hotspot-on-jetson-nano/72982/10
  # https://github.com/dusty-nv/turbo2/blob/master/scripts/config-4-hostapd.sh

  if [ "$PASS" == "MY_PASSKEY" ]; then
    printf "Error:  must specify access point WPA Passkey with -p or --password\n"
    exit 1
  fi

  printf "Enabling Access Point (AP) mode on $IFACE with ssid=$SSID\n\n"
  set -x

  nmcli con add type wifi ifname $IFACE mode ap con-name $CONN ssid $SSID
  nmcli con modify $CONN 802-11-wireless.band bg
  nmcli con modify $CONN 802-11-wireless.channel 1
  nmcli con modify $CONN 802-11-wireless-security.key-mgmt wpa-psk
  nmcli con modify $CONN 802-11-wireless-security.psk $PASS
  nmcli con modify $CONN ipv4.method shared
  nmcli con up $CONN
}

function wifi_down() {
  # TODO reset to previous non-AP mode state
  nmcli con down $CONN
}

if [ "$MODE" == "up" ]; then
  wifi_up
elif [ "$MODE" == "down" ]; then
  wifi_down
else
  printf "Error:  unrecognized command $MODE  (valid options are:  up, down)"
  exit 1
fi
