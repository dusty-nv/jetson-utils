#!/usr/bin/env bash
# Sorts directories recursively by size (in MB)
# This is like a CLI version of 'Disk Usage Analyzer'
#
# Usage: disk-sort.sh <DIR> <N>
# Where: DIR is scanned recursively (defaults to current working dir)
#        N is the largest-N directories to print (25 by default)
DIR="${1:-.}"
N="${2:-25}"

printf "\nScanning $DIR for the largest $N directories...\n\n"

du -m $DIR | sort -nr | head -n $N

# This variation prints the unsorted size of all dirs as they are scanned
# du -m . | tee /dev/tty | sort -nr

# Other variants that work with files instead of just folders:
#du -ah
#du -sh ./*
#du -cksh * | sort -rh

# See these links for more info:
# https://superuser.com/questions/162749/how-to-get-the-summarized-sizes-of-directories-and-their-subdirectories
# https://unix.stackexchange.com/questions/88065/sorting-files-according-to-size-recursively
# https://unix.stackexchange.com/questions/67806/how-to-recursively-find-the-amount-stored-in-directory