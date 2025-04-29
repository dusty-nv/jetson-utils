#!/usr/bin/env bash
# Filesystem benchmarking for sequential R/W and random R/W.
# First install these dependencies: `sudo apt install cli fio`
#
# Usage:  disk-bench <DEVICE> <LOGS>
# Where:  DEVICE is the file, directory, or storage device
#         LOGS is directory where benchmark logs are saved
#         These default to the current path and logs/
set -e

TMP_DIR="/tmp/benchmarks"
mkdir $TMP_DIR || true

: "${FILE_SIZE:=1024}" # block size to allocate in the filesystem (in MB)

function benchmark() {
	local FILE_PATH="${1:-$PWD}"  # the file, directory, or device being benchmarked
	local LOG_DIR="${2:-$PWD/logs}"
	
	mkdir -p $LOG_DIR || true
	
	if [ -d "$FILE_PATH" ]; then
		FILE_TEMP="$FILE_PATH/benchmark.temp"
	else  #if [ -f "$FILE_PATH" ]; then
		FILE_TEMP="$FILE_PATH"
	fi

	if [ ! -f "$FILE_PATH" ]; then
		echo "allocating $FILE_TEMP ($FILE_SIZE MB)"
		dd if=/dev/urandom of=$FILE_TEMP bs=1M count=$FILE_SIZE
		freeStorage=true
	fi

	# sequential read
	echo "[global]
	name=seq-read
	time_based
	ramp_time=5
	runtime=30
	readwrite=read
	bs=256k
	ioengine=libaio
	direct=1
	numjobs=1
	iodepth=32
	group_reporting=1
	[nvme]
	filename=$FILE_TEMP" > $TMP_DIR/seq-read.fio
	
	cat $TMP_DIR/seq-read.fio
	fio $TMP_DIR/seq-read.fio | tee $LOG_DIR/seq-read.txt
	echo "done sequential read ($FILE_PATH)"

	# random read
	echo "[global]
	name=rand-read
	time_based
	ramp_time=5
	runtime=30
	readwrite=randread
	random_generator=lfsr
	bs=4k
	ioengine=libaio
	direct=1
	numjobs=16
	iodepth=16
	group_reporting=1
	[nvme]
	new_group
	filename=$FILE_TEMP" > $TMP_DIR/rand-read.fio

	cat $TMP_DIR/rand-read.fio
	fio $TMP_DIR/rand-read.fio 2>&1 | tee $LOG_DIR/rand-read.txt
	echo "done random read ($FILE_PATH)"

	# sequential write
	echo "[global]
	name=seq-write
	readwrite=write
	bs=1M
	ioengine=libaio
	direct=1
	numjobs=1
	iodepth=32
	group_reporting=0
	[nvme]
	filename=$FILE_TEMP" > $TMP_DIR/seq-write.fio

	cat $TMP_DIR/seq-write.fio
	fio $TMP_DIR/seq-write.fio 2>&1 | tee $LOG_DIR/seq-write.txt
	echo "done sequential write ($FILE_PATH)"

	# random write
	echo "[global]
	name=seq-write
	readwrite=write
	bs=1M
	ioengine=libaio
	direct=1
	numjobs=1
	iodepth=32
	group_reporting=0
	[nvme]
	filename=$FILE_TEMP" > $TMP_DIR/rand-write.fio

	cat $TMP_DIR/rand-write.fio
	fio $TMP_DIR/rand-write.fio 2>&1 | tee $LOG_DIR/rand-write.txt
	echo "done random write ($FILE_PATH)"

	# free previous storage
	if [ -n "$freeStorage" ]; then
		echo "deleting $FILE_TEMP"
		rm $FILE_TEMP
	fi

	echo ""
	echo "# SEQUENTIAL READ"
	tail -n 2 $LOG_DIR/seq-read.txt

	echo ""
	echo "# SEQUENTIAL WRITE"
	tail -n 2 $LOG_DIR/seq-write.txt

	echo ""
	echo "# RANDOM READ"
	tail -n 2 $LOG_DIR/rand-read.txt

	echo ""
	echo "# RANDOM WRITE"
	tail -n 2 $LOG_DIR/rand-write.txt

	echo ""
	echo "Done benchmarking $FILE_PATH"
	echo "Logs written to $LOG_DIR"
}

#for var in "$@"
#do
#    echo "$var"
#done

if [ $# -eq 0 ]; then
  benchmark
else
	for path in "$@"
	do
		  benchmark "$path"
	done
fi

