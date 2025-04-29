#!/usr/bin/env bash
# Launches multiple processes in a tmux terminal window 
# that get split into multiple panes. By default, tiled  
# layout will be used for an equally-distributed grid.
#
#   tmux_run './foo.sh' 'bar.py --xyz'
#
# For vertical or horizontal layouts, use the -v or -h flags:
#
#   tmux_run -h './foo.sh' './bar.sh'
#   tmux_run --vertical './foo.sh' 'bar.py --xyz'
#
# Always encapsulate the process launch commands in quotes.
# To manage multiple terminals, specify unique --session names.
#
# This wrapper requires tmux to be installed with `apt install tmux`
# See here for more information about installing and using tmux:
#
# https://unix.stackexchange.com/a/149729/47116
# https://man7.org/linux/man-pages/man1/tmux.1.html
#
set -e

COMMANDS=()
LAYOUT="tiled"
SESSION="multi"
WINDOW="global"
INSTALL="/usr/local/bin/tmux_run"
SOURCE="$(readlink -f "${BASH_SOURCE[0]}")"
REMAIN_ON_EXIT="off"

# https://medium.com/@Drew_Stokes/bash-argument-parsing-54f3b81a6a8f
while (( "$#" )); do
  case "$1" in
    -h|--horizontal)
      LAYOUT="even-horizontal"
      shift
      ;;
    -v|--vertical)
      LAYOUT="even-vertical"
      shift
      ;;
    --tile|--tiled)
      LAYOUT="tiled"
      shift
      ;;
		-i|-it|-k|--keep)
      REMAIN_ON_EXIT="on"
      shift
      ;;
    -s|--session)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        SESSION=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -w|--window)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        SESSION=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --install)
      ln -s $SOURCE $INSTALL
      chmod +x $INSTALL
      echo "installed under $INSTALL"
      exit
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      COMMANDS+=("$1")
      shift
      ;;
  esac
done

# set positional arguments in their proper place
echo "Launching commands:"

for i in "${!COMMANDS[@]}"
do
	STATUS_MSG='[$BASHPID] exited (return code $?)'
	COMMANDS[$i]="${COMMANDS[$i]} ; echo \"$STATUS_MSG\" ; sleep 1"
  	
	if [ $REMAIN_ON_EXIT = "off" ]; then
		COMMANDS[$i]="${COMMANDS[$i]} ; exit"
  fi
  
	echo "  * ${COMMANDS[$i]}"
done

# remove any inactive panes from previous runs
if [ -n "$TMUX_PANE" ]; then
	tmux kill-pane -a -t $TMUX_PANE || true
fi

# create new session in detached mode
tmux new-session -s $SESSION -n $WINDOW -d || true

# invoke the first process command
tmux send-keys -t $SESSION:$WINDOW.0 "${COMMANDS[0]}" C-j

# create split panes for the remaining processes
COMMANDS=( "${COMMANDS[@]:1}" )

for i in "${!COMMANDS[@]}"
do
	#echo "launching $i '${COMMANDS[$i]}'"
	tmux split-window -v
	#tmux set-option -p remain-on-exit $REMAIN_ON_EXIT
	tmux send-keys -t $((i+1)) "${COMMANDS[$i]}" C-j
done

tmux select-layout $LAYOUT
tmux attach -t $SESSION
    
