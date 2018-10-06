#!/bin/bash
FLAGS_SKIP="-k 'not external'"
FLAGS_SUMMARY=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
      -a|--all)
      FLAGS_SKIP=""
      shift
      ;;
      -s|--summary)
      FLAGS_SUMMARY="--tb=no --no-print-logs"
      shift
      ;;
      -*) # Unknown flag
      echo "Unrecognized flag \"$key\""
      shift
      ;;
      *) # Positional arguments
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

cmd="pytest --ignore=models $FLAGS_SKIP $FLAGS_SUMMARY $POSITIONAL"
echo "$cmd"
eval $cmd
