#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey).  
# Apache 2.0.

# This script removes the examples in an egs/ directory, e.g.
# steps/nnet2/remove_egs.sh exp/nnet4b/egs/
# We give it its own script because we need to be careful about
# things that are soft links to something in storage/ (i.e. remove the
# data that's linked to as well as the soft link), and we want to not
# delete the examples if someone has done "touch $dir/egs/.nodelete".


if [ $# != 1 ]; then
  echo "Usage: $0 <egs-dir>"
  echo "e.g.: $0 data/nnet4b/egs/"
  echo "This script is usually equivalent to 'rm <egs-dir>/egs.*' but it handles following"
  echo "soft links to <egs-dir>/storage/, and avoids deleting anything in the directory if"
  echo "someone did 'touch <egs-dir>/.nodelete"
  exit 1;
fi

egs=$1

if [ ! -d $egs ]; then
  echo "$0: expected directory $egs to exist"
  exit 1;
fi

if [ -f $egs/.nodelete ]; then
  echo "$0: not delting egs in $egs since $egs/.nodelete exists"
  exit 0;
fi

flist=$egs/egs.*.ark

if [ "$flist" == '$egs/egs.*.ark' ]; then  # did not expand..
  echo "$0: directory $egs does not seem to contain egs"
  exit 1;
fi

for f in $flist; do
  if [ -L $f ]; then
    rm $(readlink $f)  # this will print a warning if it fails.
  fi
  rm $f
done


echo "$0: Finished deleting examples in $egs"
