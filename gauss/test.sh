#!/bin/bash

if [ ! -f gauss ]; then
    echo "Making executable"
    $(make)
fi

if [ ! -d "test" ]; then
  mkdir test
fi

for i in {1..8}
do
	for j in {1..8}; do mpirun -n $i gauss $j >> test/test$i-$j.txt; done
done
