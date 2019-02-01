#!/bin/bash

path=$(git rev-parse --show-cdup)

if [ ${#path} -ne 0 ]
then
	cd $path
fi

mv docs html
cd html
make html
cd ..
mv html docs
rm -r doctrees
