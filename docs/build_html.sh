#!/bin/bash

path=$(git rev-parse --show-cdup)

echo "${#path}"

if [ ${#path} -ne 0 ]
then
	echo "bob"
	cd $path
fi

mv docs html
cd html
make html
cd ..
mv html docs
rm -r doctrees
