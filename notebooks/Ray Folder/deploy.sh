#!/bin/bash
# commit

# delete existing file
rm $2

# run script
python3 $1


# add
git add .

# commit
git commit -m "new commit"

# push
git push origin main

