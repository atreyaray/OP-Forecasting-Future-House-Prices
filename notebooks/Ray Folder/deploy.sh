#!/bin/bash
# commit

# run script
python3 $1


# add
git add .

# commit
git commit -m "new commit"

# push
git push origin main

