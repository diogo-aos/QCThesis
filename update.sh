#!/bin/bash

git add -A
git commit -a -m "$1"
git push

cp $HOME/workspace/thesis_writing/latex/thesis.pdf $HOME/workspace/thesis_website/output/docs/
