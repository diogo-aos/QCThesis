#!/bin/bash

git status
git add -A
git commit -a -m "$1"
git push

cp -v $HOME/workspace/thesis_writing/latex/thesis.pdf $HOME/workspace/thesis_website/output/docs/
