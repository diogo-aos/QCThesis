#!/bin/bash

cd $HOME/workspace/thesis_writing

git status
git add -A
git commit -a -m "$1"
git push

cp -v $HOME/workspace/thesis_writing/latex/build/thesis.pdf $HOME/workspace/thesis_website/output/docs/

cd $HOME/workspace/thesis_website/output

git status
git add docs/*
git commit -m "thesis pdf update" docs/*
git push