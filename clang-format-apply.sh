#!/bin/bash

BASE_BRANCH=$1
CURRENT_BRANCH=$2

# Save unified diff with 0 context
git diff -U0 --name-only $BASE_BRANCH...$CURRENT_BRANCH > changes.diff

cat changes.diff

# Run clang-format in-place on .h and .cpp files
while IFS= read -r line; do
  if [ -n "$line" ]; then
    if [[ "$line" == *.h ]] || [[ "$line" == *.cpp ]]; then
      clang-format -i "$line"
    fi
  fi
done < changes.diff
