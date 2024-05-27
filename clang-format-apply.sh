#!/bin/bash

BASE_BRANCH=$1
CURRENT_BRANCH=$2

# Save unified diff with 0 context
git diff -U0 --name-only $BASE_BRANCH...$CURRENT_BRANCH > changes.diff

# Apply clang-format on diff
FORMAT_DIFF=$(clang-format -i -style=file changes.diff)
