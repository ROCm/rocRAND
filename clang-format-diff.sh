#!/bin/bash

# Save unified diff with 0 context
git diff -U0 HEAD > changes.diff

# Apply clang-format on diff
FORMAT_DIFF=$(clang-format -style=file changes.diff)

if [ ! -z "$FORMAT_DIFF" ]; then
    echo "The following formatting errors were found:"
    echo "$FORMAT_DIFF"
    exit 1
else
    echo "All code is properly formatted."
fi
