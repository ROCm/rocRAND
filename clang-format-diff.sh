#!/bin/bash

# Save unified diff with 0 context
DIFF=$(git diff -U0 HEAD)
echo "$DIFF" > changes.diff

# Apply clang-format on diff
curl -LO https://raw.githubusercontent.com/llvm/llvm-project/main/clang/tools/clang-format/clang-format-diff.py
chmod +x clang-format-diff.py
FORMAT_DIFF=$(./clang-format-diff.py -p1 changes.diff)

if [ ! -z "$FORMAT_DIFF" ]; then
    echo "The following formatting errors were found:"
    echo "$FORMAT_DIFF"
    exit 1
else
    echo "All code is properly formatted."
fi
