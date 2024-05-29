#!/bin/bash
# set -x
echo "[]" > comments.json
echo "Created comments.json"

while IFS= read -r line; do
    echo "Processing line: $line"
    
    if [[ "$line" =~ ^diff ]]; then
        echo "Processing diff header line: $line"

        # Extract file names from the diff header
        changed_file=$(echo "$line" | awk '{print $3}' | sed 's/^a\///')
        echo "File: $changed_file"
    elif [[ "$line" =~ ^@@ ]]; then
        echo "Line matches diff hunk header"

        position=$(echo "$line" | grep -oP '(?<=@@ -).*?(?= @@)' | awk '{split($0, a, " "); print a[2]}')
        echo "Got line number: $position"

        # Construct JSON manually
        comment="{\"path\":\"$changed_file\",\"position\":$position,\"body\":\"Please apply the suggested clang-format changes.\"}"
        echo "Generated comment: $comment"

        # Read existing comments.json and append new comment
        existing_comments=$(cat comments.json)
        new_comments=$(echo "$existing_comments" | sed -e 's/]//')
        new_comments+="$comment,]"
        echo "$new_comments" > comments.json
        echo "Appended comment to comments.json"
    fi

done < clang_format.patch
