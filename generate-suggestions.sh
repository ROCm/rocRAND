#!/usr/bin/env bash
echo "[]" > comments.json
echo "Created comments.json"

while IFS= read -r line; do
    echo "Processing line: $line"
    
    if [[ "$line" =~ ^diff ]]; then
        echo "diff header line"

        changed_file=$(echo "$line" | awk '{print $3}' | sed 's/^a\///')
        echo "File: $changed_file"

    elif [[ "$line" =~ ^@@ ]]; then
        echo "diff hunk header"

        position=$(echo "$line" | grep -oP '(?<=@@ -).*?(?= @@)' | awk '{split($0, a, " "); print a[2]}')
        echo "Got line number: $position"

        # Construct JSON manually
        comment="{\"path\":\"$changed_file\",\"position\":$position,\"body\":\"Please apply the suggested clang-format changes.\"}"
        echo "Generated comment: $comment"
        comments+=("$comment")
    fi

done < clang_format.patch

comments_json=$(printf ",%s" "${comments[@]}")
comments_json="[${comments_json:1}]"
echo "$comments_json" > comments.json
echo "Final comments.json: $comments_json"
