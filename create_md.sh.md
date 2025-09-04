# `create_md.sh`

```bash
#!/bin/bash

# This script creates a markdown file for a given code file.

# Check if a filename is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

FILENAME="$1"
MD_FILENAME="${FILENAME}.md"

# Create the markdown file
# Use a language hint for syntax highlighting if possible
EXTENSION="${FILENAME##*.}"
LANG=""
if [ "$EXTENSION" = "py" ]; then
  LANG="python"
elif [ "$EXTENSION" = "sh" ]; then
  LANG="bash"
elif [ "$EXTENSION" = "c" ]; then
  LANG="c"
elif [ "$EXTENSION" = "cl" ]; then
  LANG="cl"
fi

cat << EOF > "$MD_FILENAME"
# \`$(basename "$FILENAME")\`

\`\`\`${LANG}
$(cat "$FILENAME")
\`\`\`
EOF
```
