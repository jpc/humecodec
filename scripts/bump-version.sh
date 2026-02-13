#!/usr/bin/env bash
set -euo pipefail

# Bump the minor version in pyproject.toml, commit, tag, and push.

PYPROJECT="$(git rev-parse --show-toplevel)/pyproject.toml"

# Extract current version
current=$(grep -Po '(?<=^version = ")[^"]+' "$PYPROJECT")
if [[ -z "$current" ]]; then
  echo "Error: could not find version in $PYPROJECT" >&2
  exit 1
fi

IFS='.' read -r major minor patch <<< "$current"
new_minor=$((minor + 1))
new_version="${major}.${new_minor}.0"

echo "Bumping version: ${current} -> ${new_version}"

# Update pyproject.toml
sed -i "s/^version = \"${current}\"/version = \"${new_version}\"/" "$PYPROJECT"

# Commit and tag
git add "$PYPROJECT"
git commit -m "Bump version to ${new_version}"
git tag "v${new_version}"

# Push commit and tag
git push origin HEAD "v${new_version}"

echo "Done: pushed v${new_version}"
