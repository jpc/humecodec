#!/usr/bin/env bash
set -euo pipefail

# Bump the version in pyproject.toml, commit, tag, and push.
# Usage: bump-version.sh [--patch]
#   Default: bump minor version (e.g. 0.2.0 -> 0.3.0)
#   --patch: bump patch version (e.g. 0.2.0 -> 0.2.1)

BUMP="minor"
if [[ "${1:-}" == "--patch" ]]; then
  BUMP="patch"
fi

PYPROJECT="$(git rev-parse --show-toplevel)/pyproject.toml"

# Extract current version
current=$(grep -Po '(?<=^version = ")[^"]+' "$PYPROJECT")
if [[ -z "$current" ]]; then
  echo "Error: could not find version in $PYPROJECT" >&2
  exit 1
fi

IFS='.' read -r major minor patch <<< "$current"

if [[ "$BUMP" == "patch" ]]; then
  new_patch=$((patch + 1))
  new_version="${major}.${minor}.${new_patch}"
else
  new_minor=$((minor + 1))
  new_version="${major}.${new_minor}.0"
fi

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
