#!/usr/bin/env bash
# Release flyto-ai — auto-sync flyto-core + flyto-blueprint version pins
set -euo pipefail

CORE_TOML="/Library/其他專案/flytohub/flyto-core/pyproject.toml"
BP_TOML="/Library/其他專案/flytohub/flyto-blueprint/pyproject.toml"
AI_TOML="pyproject.toml"
AI_INIT="flyto_ai/__init__.py"
AI_TEST="tests/test_cli.py"

# Read dependency versions
CORE_VER=$(grep '^version' "$CORE_TOML" | head -1 | sed 's/.*"\(.*\)"/\1/')
BP_VER=$(grep '^version' "$BP_TOML" | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "flyto-core      : $CORE_VER"
echo "flyto-blueprint : $BP_VER"

# Read flyto-ai current version
OLD_VER=$(grep '^version' "$AI_TOML" | head -1 | sed 's/.*"\(.*\)"/\1/')
echo "flyto-ai current: $OLD_VER"

# Bump: arg1 = new version, or auto patch bump
if [ "${1:-}" ]; then
    NEW_VER="$1"
else
    IFS='.' read -r major minor patch <<< "$OLD_VER"
    NEW_VER="$major.$minor.$((patch + 1))"
fi
echo "flyto-ai new    : $NEW_VER"
echo ""

# 1. Update dependency pins
sed -i '' "s/\"flyto-core\[browser\]>=.*\"/\"flyto-core[browser]>=$CORE_VER\"/" "$AI_TOML"
sed -i '' "s/\"flyto-blueprint>=.*\"/\"flyto-blueprint>=$BP_VER\"/" "$AI_TOML"

# 2. Update flyto-ai version in 3 files
sed -i '' "s/^version = \"$OLD_VER\"/version = \"$NEW_VER\"/" "$AI_TOML"
sed -i '' "s/__version__ = \"$OLD_VER\"/__version__ = \"$NEW_VER\"/" "$AI_INIT"
sed -i '' "s/v$OLD_VER/v$NEW_VER/" "$AI_TEST"

# 3. Run tests
echo "Running tests..."
python -m pytest tests/ -q || { echo "Tests failed!"; exit 1; }

# 4. Build
rm -rf dist/
python -m build 2>&1 | tail -3

# 5. Confirm before publish
echo ""
echo "Ready to publish flyto-ai $NEW_VER (core >=$CORE_VER, blueprint >=$BP_VER)"
read -p "Publish to PyPI? [y/N] " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

python -m twine upload dist/*

# 6. Commit + push
git add "$AI_TOML" "$AI_INIT" "$AI_TEST"
git commit -m "release: flyto-ai v$NEW_VER (core >=$CORE_VER, blueprint >=$BP_VER)"
git push

echo ""
echo "Done! flyto-ai $NEW_VER published."
