"""Exact filesystem calls eligible for the existing workspace permission tier."""
from pathlib import Path
from typing import Any, Dict


def is_workspace_file_call(
    module_id: str, arguments: Dict[str, Any], workspace_root: Path,
) -> bool:
    """Check literal paths using Core's current-directory resolution semantics.

    This only classifies permission. Core remains responsible for its schema,
    environment sandbox and actual operation, including a fresh path check.
    Unknown file operations and unresolved bindings receive no lower tier.
    """
    if module_id not in {"file.read", "file.write"}:
        return False
    params = arguments.get("params")
    if not isinstance(params, dict):
        return False
    path = params.get("path")
    if not isinstance(path, str) or not path.strip():
        return False
    if any(marker in path for marker in ("\x00", "${", "{{")):
        return False
    try:
        # Relative paths are interpreted by Core from the process cwd, not
        # silently rebased to this session's captured root after a chdir.
        resolved = Path(path).expanduser().resolve()
        if workspace_root.resolve() != workspace_root or resolved == workspace_root:
            return False
        resolved.relative_to(workspace_root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True
