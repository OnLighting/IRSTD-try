"""Named Stage-A objective contracts."""

from __future__ import annotations

V5_1_OBJECTIVE = "v5.1-ur"
V5_2A_OBJECTIVE = "v5.2a-signed-r"
SUPPORTED_OBJECTIVES = frozenset((V5_1_OBJECTIVE, V5_2A_OBJECTIVE))


def validate_objective_version(value: str) -> str:
    version = str(value)
    if version not in SUPPORTED_OBJECTIVES:
        supported = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise ValueError(
            f"unsupported Stage-A objective {version!r}; expected one of: {supported}"
        )
    return version
