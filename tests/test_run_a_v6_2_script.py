import os
from pathlib import Path


def test_remote_script_prints_packaged_output_files_at_end() -> None:
    script_path = Path(os.environ.get("A_V6_RUN_SCRIPT", "run_a_v6_2_remote.sh"))
    script = script_path.read_text(encoding="utf-8")

    marker = 'echo "[A-v6.2] packaged output files:"'
    assert 'export A_V6_RUN_SCRIPT="${BASH_SOURCE[0]}"' in script
    assert marker in script
    assert 'realpath "$ARCHIVE"' in script
    assert 'realpath "${ARCHIVE}.sha256"' in script
    assert 'ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"' in script
    assert script.index(marker) > script.index('sha256sum "$ARCHIVE"')
