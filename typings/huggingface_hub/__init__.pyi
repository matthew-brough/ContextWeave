from pathlib import Path

def hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str: ...
def snapshot_download(
    repo_id: str,
    *,
    local_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str: ...
