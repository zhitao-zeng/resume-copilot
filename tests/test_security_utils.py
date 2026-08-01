import asyncio
from pathlib import Path

import pytest

from security_utils import safe_child_path, safe_filename, safe_task_id, validate_public_http_url


def test_task_id_rejects_path_traversal():
    with pytest.raises(ValueError):
        safe_task_id("../../tmp/pwn")


def test_filename_is_reduced_to_safe_basename():
    assert safe_filename("../../private/resume.docx") == "resume.docx"


def test_safe_child_path_rejects_escape(tmp_path: Path):
    with pytest.raises(ValueError):
        safe_child_path(tmp_path, "../outside.txt")


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/secret",
    "http://169.254.169.254/latest/meta-data",
    "file:///etc/passwd",
    "http://localhost/internal",
])
def test_ssrf_guard_rejects_non_public_targets(url: str):
    with pytest.raises(ValueError):
        asyncio.run(validate_public_http_url(url))
