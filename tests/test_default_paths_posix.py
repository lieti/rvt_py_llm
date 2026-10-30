"""The default output paths of save_default_to_file and create_blender_file_example work on every operating
system (0.2.2): ``settings/<name>.json`` under the current working directory, built with os.path.join.

Expected values come from the file system, not from the code: after the call the file must exist at the
path the operating system builds from ("settings", name), and a second call must overwrite it without error.
Before 0.2.2 the path was a literal with a backslash, so on POSIX the first call raised FileNotFoundError
(KNOWN_ISSUES KI-35, KI-52); tests/test_char_default.py and tests/test_char_blend.py pin the Windows behaviour,
which is unchanged.
"""
import json
import os

import pytest

import rvt.blend
import rvt.default


@pytest.mark.parametrize("call, name", [
    (lambda: rvt.default.DefaultValues().save_default_to_file(), "default_settings.json"),
    (lambda: rvt.blend.create_blender_file_example(), "blender_file_example.json"),
], ids=["save_default_to_file", "create_blender_file_example"])
def test_default_path_is_settings_dir_of_cwd_on_any_os(tmp_path, monkeypatch, call, name):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "settings").exists()
    call()
    target = tmp_path / "settings" / name
    assert target.is_file(), sorted(os.listdir(tmp_path))
    assert os.path.join("settings", name) == os.path.relpath(target, tmp_path)
    first = target.read_text()
    json.loads(first)
    call()  # the directory exists now: the file is overwritten, nothing raises
    assert target.read_text() == first
    assert sorted(os.listdir(tmp_path)) == ["settings"]
