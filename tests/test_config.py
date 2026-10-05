from dataclasses import dataclass

import pytest

import soren
from soren.config import from_dict, load_config, load_yaml


@dataclass
class _Cfg:
    a: int = 1
    b: str = "x"


def test_package_imports():
    assert soren.__version__


def test_from_dict_defaults_and_overrides():
    assert from_dict(_Cfg, None) == _Cfg()
    assert from_dict(_Cfg, {"a": 5}) == _Cfg(a=5)


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown _Cfg keys: c"):
        from_dict(_Cfg, {"c": 1})


def test_load_config_section(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("section:\n  a: 3\n  b: y\n")
    assert load_config(_Cfg, path, section="section") == _Cfg(a=3, b="y")
    assert load_config(_Cfg, path, section="missing") == _Cfg()


def test_load_yaml_empty_and_non_mapping(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_yaml(empty) == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        load_yaml(bad)
