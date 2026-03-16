"""YAML config loader for mini OCR.

Author: David
"""

import ast
from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with attribute access for nested config values."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def from_dict(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls({k: cls.from_dict(v) for k, v in value.items()})
        if isinstance(value, list):
            return [cls.from_dict(v) for v in value]
        return value


class YamlConfigLoader:
    """Load project config from a YAML file."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)

    def load(self, overrides: list[str] | None = None) -> ConfigNode:
        config_data = self._read_yaml()
        self._apply_defaults(config_data)
        if overrides:
            self._apply_dotlist(config_data, overrides)
            self._apply_defaults(config_data)
        return ConfigNode.from_dict(config_data)

    def _read_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file root must be a mapping object")
        return data

    @staticmethod
    def _apply_defaults(config_data: dict[str, Any]) -> None:
        datasets = config_data.setdefault("DATASETS", {})
        datasets.setdefault("IS_HWR", False)

        path_key_pairs = (
            ("TRAIN_PATH", "TRAIN_LIST"),
            ("VAL_PATH", "VAL_LIST"),
            ("TEST_PATH", "TEST_LIST"),
        )
        for new_key, legacy_key in path_key_pairs:
            value = datasets.get(new_key, datasets.get(legacy_key, ""))
            if value is None:
                value = ""
            elif isinstance(value, (list, tuple)):
                raise ValueError(
                    f"DATASETS.{new_key} must be a single path string, got {value!r}"
                )
            else:
                value = str(value)
            datasets[new_key] = value

        input_cfg = config_data.setdefault("INPUT", {})
        size = input_cfg.get("SIZE_CRNN")
        if isinstance(size, str):
            try:
                size = ast.literal_eval(size)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("INPUT.SIZE_CRNN must be a tuple/list like (32, 400)") from exc
        if isinstance(size, tuple):
            size = list(size)
        if isinstance(size, list) and len(size) == 2:
            input_cfg["SIZE_CRNN"] = [int(size[0]), int(size[1])]

    @staticmethod
    def _apply_dotlist(config_data: dict[str, Any], overrides: list[str]) -> None:
        if len(overrides) % 2 != 0:
            raise ValueError("opts must be key-value pairs, e.g. SOLVER.BATCH_SIZE 32")

        for index in range(0, len(overrides), 2):
            path = overrides[index]
            raw_value = overrides[index + 1]
            value = yaml.safe_load(raw_value)

            target = config_data
            keys = path.split(".")
            for key in keys[:-1]:
                target = target.setdefault(key, {})
                if not isinstance(target, dict):
                    raise ValueError(f"Cannot set nested key on non-dict path: {path}")
            target[keys[-1]] = value


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> ConfigNode:
    return YamlConfigLoader(config_path).load(overrides=overrides)
