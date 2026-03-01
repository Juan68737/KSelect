import pytest
import yaml
from pydantic import ValidationError

from kselect.models.config import ChunkingConfig, KSelectConfig


def test_config_round_trip(tmp_path):
    """Config serializes to YAML and reloads with identical values."""
    cfg = KSelectConfig()
    yaml_path = tmp_path / "kselect.yaml"
    yaml_path.write_text(yaml.dump(cfg.model_dump()))
    reloaded = KSelectConfig.from_yaml(str(yaml_path))
    assert reloaded == cfg


def test_chunk_overlap_validator():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=128, chunk_overlap=128)


def test_env_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    yaml_path = tmp_path / "kselect.yaml"
    yaml_path.write_text('llm:\n  api_key: "${OPENAI_API_KEY}"\n')
    cfg = KSelectConfig.from_yaml(str(yaml_path))
    assert cfg.llm.api_key == "sk-test"
