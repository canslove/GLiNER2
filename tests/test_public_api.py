"""Public import surface must remain stable across the refactor."""

import importlib


def test_extractor_import_is_preserved():
    from gliner2 import Extractor  # noqa: F401
    from gliner2.model import Extractor as ModelExtractor  # noqa: F401

    assert Extractor is ModelExtractor


def test_gliner2_import_is_preserved():
    from gliner2 import GLiNER2  # noqa: F401
    from gliner2.inference.engine import GLiNER2 as EngineGLiNER2  # noqa: F401

    assert GLiNER2 is EngineGLiNER2


def test_extractor_config_import_is_preserved():
    from gliner2 import ExtractorConfig  # noqa: F401
    from gliner2.model import ExtractorConfig as ModelConfig  # noqa: F401

    assert ExtractorConfig is ModelConfig


def test_gliner2_class_remains_span_architecture():
    from gliner2 import GLiNER2

    assert getattr(GLiNER2, "architecture", "span") == "span"


def test_schema_and_structure_builder_importable():
    from gliner2 import Schema, StructureBuilder  # noqa: F401

    schema = Schema()
    assert schema is not None
