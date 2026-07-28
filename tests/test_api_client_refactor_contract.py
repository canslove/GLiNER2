"""Transport-level regression tests for GLiNER2API's public wrappers."""
from __future__ import annotations

import json
import warnings

import pytest

from gliner2.api_client import (
    AuthenticationError,
    GLiNER2API,
    GLiNER2APIError,
    ServerError,
    ValidationError,
)


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.content = json.dumps(body).encode() if body is not None else b""

    def json(self):
        if self._body is None:
            raise ValueError("empty response")
        return self._body


class _Session:
    def __init__(self, body, status_code=200):
        self.body = body
        self.status_code = status_code
        self.calls = []
        self.headers = {}

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "payload": json, "timeout": timeout})
        return _Response(self.body, self.status_code)

    def close(self):
        pass


def _client(body={"result": {"canned": True}}, status_code=200):
    client = GLiNER2API(api_key="test-key", api_base_url="https://api.example.test")
    client.session = _Session(body, status_code)
    return client


def _payload(client):
    assert len(client.session.calls) == 1
    return client.session.calls[0]["payload"]


def test_entity_single_and_batch_preserve_reply_shape_asymmetry():
    single = _client({"result": {"person": ["Ada"]}})
    assert single.extract_entities("Ada", {"person": "human"}) == {
        "entities": {"person": ["Ada"]}
    }
    assert _payload(single)["schema"] == ["person"]

    batch = _client({"result": {"person": ["Ada"]}})
    assert batch.batch_extract_entities(["Ada"], ["person"]) == [{"person": ["Ada"]}]
    assert _payload(batch)["task"] == "extract_entities"


def test_classification_preserves_single_and_batch_wire_contracts():
    single = _client({"result": {"classification": "positive"}})
    assert single.classify_text("great", {"sentiment": ["positive", "negative"]}) == {
        "sentiment": "positive"
    }
    assert _payload(single)["task"] == "classify_text"
    assert _payload(single)["schema"] == {"categories": ["positive", "negative"]}

    multi = _client({"result": {"sentiment": "positive", "topic": "tech"}})
    multi.classify_text("great", {"sentiment": ["positive"], "topic": ["tech"]})
    assert _payload(multi)["task"] == "schema"

    batch = _client({"result": {"sentiment": "positive"}})
    assert batch.batch_classify_text(["great"], {"sentiment": ["positive"]}) == [
        {"sentiment": "positive"}
    ]
    assert _payload(batch)["task"] == "schema"


def test_batch_wrappers_listify_except_generic_extract():
    json_client = _client({"result": {"invoice": {}}})
    assert json_client.batch_extract_json(["x"], {"invoice": ["total"]}) == [{"invoice": {}}]

    relation_client = _client({"result": {"relation_extraction": {}}})
    assert relation_client.batch_extract_relations(["x"], ["works_at"]) == [
        {"relation_extraction": {}}
    ]

    extract_client = _client({"result": {"only": "a dict"}})
    assert extract_client.batch_extract(["x"], {"entities": ["person"]}) == {"only": "a dict"}


def test_generic_extract_preserves_validation_short_circuit_and_fanout_warning():
    client = _client()
    assert client.batch_extract([], {"entities": ["person"]}) == []
    assert client.session.calls == []

    with pytest.raises(ValueError, match="at least one extraction task"):
        client.extract("x", {"unknown": True})

    fanout = _client({"result": {"ok": True}})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert fanout.batch_extract(["a", "b"], [{"entities": ["a"]}, {"entities": ["b"]}]) == [
            {"ok": True},
            {"ok": True},
        ]
    assert len(fanout.session.calls) == 2
    assert [warning.category for warning in captured] == [UserWarning]


def test_generic_extract_keeps_schema_api_wire_metadata():
    client = _client({"result": {}})
    schema = client.create_schema().entities(["person"], dtype="str", threshold=0.4)
    schema = schema.classification("sentiment", ["positive"], cls_threshold=0.8)
    client.extract("x", schema)
    assert _payload(client)["schema"] == {
        "entities": ["person"],
        "entity_dtype": "str",
        "entity_threshold": 0.4,
        "classifications": {
            "sentiment": {
                "labels": ["positive"],
                "multi_label": False,
                "cls_threshold": 0.8,
            }
        },
    }


@pytest.mark.parametrize(
    ("status_code", "body", "exception"),
    [
        (401, {"detail": "bad key"}, AuthenticationError),
        (400, {"detail": "bad request"}, ValidationError),
        (422, None, ValidationError),
        (500, {"detail": "server"}, ServerError),
        (418, {"detail": "teapot"}, GLiNER2APIError),
    ],
)
def test_http_error_mapping_preserves_exception_data(status_code, body, exception):
    client = _client(body, status_code)
    with pytest.raises(exception) as caught:
        client.extract_entities("x", ["person"])
    assert caught.value.status_code == status_code
    assert caught.value.response_data == body


def test_empty_success_response_is_an_api_error():
    client = _client(None)
    with pytest.raises(GLiNER2APIError, match="Empty response body from API") as caught:
        client.extract_entities("x", ["person"])
    assert caught.value.status_code == 200
