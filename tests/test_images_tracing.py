import io
import json

import pytest
from PIL import Image

from taxon.labeling.env import Secrets
from taxon.labeling.images import MAX_SIDE, ImageStore, download, to_small_jpeg
from taxon.labeling.llm import LLMRequest, LLMResult, ModelConfig, Status, Usage
from taxon.labeling.tracing import LangfuseTracer, build_tracer


def png(size=(1500, 900), mode="RGBA"):
    out = io.BytesIO()
    Image.new(mode, size).save(out, format="PNG")
    return out.getvalue()


def product(asin="A1"):
    return {"parent_asin": asin, "image_url": "https://m.media-amazon.com/images/I/x.jpg"}


def test_images_are_shrunk_and_converted_to_rgb_jpeg():
    with Image.open(io.BytesIO(to_small_jpeg(png()))) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert max(image.size) == MAX_SIDE


def test_small_images_are_not_enlarged():
    with Image.open(io.BytesIO(to_small_jpeg(png((64, 32))))) as image:
        assert image.size == (64, 32)


def test_garbage_is_rejected():
    with pytest.raises(Exception, match="cannot identify"):
        to_small_jpeg(b"not an image")


@pytest.mark.parametrize(
    "url",
    [
        "http://m.media-amazon.com/x.jpg",
        "https://evil.example/x.jpg",
        "file:///etc/passwd",
        "https://m.media-amazon.com.evil.example/x.jpg",
    ],
)
def test_only_https_on_known_hosts_is_downloaded(url):
    with pytest.raises(ValueError, match="allowed image host"):
        download(url)


def test_a_store_fetches_once_and_hashes_what_it_kept(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return png()

    store = ImageStore(tmp_path, fetch)
    assert store.ensure_all([product()]) == {"images": 1, "without_image": 0}
    store.ensure_all([product()])
    assert len(calls) == 1
    path, sha = store.image_for(product())
    assert path.endswith("A1.jpg")
    assert len(sha) == 64


def test_a_failed_image_is_remembered_and_the_product_goes_without(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        raise OSError("gone")

    store = ImageStore(tmp_path, fetch)
    assert store.ensure_all([product()]) == {"images": 0, "without_image": 1}
    assert store.image_for(product()) is None
    assert ImageStore(tmp_path, fetch).ensure_all([product()]) == {"images": 0, "without_image": 1}
    assert len(calls) == 1
    assert json.loads((tmp_path / "failures.json").read_text()) == {"A1": "OSError"}


class FakeObservation:
    def __init__(self):
        self.ended = False

    def end(self):
        self.ended = True


class FakeLangfuse:
    def __init__(self):
        self.observations = []
        self.flushed = False

    def start_observation(self, **kwargs):
        observation = FakeObservation()
        self.observations.append((kwargs, observation))
        return observation

    def flush(self):
        self.flushed = True


def test_the_tracer_records_a_generation_without_image_bytes():
    fake = FakeLangfuse()
    tracer = LangfuseTracer(fake)
    config = ModelConfig("openai", "gpt-6-luna", 10, "low")
    request = LLMRequest(config, "pick", "system", "user text", "/tmp/p.jpg", "sha")
    result = LLMResult(request.key, Status.OK, "{}", Usage(1, 2, 3, 4), cost=0.25)
    tracer.record(request, result)
    tracer.flush()
    [(kwargs, observation)] = fake.observations
    assert kwargs["as_type"] == "generation"
    assert kwargs["model"] == "gpt-6-luna"
    assert kwargs["input"] == "user text"
    assert kwargs["usage_details"] == {"input": 1, "cached": 2, "cache_write": 3, "output": 4}
    assert kwargs["cost_details"] == {"total": 0.25}
    assert kwargs["metadata"]["has_image"] is True
    assert observation.ended
    assert fake.flushed


def test_tracing_is_off_without_both_keys():
    assert build_tracer(Secrets({})) is None
    assert build_tracer(Secrets({"LANGFUSE_PUBLIC_KEY": "a"})) is None


def test_the_tracer_matches_the_installed_langfuse_signature():
    import inspect

    from langfuse import Langfuse

    parameters = inspect.signature(Langfuse.start_observation).parameters
    for name in ("name", "as_type", "model", "input", "output", "metadata"):
        assert name in parameters
    assert "usage_details" in parameters
    assert "cost_details" in parameters
