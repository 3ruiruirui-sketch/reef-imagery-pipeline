import pytest

from src import stac_ingest


class DummyAsset:
    def __init__(self, href: str):
        self.href = href


class DummyItem:
    def __init__(self, item_id: str, cloud_cover: float, assets: dict):
        self.id = item_id
        self.properties = {"eo:cloud_cover": cloud_cover}
        self.collection_id = "sentinel-2-l2a"
        self.assets = {k: DummyAsset(v) for k, v in assets.items()}
        self.datetime = None


class DummySearch:
    def __init__(self, items):
        self._items = items

    def get_items(self):
        return self._items

    def items(self):
        return self._items


class DummyClient:
    def __init__(self, items):
        self._items = items

    def search(self, *args, **kwargs):
        return DummySearch(self._items)


def test_normalize_datetime_range_string():
    assert stac_ingest.normalize_datetime_range("2025-09-01/2025-09-30") == "2025-09-01/2025-09-30"


def test_normalize_datetime_range_tuple():
    assert stac_ingest.normalize_datetime_range(("2025-09-01", "2025-09-30")) == "2025-09-01/2025-09-30"


def test_choose_least_cloudy():
    items = [
        DummyItem("item-a", 32.0, {}),
        DummyItem("item-b", 10.5, {}),
        DummyItem("item-c", 15.0, {}),
    ]
    best = stac_ingest.choose_least_cloudy(items)
    assert best is not None
    assert best.id == "item-b"


def test_get_asset_hrefs():
    item = DummyItem("item-001", 5.0, {"B02": "https://example.com/B02.tif", "B03": "https://example.com/B03.tif"})
    hrefs = stac_ingest.get_asset_hrefs(item, ["B02", "B03", "B04"])
    assert hrefs["B02"] == "https://example.com/B02.tif"
    assert hrefs["B03"] == "https://example.com/B03.tif"
    assert "B04" not in hrefs


def test_open_stac_catalog(monkeypatch):
    dummy_client = DummyClient([])

    def fake_open(url, modifier=None):
        assert url == stac_ingest.EARTH_SEARCH_STAC_URL
        return dummy_client

    monkeypatch.setattr(stac_ingest.Client, "open", fake_open)
    client = stac_ingest.open_stac_catalog(stac_ingest.EARTH_SEARCH_STAC_URL)
    assert client is dummy_client


def test_search_sentinel2_scenes(monkeypatch):
    items = [DummyItem("item-x", 12.0, {})]
    dummy_client = DummyClient(items)

    def fake_open(url, modifier=None):
        assert url == stac_ingest.EARTH_SEARCH_STAC_URL
        return dummy_client

    monkeypatch.setattr(stac_ingest.Client, "open", fake_open)
    scenes = stac_ingest.search_sentinel2_scenes(
        37.07, -8.21, ("2025-09-01", "2025-09-15"), max_cloud_cover=30.0
    )
    assert scenes == items


def test_search_sentinel2_scenes_fallback(monkeypatch):
    items = [DummyItem("item-y", 11.0, {})]
    dummy_client = DummyClient(items)
    calls = []

    def fake_open(url, modifier=None):
        calls.append(url)
        if url == stac_ingest.EARTH_SEARCH_STAC_URL:
            raise RuntimeError("Earth Search unavailable")
        return dummy_client

    monkeypatch.setattr(stac_ingest.Client, "open", fake_open)
    scenes = stac_ingest.search_sentinel2_scenes(
        37.07, -8.21, ("2025-09-01", "2025-09-15"), max_cloud_cover=30.0
    )
    assert scenes == items
    assert calls == [stac_ingest.EARTH_SEARCH_STAC_URL, stac_ingest.PC_STAC_URL]
