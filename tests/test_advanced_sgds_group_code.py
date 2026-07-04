import os
from unittest import mock

import pandas as pd
import pytest
import requests

import AdvancedSGDSGroupCode as code


# ---------------------------------------------------------------------------
# normalize_classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a", "A"),
        ("  b  ", "B"),
        ("C", "C"),
        (" c\n", "C"),
        (123, "123"),
    ],
)
def test_normalize_classification(raw, expected):
    assert code.normalize_classification(raw) == expected


# ---------------------------------------------------------------------------
# is_valid_classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        ("A", True),
        ("B", True),
        ("C", True),
        ("D", False),
        ("", False),
        ("a", False),  # expects normalized (upper) input
    ],
)
def test_is_valid_classification_default(value, expected):
    assert code.is_valid_classification(value) is expected


def test_is_valid_classification_custom_set():
    assert code.is_valid_classification("X", ["X", "Y"]) is True
    assert code.is_valid_classification("A", ["X", "Y"]) is False


# ---------------------------------------------------------------------------
# build_image_url
# ---------------------------------------------------------------------------
def test_build_image_url_contains_all_parts():
    url = code.build_image_url(51.5, -0.12, api_key="secret")
    assert url.startswith(
        "https://maps.googleapis.com/maps/api/streetview?"
    )
    assert "location=51.5,-0.12" in url
    assert "size=400x400" in url
    assert "fov=90" in url
    assert "heading=80" in url
    assert "pitch=10" in url
    assert "key=secret" in url


def test_build_image_url_default_api_key():
    url = code.build_image_url(1, 2)
    assert f"key={code.API_KEY}" in url


# ---------------------------------------------------------------------------
# create_classification_folders
# ---------------------------------------------------------------------------
def test_create_classification_folders(tmp_path):
    folders = code.create_classification_folders(base_dir=str(tmp_path))
    assert set(folders) == {"A", "B", "C"}
    for classification, path in folders.items():
        assert os.path.isdir(path)
        assert path == os.path.join(str(tmp_path), classification)


def test_create_classification_folders_idempotent(tmp_path):
    code.create_classification_folders(base_dir=str(tmp_path))
    # Should not raise when the folders already exist.
    folders = code.create_classification_folders(base_dir=str(tmp_path))
    assert all(os.path.isdir(p) for p in folders.values())


def test_create_classification_folders_custom(tmp_path):
    folders = code.create_classification_folders(
        base_dir=str(tmp_path), classifications=["X", "Y"]
    )
    assert set(folders) == {"X", "Y"}


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------
def test_save_image_writes_bytes(tmp_path):
    path = code.save_image(b"\x89PNGfake", str(tmp_path), "streetview_0")
    assert path == os.path.join(str(tmp_path), "streetview_0.jpg")
    with open(path, "rb") as f:
        assert f.read() == b"\x89PNGfake"


# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------
def test_download_image_returns_content():
    session = mock.Mock()
    resp = mock.Mock()
    resp.content = b"imgbytes"
    session.get.return_value = resp
    content = code.download_image("http://x", session=session, timeout=5)
    assert content == b"imgbytes"
    session.get.assert_called_once_with("http://x", timeout=5)
    resp.raise_for_status.assert_called_once()


def test_download_image_raises_on_http_error():
    session = mock.Mock()
    resp = mock.Mock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("boom")
    session.get.return_value = resp
    with pytest.raises(requests.exceptions.HTTPError):
        code.download_image("http://x", session=session)


# ---------------------------------------------------------------------------
# fetch_dataframe
# ---------------------------------------------------------------------------
def test_fetch_dataframe_parses_csv():
    session = mock.Mock()
    resp = mock.Mock()
    resp.text = "latitude,longitude,classification\n1.0,2.0,A\n3.0,4.0,B\n"
    session.get.return_value = resp
    df = code.fetch_dataframe("http://data", session=session)
    assert list(df.columns) == ["latitude", "longitude", "classification"]
    assert len(df) == 2
    session.get.assert_called_once_with("http://data", headers=code.HEADERS)
    resp.raise_for_status.assert_called_once()


def test_fetch_dataframe_custom_headers():
    session = mock.Mock()
    resp = mock.Mock()
    resp.text = "latitude,longitude,classification\n1,2,A\n"
    session.get.return_value = resp
    code.fetch_dataframe("http://data", headers={"X": "Y"}, session=session)
    session.get.assert_called_once_with("http://data", headers={"X": "Y"})


# ---------------------------------------------------------------------------
# process_dataframe
# ---------------------------------------------------------------------------
def _df(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude", "classification"])


def test_process_dataframe_downloads_and_records(tmp_path):
    folders = code.create_classification_folders(base_dir=str(tmp_path))
    df = _df(
        [
            [1.0, 2.0, "a"],
            [3.0, 4.0, "B"],
        ]
    )
    session = mock.Mock()
    resp = mock.Mock()
    resp.content = b"img"
    session.get.return_value = resp

    records = code.process_dataframe(df, folders, api_key="k", session=session)

    assert len(records["A"]) == 1
    assert len(records["B"]) == 1
    assert records["C"] == []
    assert records["A"][0] == {"image 1": "streetview_0.jpg", "lat": 1.0, "long": 2.0}
    # Files written into the right folders.
    assert os.path.isfile(os.path.join(folders["A"], "streetview_0.jpg"))
    assert os.path.isfile(os.path.join(folders["B"], "streetview_1.jpg"))


def test_process_dataframe_skips_unknown_classification(tmp_path):
    folders = code.create_classification_folders(base_dir=str(tmp_path))
    df = _df([[1.0, 2.0, "Z"]])
    session = mock.Mock()

    records = code.process_dataframe(df, folders, session=session)

    assert records == {"A": [], "B": [], "C": []}
    session.get.assert_not_called()


def test_process_dataframe_continues_on_request_error(tmp_path):
    folders = code.create_classification_folders(base_dir=str(tmp_path))
    df = _df(
        [
            [1.0, 2.0, "A"],
            [3.0, 4.0, "A"],
        ]
    )
    session = mock.Mock()
    ok = mock.Mock()
    ok.content = b"img"
    session.get.side_effect = [
        requests.exceptions.ConnectionError("down"),
        ok,
    ]

    records = code.process_dataframe(df, folders, session=session)

    # First row failed, second succeeded -> only one record.
    assert len(records["A"]) == 1
    assert records["A"][0]["image 1"] == "streetview_1.jpg"


# ---------------------------------------------------------------------------
# write_records_csv
# ---------------------------------------------------------------------------
def test_write_records_csv_writes_non_empty_only(tmp_path):
    folders = code.create_classification_folders(base_dir=str(tmp_path))
    image_records = {
        "A": [{"image 1": "streetview_0.jpg", "lat": 1.0, "long": 2.0}],
        "B": [],
        "C": [],
    }
    written = code.write_records_csv(image_records, folders)

    assert set(written) == {"A"}
    csv_path = written["A"]
    assert os.path.isfile(csv_path)
    out = pd.read_csv(csv_path)
    assert list(out.columns) == ["image 1", "lat", "long"]
    assert out.iloc[0]["image 1"] == "streetview_0.jpg"
    # Empty buckets produce no CSV file.
    assert not os.path.isfile(os.path.join(folders["B"], "image_coordinates.csv"))


# ---------------------------------------------------------------------------
# main (integration wiring)
# ---------------------------------------------------------------------------
def test_main_wires_pipeline_together(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _df([[1.0, 2.0, "A"]])

    monkeypatch.setattr(code, "fetch_dataframe", lambda url: df)

    session_resp = mock.Mock()
    session_resp.content = b"img"
    monkeypatch.setattr(code.requests, "get", lambda *a, **k: session_resp)

    code.main()

    assert os.path.isfile(os.path.join(tmp_path, "A", "streetview_0.jpg"))
    assert os.path.isfile(os.path.join(tmp_path, "A", "image_coordinates.csv"))
