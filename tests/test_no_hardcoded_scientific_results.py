from pathlib import Path


def test_no_hardcoded_proxy_validity_in_postfit():
    text = Path("src/postfit_sqlite.py").read_text()
    assert 'proxy_validity_class": "moderate"' not in text
    assert 'proxy_pearson_r": 0.75' not in text


def test_no_file_specific_atf_alias():
    text = Path("src/atf_features.py").read_text()
    assert 'path.name == "1_DH_1.atf"' not in text
