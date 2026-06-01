import sys
import types


openai_stub = types.ModuleType("openai")
openai_stub.AsyncOpenAI = object
sys.modules.setdefault("openai", openai_stub)

from dehydrator import Dehydrator


def make_dehydrator(tmp_path):
    buckets = tmp_path / "buckets"
    buckets.mkdir()
    return Dehydrator({
        "buckets_dir": str(buckets),
        "dehydration": {"api_key": "", "base_url": "https://example.com", "model": "test"},
    })


def test_parse_analysis_extracts_json_from_prose(tmp_path):
    dehydrator = make_dehydrator(tmp_path)

    result = dehydrator._parse_analysis(
        '好的，结果如下：\n{"domain":["工作"],"valence":0.6,"arousal":0.4,'
        '"tags":["项目"],"suggested_name":"项目推进"}\n希望有帮助。'
    )

    assert result["domain"] == ["工作"]
    assert result["tags"] == ["项目"]
    assert result["suggested_name"] == "项目推进"


def test_parse_digest_accepts_wrapped_items(tmp_path):
    dehydrator = make_dehydrator(tmp_path)

    result = dehydrator._parse_digest(
        '```json\n{"items":[{"name":"晚间记录","content":"今天调好了部署。",'
        '"domain":["项目"],"valence":0.7,"arousal":0.5,"tags":["部署"],"importance":6}]}\n```'
    )

    assert len(result) == 1
    assert result[0]["name"] == "晚间记录"
    assert result[0]["domain"] == ["项目"]
    assert result[0]["importance"] == 6
