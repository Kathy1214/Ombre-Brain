import asyncio
import sys
import types
from unittest.mock import AsyncMock


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


def test_analyze_allows_room_for_tags(tmp_path):
    dehydrator = make_dehydrator(tmp_path)
    dehydrator.model = "deepseek-v4-pro"
    dehydrator.max_tokens = 1024
    dehydrator.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=AsyncMock(
                    return_value=types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(
                                    content='{"domain":["测试"],"valence":0.5,"arousal":0.4,"tags":["自动分类"],"suggested_name":"分类测试"}'
                                )
                            )
                        ]
                    )
                )
            )
        )
    )

    result = asyncio.run(dehydrator._api_analyze("测试 DeepSeek 自动分类"))

    assert result["domain"] == ["测试"]
    assert dehydrator.client.chat.completions.create.call_args.kwargs["max_tokens"] == 1024
