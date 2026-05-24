"""Schema-strict prompt and repair helpers for structured extraction.

The repair layer is intentionally conservative: it can project model JSON into
known schema fields and normalize field aliases, but it does not invent values
for evaluation. Missing-field filling is available for serving contracts only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any


FIELD_ALIASES = {
    "创办者": ["创始人", "创建者", "建立者", "发起人"],
    "位于": ["位置", "地点", "所在地", "地址", "所在", "主办城市", "举办地", "主办地"],
    "发现者或发明者": ["发现者", "发明者", "发现人", "发明人"],
    "创建或成立时间": ["建造时间", "创建时间", "建立时间", "建成时间", "启用时间"],
    "发生时间": ["时间", "举办时间", "开始时间", "结束时间"],
    "发生地点": ["地点", "位置", "举办地", "主办城市", "主办地"],
    "常见并发症": ["并发症", "合并症"],
    "症状": ["主要症状", "常见症状", "临床表现"],
    "治疗方法": ["治疗", "疗法", "治疗方式"],
    "成就": ["获奖", "奖项", "荣誉", "成就奖"],
    "子组织": ["旗下", "下属组织", "子公司", "分支机构", "所属公司"],
    "别名": ["又称", "又名", "别称", "也叫"],
    "组成": ["成分", "原料", "组成部分", "构成"],
    "用途": ["应用", "应用领域", "主要用途", "作用"],
    "所属科室": ["科室", "所属科"],
    "线路": ["所属线路", "路线"],
    "车站等级": ["等级"],
    "开通时间": ["启用时间", "运营时间", "通车时间"],
    "保护级别": ["濒危等级", "保护等级"],
    "成立时间": ["成立年份", "创立时间", "创建时间", "建立时间"],
    "出生地": ["出生地点", "籍贯"],
    "出生日期": ["生日", "出生年月"],
    "参与者": ["参加者", "参赛者", "参赛方", "参与方", "主办方"],
    "起因": ["原因", "导火索"],
    "导致": ["结果", "后果"],
    "作者": ["创作者", "编写者"],
    "出版时间": ["发布时间", "发行时间"],
    "出版商": ["出版社"],
    "材料": ["材质"],
    "长度": ["长"],
    "宽度": ["宽"],
    "高度": ["高"],
    "面积": ["占地面积"],
    "直径": ["半径"],
    "属于": ["所属", "分类"],
}


def clean_model_output(raw: str) -> str:
    text = raw.strip()
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def _merge_pair_dict(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in pairs:
        if key not in merged:
            merged[key] = value
            continue
        old = merged[key]
        if isinstance(old, dict) and isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if inner_key in old:
                    old[inner_key] = _merge_values(old[inner_key], inner_value)
                else:
                    old[inner_key] = inner_value
        else:
            merged[key] = _merge_values(old, value)
    return merged


def try_parse_json(text: str) -> tuple[Any, bool]:
    try:
        return json.loads(clean_model_output(text), object_pairs_hook=_merge_pair_dict), True
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, False


def normalize_field_name(field: str, aliases: dict[str, list[str]] | None = None) -> str:
    alias_map = aliases or FIELD_ALIASES
    for canonical, variants in alias_map.items():
        if field == canonical or field in variants:
            return canonical
    return field


def _value_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _merge_values(old: Any, new: Any) -> Any:
    if old == new:
        return old
    merged = []
    for value in _value_list(old) + _value_list(new):
        if value not in merged:
            merged.append(value)
    return merged


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _flatten_scalar_list(values: Iterable[Any]) -> list[Any]:
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, list):
            for item in _flatten_scalar_list(value):
                if item not in flattened:
                    flattened.append(item)
        elif not isinstance(value, dict) and not _is_empty(value) and value not in flattened:
            flattened.append(value)
    return flattened


def _dict_scalars(value: dict[str, Any]) -> list[str]:
    scalars = []
    for item in value.values():
        if isinstance(item, dict):
            continue
        if isinstance(item, list):
            scalars.extend(str(v) for v in _flatten_scalar_list(item))
        elif not _is_empty(item):
            scalars.append(str(item))
    return scalars


def _dict_contains_allowed_field(
    value: Any,
    allowed: set[str],
    *,
    use_aliases: bool,
    aliases: dict[str, list[str]] | None,
) -> bool:
    if isinstance(value, dict):
        for key, inner in value.items():
            target = normalize_field_name(str(key), aliases) if use_aliases else str(key)
            if target in allowed:
                return True
            if _dict_contains_allowed_field(inner, allowed, use_aliases=use_aliases, aliases=aliases):
                return True
    elif isinstance(value, list):
        return any(_dict_contains_allowed_field(item, allowed, use_aliases=use_aliases, aliases=aliases) for item in value)
    return False


def normalize_value_for_type(value: Any, expected_type: str | None) -> Any | None:
    """Return a schema-consumable value, or None when the value is structural."""
    if isinstance(value, dict):
        return None

    if expected_type == "string":
        if isinstance(value, list):
            scalars = _flatten_scalar_list(value)
            return str(scalars[0]) if scalars else None
        if _is_empty(value):
            return None
        return str(value)

    if expected_type == "string_or_list":
        if isinstance(value, list):
            scalars = [str(item) for item in _flatten_scalar_list(value)]
            if not scalars:
                return None
            return scalars if len(scalars) > 1 else scalars[0]
        if _is_empty(value):
            return None
        return str(value)

    if isinstance(value, list):
        scalars = _flatten_scalar_list(value)
        if not scalars:
            return None
        return scalars if len(scalars) > 1 else scalars[0]
    if _is_empty(value):
        return None
    return value


def normalize_direct_field_value(
    field: str,
    value: Any,
    expected_type: str | None,
    *,
    field_is_required: bool,
    has_inner_schema_fields: bool,
) -> Any | None:
    if isinstance(value, dict):
        if has_inner_schema_fields and field not in {"分布"}:
            return None
        scalars = _dict_scalars(value)
        if not scalars:
            return None
        if expected_type == "string":
            return " ".join(scalars)
        return scalars if len(scalars) > 1 else scalars[0]
    return normalize_value_for_type(value, expected_type)


def coerce_enum_value(field: str, value: Any, allowed_values: list[str]) -> Any:
    allowed = set(allowed_values)

    def coerce_one(item: Any) -> Any:
        if item in allowed:
            return item
        text = str(item)
        for allowed_item in allowed_values:
            if allowed_item in text:
                return allowed_item
        if field == "用途" and "宗教" in allowed and any(word in text for word in ["教堂", "天主教", "寺", "庙"]):
            return "宗教"
        if field == "所属科室" and "内科" in allowed and any(word in text for word in ["呼吸", "肺", "心血管", "消化"]):
            return "内科"
        if field == "材料":
            material_map = [
                ("金属", ["金属", "铁", "钢", "铜", "银", "金"]),
                ("木材", ["木", "竹"]),
                ("石材", ["石", "磁石", "玉"]),
                ("陶瓷", ["陶", "瓷"]),
                ("塑料", ["塑料"]),
                ("玻璃", ["玻璃"]),
                ("橡胶", ["橡胶"]),
                ("纤维", ["纤维", "布", "棉", "丝"]),
            ]
            for canonical, words in material_map:
                if canonical in allowed and any(word in text for word in words):
                    return canonical
        return item

    if isinstance(value, list):
        coerced = []
        for item in value:
            next_item = coerce_one(item)
            if next_item not in coerced:
                coerced.append(next_item)
        return coerced if len(coerced) > 1 else coerced[0] if coerced else value
    return coerce_one(value)


def default_missing_value(expected_type: str | None) -> Any:
    if expected_type == "string_or_list":
        return []
    return None


def repair_to_schema(
    parsed: Any,
    schema_def: dict[str, Any],
    *,
    use_aliases: bool = True,
    fill_missing: bool = False,
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Project arbitrary JSON into schema fields.

    The function recurses through entity-nested JSON and flat JSON alike. When
    a schema field's value is itself an object, the object is treated as a
    container and scanned for usable inner fields instead of being accepted as
    a scalar value.
    """
    allowed_fields = list(schema_def.get("allowed_fields") or schema_def.get("required_fields") or [])
    allowed = set(allowed_fields)
    required = list(schema_def.get("required_fields") or [])
    types = schema_def.get("types", {}) or {}
    required_set = set(required)
    repaired: dict[str, Any] = {}

    def add_field(field: str, value: Any) -> bool:
        target = normalize_field_name(field, aliases) if use_aliases else field
        if target not in allowed:
            return False
        has_inner = _dict_contains_allowed_field(value, allowed, use_aliases=use_aliases, aliases=aliases)
        normalized = normalize_direct_field_value(
            target,
            value,
            types.get(target),
            field_is_required=target in required_set,
            has_inner_schema_fields=has_inner,
        )
        if normalized is None:
            return False
        if target in repaired:
            repaired[target] = _merge_values(repaired[target], normalized)
        else:
            repaired[target] = normalized
        return True

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                add_field(str(key), value)
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    walk(item)

    walk(parsed)

    if fill_missing:
        for field in required:
            repaired.setdefault(field, default_missing_value(types.get(field)))

    for field, allowed_values in (schema_def.get("enum_constraints", {}) or {}).items():
        if field in repaired:
            repaired[field] = coerce_enum_value(field, repaired[field], allowed_values)

    return {field: repaired[field] for field in allowed_fields if field in repaired}


def enum_values_ok(fields: dict[str, Any], schema_def: dict[str, Any]) -> bool:
    enums = schema_def.get("enum_constraints", {}) or {}
    for field, allowed_values in enums.items():
        if field not in fields:
            continue
        allowed = set(allowed_values)
        for value in _value_list(fields[field]):
            if value is not None and value not in allowed:
                return False
    return True


def score_repaired_fields(fields: dict[str, Any], schema_def: dict[str, Any]) -> dict[str, Any]:
    required = list(schema_def.get("required_fields") or [])
    allowed = set(schema_def.get("allowed_fields") or required)
    missing = [field for field in required if _is_empty(fields.get(field))]
    extra = [field for field in fields if field not in allowed]
    enum_ok = enum_values_ok(fields, schema_def)
    return {
        "missing_fields": missing,
        "extra_fields": extra,
        "enum_ok": enum_ok,
        "schema_strict": not missing and not extra and enum_ok,
    }


def build_schema_strict_messages(prompt_item: dict[str, Any]) -> list[dict[str, str]]:
    instruction = prompt_item.get("instruction", "")
    schema_list = prompt_item.get("schema", [])
    input_text = prompt_item.get("input", "")
    schema_def = prompt_item.get("schema_def", {})
    required = schema_def.get("required_fields", [])

    parts = []
    if instruction:
        parts.append(instruction)
    if schema_list and "Schema:" not in instruction and "schema" not in instruction.lower():
        parts.append(f"Schema: {json.dumps(schema_list, ensure_ascii=False)}")
    if "文本:" not in instruction and "从文本" not in instruction:
        parts.append(f"文本: {input_text}")
    else:
        parts.append(input_text)

    contract = [
        "输出契约:",
        "1. 只输出一个 JSON object，不要输出 markdown、解释或多余文字。",
        "2. 顶层 key 必须严格来自 Schema 字段名，不要把实体名称作为顶层 key。",
        "3. 字段名必须原样复制 Schema 中的中文字段，不要翻译、改写或使用近义词。",
        "4. 禁止输出 Schema 之外的字段。",
        "5. 所有 required 字段都必须出现；无法确定的字段请填 null，列表字段无法确定时填 []。",
        "正确格式示例: {\"出生地\": \"浙江绍兴\", \"出生日期\": \"1881年9月25日\", \"职业\": \"作家\"}",
        "错误格式示例: {\"鲁迅\": {\"出生地\": \"浙江绍兴\"}}。不要输出这种实体嵌套格式。",
        f"Allowed fields: {json.dumps(schema_list, ensure_ascii=False)}",
    ]
    if required:
        contract.append(f"Required fields: {json.dumps(required, ensure_ascii=False)}")
    parts.append("\n".join(contract))

    return [
        {
            "role": "system",
            "content": "你是一个 schema-strict JSON 生成器。输出必须是单个 JSON object；顶层 key 只能来自用户给定 Schema；禁止实体名做顶层 key；禁止 Schema 外字段；缺失信息用 null 或 [] 补齐。",
        },
        {"role": "user", "content": "\n".join(parts)},
    ]
