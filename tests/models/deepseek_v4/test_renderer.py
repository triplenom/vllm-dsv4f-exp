# SPDX-License-Identifier: Apache-2.0
"""Renderer-level tests for DeepSeek V4 Vision-Exp image content handling.

The chat renderer receives "<##IMAGE##>" markers inline in the text (from
parse_chat_messages with content_format="string") and must rewrite them to
DeepSeek's official "<｜deepseek_image｜>" placeholder before the chat
template runs, leaving everything else untouched.
"""

import pytest

pytest.importorskip("vllm.renderers.deepseek_v4", reason="needs the vLLM renderer stack")

from vllm.renderers.deepseek_v4 import (
    _FRAMEWORK_IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    _substitute_image_placeholders,
)


def test_placeholder_constant_matches_reference():
    # DeepSeek official encoding (encoding/encoding_dsv4.py).
    assert IMAGE_PLACEHOLDER == "<｜deepseek_image｜>"
    assert _FRAMEWORK_IMAGE_PLACEHOLDER == "<##IMAGE##>"


def test_substitution_rewrites_image_markers_only():
    conversation = [
        {"role": "user", "content": f"before{_FRAMEWORK_IMAGE_PLACEHOLDER}after"},
        {"role": "assistant", "content": "no images here"},
        {"role": "user", "content": f"{_FRAMEWORK_IMAGE_PLACEHOLDER}{_FRAMEWORK_IMAGE_PLACEHOLDER}"},
    ]
    _substitute_image_placeholders(conversation)
    assert conversation[0]["content"] == f"before{IMAGE_PLACEHOLDER}after"
    assert conversation[1]["content"] == "no images here"
    assert conversation[2]["content"] == IMAGE_PLACEHOLDER * 2


def test_text_only_conversation_unchanged():
    conversation = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "describe this"},
    ]
    snapshot = [dict(m) for m in conversation]
    _substitute_image_placeholders(conversation)
    assert conversation == snapshot


def test_non_string_content_untouched():
    # Tool messages with structured content lists are left alone.
    conversation = [
        {"role": "tool", "content": [{"type": "text", "text": "result"}]},
    ]
    snapshot = [dict(m) for m in conversation]
    _substitute_image_placeholders(conversation)
    assert conversation == snapshot
