# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from unittest.mock import MagicMock, patch

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.generate.base.protocol import DeltaMessage, FunctionCall
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser, ToolParserManager

# https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#model-response-format-1
SIMPLE_FUNCTION_OUTPUT = "get_weather(city='San Francisco', metric='celsius')"
SIMPLE_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "San Francisco", "metric": "celsius"}',
)
MORE_TYPES_FUNCTION_OUTPUT = (
    "register_user(name='John Doe', "
    "age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, "
    "role=None, "
    "passed_test=True, "
    "aliases=['John', 'Johnny'])"
)
MORE_TYPES_FUNCTION_OUTPUT_JSON_LITERALS = (
    "register_user(name='John Doe', "
    "age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, "
    "role=null, "
    "passed_test=true, "
    "aliases=['John', 'Johnny'])"
)
MORE_TYPES_FUNCTION_CALL = FunctionCall(
    name="register_user",
    arguments='{"name": "John Doe", '
    '"age": 37, '
    '"address": {"city": "San Francisco", "state": "CA"}, '
    '"role": null, '
    '"passed_test": true, '
    '"aliases": ["John", "Johnny"]}',
)
PARAMETERLESS_FUNCTION_OUTPUT = "get_weather()"
PARAMETERLESS_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments="{}",
)
EMPTY_DICT_FUNCTION_OUTPUT = "do_something_cool(additional_data={})"
EMPTY_DICT_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"additional_data": {}}',
)
EMPTY_LIST_FUNCTION_OUTPUT = "do_something_cool(steps=[])"
EMPTY_LIST_FUNCTION_CALL = FunctionCall(
    name="do_something_cool",
    arguments='{"steps": []}',
)
ESCAPED_STRING_FUNCTION_OUTPUT = (
    r"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"
)
ESCAPED_STRING_FUNCTION_CALL = FunctionCall(
    name="get_weather",
    arguments='{"city": "Martha\'s Vineyard", "metric": "\\"cool units\\""}',
)
END_TAG_IN_ARG_OUTPUT = (
    "write_file(path='README.md', content='Calls are wrapped in "
    "<function_calls>...</function_calls>.')"
)
END_TAG_IN_ARG_CALL = FunctionCall(
    name="write_file",
    arguments=(
        '{"path": "README.md", "content": "Calls are wrapped in '
        '<function_calls>...</function_calls>."}'
    ),
)


def _char_deltas(tool_parser: ToolParser, model_output: str) -> list[DeltaMessage]:
    """Feed model_output one character at a time and collect emitted deltas."""
    request = ChatCompletionRequest(messages=[], model="test-model")
    previous = ""
    deltas: list[DeltaMessage] = []
    for char in model_output:
        current = previous + char
        message = tool_parser.extract_tool_calls_streaming(
            previous,
            current,
            char,
            [],
            [],
            [],
            request,
        )
        if message is not None:
            deltas.append(message)
        previous = current
    return deltas


@pytest.mark.parametrize(
    "model_output",
    [
        "How can I help you today?",
        "<p>hello",
        "<b>bold</b>",
        "<function> is an HTML-like tag, not a tool call.",
    ],
)
@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(
    streaming: bool, model_output: str, default_tokenizer: TokenizerLike
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content == model_output
    assert len(tool_calls) == 0


TEST_CASES = [
    pytest.param(
        True,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL],
        id="simple_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL],
        id="simple_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT_JSON_LITERALS}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_streaming_json_literals",
    ),
    pytest.param(
        False,
        f"<function_calls>{MORE_TYPES_FUNCTION_OUTPUT_JSON_LITERALS}</function_calls>",
        [MORE_TYPES_FUNCTION_CALL],
        id="more_types_nonstreaming_json_literals",
    ),
    pytest.param(
        True,
        f"<function_calls>{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>",
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>",
        [PARAMETERLESS_FUNCTION_CALL],
        id="parameterless_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{EMPTY_DICT_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{EMPTY_DICT_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_DICT_FUNCTION_CALL],
        id="empty_dict_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{EMPTY_LIST_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{EMPTY_LIST_FUNCTION_OUTPUT}</function_calls>",
        [EMPTY_LIST_FUNCTION_CALL],
        id="empty_list_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{ESCAPED_STRING_FUNCTION_OUTPUT}</function_calls>",
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{ESCAPED_STRING_FUNCTION_OUTPUT}</function_calls>",
        [ESCAPED_STRING_FUNCTION_CALL],
        id="escaped_string_nonstreaming",
    ),
    pytest.param(
        True,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}\n{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        id="parallel_calls_streaming",
    ),
    pytest.param(
        False,
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}\n{MORE_TYPES_FUNCTION_OUTPUT}</function_calls>",
        [SIMPLE_FUNCTION_CALL, MORE_TYPES_FUNCTION_CALL],
        id="parallel_calls_nonstreaming",
    ),
]


@pytest.mark.parametrize("streaming, model_output, expected_tool_calls", TEST_CASES)
def test_tool_call(
    streaming: bool,
    model_output: str,
    expected_tool_calls: list[FunctionCall],
    default_tokenizer: TokenizerLike,
):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content is None
    assert len(tool_calls) == len(expected_tool_calls)
    for actual, expected in zip(tool_calls, expected_tool_calls):
        assert actual.type == "function"
        assert actual.function == expected


def test_streaming_tool_call_with_large_steps(default_tokenizer: TokenizerLike):
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output_deltas = [
        "<function_calls>get_weather(city='San",
        (
            " Francisco', metric='celsius')\n"
            f"{PARAMETERLESS_FUNCTION_OUTPUT}\n"
            f"{EMPTY_LIST_FUNCTION_OUTPUT}</function_calls>"
        ),
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 3
    assert reconstructor.tool_calls[0].function == SIMPLE_FUNCTION_CALL
    assert reconstructor.tool_calls[1].function == PARAMETERLESS_FUNCTION_CALL
    assert reconstructor.tool_calls[2].function == EMPTY_LIST_FUNCTION_CALL


@pytest.mark.parametrize("streaming", [False])
def test_regex_timeout_handling(streaming: bool, default_tokenizer: TokenizerLike):
    """Test regex timeout is handled gracefully."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )

    fake_problematic_input = "hello world[A(A=" + "\t)A(A=,\t" * 2

    # create a mock regex that raises TimeoutError
    mock_regex = MagicMock()
    mock_regex.match.side_effect = TimeoutError("Regex timeout")

    with patch.object(tool_parser, "TOOL_CALL_REGEX", mock_regex):
        content, tool_calls = run_tool_extraction(
            tool_parser, fake_problematic_input, streaming=streaming
        )

        # should treat as regular text when regex times out
        assert content == fake_problematic_input
        assert len(tool_calls) == 0
        mock_regex.match.assert_called_once()


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "leading_prose, body, expected_calls",
    [
        pytest.param(
            "Let me check.\n",
            SIMPLE_FUNCTION_OUTPUT,
            [SIMPLE_FUNCTION_CALL],
            id="prose_single_call",
        ),
        pytest.param(
            "Use 1 < 2 and <function>ok</function> first.\n",
            SIMPLE_FUNCTION_OUTPUT,
            [SIMPLE_FUNCTION_CALL],
            id="prose_with_angle_brackets",
        ),
        pytest.param(
            "Checking both:\n",
            f"{SIMPLE_FUNCTION_OUTPUT}\n{PARAMETERLESS_FUNCTION_OUTPUT}",
            [SIMPLE_FUNCTION_CALL, PARAMETERLESS_FUNCTION_CALL],
            id="prose_parallel_calls",
        ),
    ],
)
def test_leading_prose_before_tool_calls(
    streaming: bool,
    leading_prose: str,
    body: str,
    expected_calls: list[FunctionCall],
    default_tokenizer: TokenizerLike,
):
    """Leading prose before <function_calls> streams as content and preserves
    tool extraction with streaming/non-streaming parity."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = f"{leading_prose}<function_calls>{body}</function_calls>"

    content, tool_calls = run_tool_extraction(
        tool_parser, model_output, streaming=streaming
    )

    assert content == leading_prose
    assert [tc.function for tc in tool_calls] == expected_calls


def test_streaming_split_tags_and_closing_suffix(default_tokenizer: TokenizerLike):
    """Streaming handles prose+<function_calls> and )+</function_calls> in one delta."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output_deltas = [
        "Let me ",
        "check.\n<func",
        "tion_calls>get_weather(city='San",
        " Francisco', metric='celsius')</function",
        "_calls>\n",
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_deltas, assert_one_tool_per_delta=False
    )

    assert reconstructor.other_content == "Let me check.\n"
    assert [tc.function for tc in reconstructor.tool_calls] == [SIMPLE_FUNCTION_CALL]


def test_streaming_no_unexpected_ast_errors(
    default_tokenizer: TokenizerLike, caplog: pytest.LogCaptureFixture
):
    """Char-by-char streaming logs no UnexpectedAstError on call names or end tags."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = (
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}\n"
        f"{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>"
    )

    with caplog.at_level(logging.ERROR, logger="vllm.tool_parsers.olmo3_tool_parser"):
        reconstructor = run_tool_extraction_streaming(
            tool_parser, list(model_output), assert_one_tool_per_delta=False
        )

    assert caplog.records == []
    assert reconstructor.other_content == ""
    assert [tc.function for tc in reconstructor.tool_calls] == [
        SIMPLE_FUNCTION_CALL,
        PARAMETERLESS_FUNCTION_CALL,
    ]


def test_streaming_closing_tag_prefix_inside_string_arg(
    default_tokenizer: TokenizerLike, caplog: pytest.LogCaptureFixture
):
    """A '</' substring inside a string argument is held briefly, not dropped.

    The finished call matches the pre-fix parser. This also asserts that the
    hold-back does not log UnexpectedAstError while '</' is still streaming.
    """
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = "<function_calls>render(html='<div></div>')</function_calls>"

    with caplog.at_level(logging.ERROR, logger="vllm.tool_parsers.olmo3_tool_parser"):
        reconstructor = run_tool_extraction_streaming(
            tool_parser, list(model_output), assert_one_tool_per_delta=False
        )

    assert caplog.records == []
    assert reconstructor.other_content == ""
    assert [tc.function for tc in reconstructor.tool_calls] == [
        FunctionCall(name="render", arguments='{"html": "<div></div>"}')
    ]


def test_streaming_end_tag_inside_string_argument(
    default_tokenizer: TokenizerLike, caplog: pytest.LogCaptureFixture
):
    """A full closing tag inside a string argument stays part of that argument."""
    model_output = f"<function_calls>{END_TAG_IN_ARG_OUTPUT}</function_calls>\nThanks"
    inner_end = model_output.find("</function_calls>") + len("</function_calls>")
    real_end = model_output.rfind(">") + 1
    streams = [
        list(model_output),
        [
            model_output[:inner_end],
            model_output[inner_end:real_end],
            model_output[real_end:],
        ],
    ]

    for deltas in streams:
        tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
            default_tokenizer
        )
        with caplog.at_level(
            logging.ERROR, logger="vllm.tool_parsers.olmo3_tool_parser"
        ):
            reconstructor = run_tool_extraction_streaming(
                tool_parser, deltas, assert_one_tool_per_delta=False
            )
        assert caplog.records == []
        caplog.clear()
        assert reconstructor.other_content == ""
        assert [tc.function for tc in reconstructor.tool_calls] == [END_TAG_IN_ARG_CALL]

    tool_parser = ToolParserManager.get_tool_parser("olmo3")(default_tokenizer)
    content, tool_calls = run_tool_extraction(tool_parser, model_output, streaming=True)
    assert content is None
    assert [tc.function for tc in tool_calls] == [END_TAG_IN_ARG_CALL]


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize(
    "model_output",
    [
        "Explain this.\n<function_calls>this is not a call</function_calls>",
        "Hello<function_calls></function_calls>",
        "Hello<function_calls></function_calls> tail",
    ],
)
def test_non_call_function_calls_block_is_content(
    streaming: bool,
    model_output: str,
    default_tokenizer: TokenizerLike,
    caplog: pytest.LogCaptureFixture,
):
    """A <function_calls> block that is not a call stays in content."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )

    with caplog.at_level(logging.ERROR, logger="vllm.tool_parsers.olmo3_tool_parser"):
        content, tool_calls = run_tool_extraction(
            tool_parser, model_output, streaming=streaming
        )

    assert content == model_output
    assert tool_calls == []
    assert caplog.records == []


def test_streaming_non_call_block_char_deltas_do_not_log(
    default_tokenizer: TokenizerLike, caplog: pytest.LogCaptureFixture
):
    """Char deltas of a non-call block log nothing and keep the whole string."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = "Explain this.\n<function_calls>this is not a call</function_calls>"

    with caplog.at_level(logging.ERROR, logger="vllm.tool_parsers.olmo3_tool_parser"):
        deltas = _char_deltas(tool_parser, model_output)

    assert caplog.records == []
    assert "".join(delta.content or "" for delta in deltas) == model_output
    assert all(not delta.tool_calls for delta in deltas)


def test_streaming_empty_content_only_after_end_tag(
    default_tokenizer: TokenizerLike,
):
    """Name tokens and a partial closing tag do not emit empty content deltas."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = (
        f"<function_calls>{SIMPLE_FUNCTION_OUTPUT}\n"
        f"{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>"
    )

    deltas = _char_deltas(tool_parser, model_output)
    empty_content = [
        delta for delta in deltas if delta.content == "" and not delta.tool_calls
    ]

    assert len(empty_content) == 1
    assert "".join(delta.content or "" for delta in deltas) == ""
    reconstructor = run_tool_extraction_streaming(
        ToolParserManager.get_tool_parser("olmo3")(default_tokenizer),
        list(model_output),
        assert_one_tool_per_delta=False,
    )
    assert [tc.function for tc in reconstructor.tool_calls] == [
        SIMPLE_FUNCTION_CALL,
        PARAMETERLESS_FUNCTION_CALL,
    ]


def test_streaming_drops_text_after_completed_call(
    default_tokenizer: TokenizerLike, caplog: pytest.LogCaptureFixture
):
    """Prose after a finished call block is not content and is not reparsed."""
    tool_parser: ToolParser = ToolParserManager.get_tool_parser("olmo3")(
        default_tokenizer
    )
    model_output = (
        f"<function_calls>{PARAMETERLESS_FUNCTION_OUTPUT}</function_calls>\nThanks"
    )

    with caplog.at_level(logging.ERROR, logger="vllm.tool_parsers.olmo3_tool_parser"):
        reconstructor = run_tool_extraction_streaming(
            tool_parser, list(model_output), assert_one_tool_per_delta=False
        )

    assert caplog.records == []
    assert reconstructor.other_content == ""
    assert [tc.function for tc in reconstructor.tool_calls] == [
        PARAMETERLESS_FUNCTION_CALL
    ]
