# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from collections.abc import Sequence

import regex as re
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    compute_tool_delta,
    handle_single_tool,
    make_valid_python,
    partial_tag_overlap,
)

logger = init_logger(__name__)

_START_TAG = "<function_calls>"
_END_TAG = "</function_calls>"


def _split_call_body(body: str) -> tuple[str, bool]:
    """Return the call text and whether a closing tag suffix is present.

    Only a suffix counts, after trailing whitespace. A closing tag inside a
    string argument is not a suffix once later chunks arrive, and the full
    buffer is reparsed, so that tag stays in the argument. A trailing prefix
    of the closing tag is withheld so partial tags are not parsed.
    """
    stripped = body.rstrip()
    if stripped.endswith(_END_TAG):
        return stripped[: -len(_END_TAG)], True
    overlap = partial_tag_overlap(stripped, _END_TAG)
    if overlap:
        return stripped[:-overlap], False
    return body, False


class Olmo3PythonicToolParser(ToolParser):
    """Tool call parser for Olmo 3 models that produce tool calls as
    newline-separated pythonic strings.
    Used when --enable-auto-tool-choice --tool-call-parser pythonic are all set
    Code copied from pythonic_tool_parser.py and updated to handle
    - newline separated pythonic tool calls.
    - argument values being null/true/false instead of Pythonic literals.
    """

    # TODO(mdepinet): Possible future improvements:
    #   1. Support text + tools separated by either <|python_tag|> or \n\n
    #   2. Support tools outside of a list (or separated by a semicolon).
    #      This depends on item 1 for consistent streaming.
    # Neither of these are necessary for e.g. ToolACE, but both would help make
    # Llama3.2 models more reliable.

    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)
        self._sent_content_idx: int = 0
        # Once a closed block is known not to be a call, later chunks are text.
        self._pass_through_content: bool = False
        # Index just after a finished call block. Text beyond it is trailing
        # prose and is not parsed again.
        self._closed_upto: int | None = None

    # Rename for readability. This is NOT a tool id.
    @property
    def current_tool_index(self) -> int:
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        self.current_tool_id = value

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract the tool calls from a complete model response."""
        original_model_output = model_output
        content: str | None = None
        # Remove xml tags.
        match = re.search(
            r"<function_calls>(.*?)</function_calls>", model_output, re.DOTALL
        )
        if match:
            content = original_model_output[: match.start()] or None
            model_output = match.group(1).strip()
        # Make the newline separated function calls into a list.
        model_output = ", ".join(
            [line.strip() for line in model_output.splitlines() if line.strip()]
        )
        model_output = f"[{model_output}]"

        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = (
                self.TOOL_CALL_REGEX.match(
                    model_output, timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
                )
                is not None
            )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug(
                "Regex timeout occurred when matching user input: %s", model_output
            )

        if not is_tool_call_pattern:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=original_model_output
            )

        try:
            module = ast.parse(model_output)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[
                        handle_single_tool(e)  # type: ignore
                        for e in parsed.elts
                    ],
                    content=content,
                )
            else:
                raise UnexpectedAstError("Tool output must be a list of function calls")
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # Treat as regular text
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=original_model_output
            )

    def _extract_content(self, current_text: str) -> str | None:
        """Return unsent leading prose, holding back partial start-tag prefixes."""
        start_idx = current_text.find(_START_TAG)
        if start_idx == -1:
            overlap = partial_tag_overlap(current_text, _START_TAG)
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = start_idx

        if sendable_idx > self._sent_content_idx:
            content = current_text[self._sent_content_idx : sendable_idx]
            self._sent_content_idx = sendable_idx
            return content
        return None

    def _begin_pass_through(
        self, current_text: str, content: str | None
    ) -> DeltaMessage | None:
        """Emit every not-yet-sent character and keep doing so."""
        self._pass_through_content = True
        rest = current_text[self._sent_content_idx :]
        self._sent_content_idx = len(current_text)
        if content:
            rest = content + rest
        return DeltaMessage(content=rest) if rest else None

    def _no_tools_streamed(self) -> bool:
        return self.current_tool_id < 0 and not any(self.streamed_args_for_tool)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if not previous_text:
            self._sent_content_idx = 0
            self._pass_through_content = False
            self._closed_upto = None

        if self._pass_through_content:
            return self._begin_pass_through(current_text, None)
        if self._closed_upto is not None and len(current_text) > self._closed_upto:
            return None

        content = self._extract_content(current_text)
        start_idx = current_text.find(_START_TAG)
        if start_idx == -1:
            return DeltaMessage(content=content) if content is not None else None

        body, has_end_tag = _split_call_body(
            current_text[start_idx + len(_START_TAG) :]
        )
        # A block with no call syntax is ordinary text once it is closed.
        # Until then, hold the tag back so a real call can still follow.
        if "(" not in body:
            if has_end_tag:
                return self._begin_pass_through(current_text, content)
            return DeltaMessage(content=content) if content is not None else None

        try:
            valid_and_added_text = make_valid_python(body)
            if valid_and_added_text is None:
                if has_end_tag and self._no_tools_streamed():
                    return self._begin_pass_through(current_text, content)
                return DeltaMessage(content=content) if content is not None else None
            valid_text, added_text = valid_and_added_text

            # Make the newline separated function calls into a list.
            valid_text = ", ".join(
                [line.strip() for line in valid_text.splitlines() if line.strip()]
            )
            valid_text = f"[{valid_text}]"
            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List):
                raise UnexpectedAstError(
                    "Tool output must be a sequence of newline-separated calls"
                )
            call_nodes = parsed.elts
            if (
                not has_end_tag
                and not added_text
                and call_nodes
                and isinstance(call_nodes[-1], ast.Name | ast.Attribute)
            ):
                # Function name is still streaming before its opening '('.
                call_nodes = call_nodes[:-1]
            if not all(isinstance(e, ast.Call) for e in call_nodes):
                raise UnexpectedAstError(
                    "Tool output must be a sequence of newline-separated calls"
                )
            tool_calls = [
                handle_single_tool(e)  # type: ignore
                for e in call_nodes
            ]

            tool_deltas = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = index < len(tool_calls) - 1 or ")" not in added_text
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-1] if not new_call_complete else ""
                if not new_call_complete and added_text[-1] == ")":
                    # Function call is incomplete. Withhold the closing bracket.
                    withheld_suffix = withheld_suffix + "}"
                # Strings get single quotes in the model-produced string.
                # JSON requires double quotes.
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = compute_tool_delta(
                    self.streamed_args_for_tool[index], new_call, index, withheld_suffix
                )

                if delta is not None:
                    tool_deltas.append(delta)
                    if (
                        delta.function is not None
                        and delta.function.arguments is not None
                    ):
                        self.streamed_args_for_tool[index] += delta.function.arguments

            # A suffix tag closed this block. Do not freeze when the suffix was
            # only the end of an unclosed string: added quotes mean the next
            # chunk still belongs to the argument.
            if has_end_tag and "'" not in added_text and '"' not in added_text:
                self._closed_upto = len(current_text.rstrip())

            # HACK: serving_chat.py inspects the internal state of tool parsers
            # when determining its final streaming delta, automatically
            # adding autocompleted JSON.
            # These two lines avoid that nonsense while ensuring finish_reason
            # is set to tool_calls when at least one tool is called.
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(content=content, tool_calls=tool_deltas)
            elif content is not None:
                return DeltaMessage(content=content)
            elif has_end_tag and not added_text and self.current_tool_id > 0:
                # One empty delta once the closing tag has arrived. Name
                # tokens and partial closing tags are not finished calls.
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            if has_end_tag and self._no_tools_streamed():
                return self._begin_pass_through(current_text, content)
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return DeltaMessage(content=content) if content is not None else None
