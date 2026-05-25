"""OpenAI-compatible VLM provider for local and self-hosted models."""

import json
import os
from typing import Any

from openai import OpenAI
from typing_extensions import override

from askui.model_providers.vlm_provider import VlmProvider
from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    UsageParam,
)
from askui.models.shared.prompts import SystemPrompt
from askui.models.shared.tools import ToolCollection


class OpenAICompatibleVlmProvider(VlmProvider):
    """VLM provider backed by an OpenAI-compatible chat API.

    This is useful for local or self-hosted servers that expose an OpenAI-style
    `/v1/chat/completions` endpoint. The provider uses the message shape expected
    by AskUI and maps OpenAI tool calls back into `MessageParam` objects.

    Args:
        model_id (str | None, optional): Model identifier. Falls back to
            `VLM_PROVIDER_MODEL_ID` or `"local-model"`.
        base_url (str | None, optional): Base URL for the OpenAI-compatible API.
            If a host URL is provided without `/v1`, it is normalized to add
            `/v1` automatically.
        api_key (str | None, optional): API key. Falls back to
            `OPENAI_API_KEY` or `"local"`.
        client (OpenAI | None, optional): Pre-configured OpenAI client.
    """

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._model_id_value = (
            model_id
            or os.environ.get("VLM_PROVIDER_MODEL_ID")
            or "local-model"
        )
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY") or "local",
                base_url=self._normalize_base_url(base_url),
            )

    @staticmethod
    def _normalize_base_url(base_url: str | None) -> str | None:
        if base_url is None:
            return None

        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"

    @property
    @override
    def model_id(self) -> str:
        return self._model_id_value

    @override
    def create_message(
        self,
        messages: list[MessageParam],
        tools: ToolCollection | None = None,
        max_tokens: int | None = None,
        system: SystemPrompt | None = None,
        thinking: dict[str, Any] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> MessageParam:
        create_kwargs: dict[str, Any] = {
            "model": self._model_id_value,
            "messages": self._to_openai_messages(messages, system),
            "stream": False,
        }

        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if tools is not None:
            create_kwargs["tools"] = self._to_openai_tools(tools)
        if tool_choice is not None:
            create_kwargs["tool_choice"] = tool_choice

        extra_body: dict[str, Any] = {}
        if provider_options is not None:
            extra_body.update(provider_options)
        if thinking is not None:
            extra_body["thinking"] = thinking
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**create_kwargs)
        choice = response.choices[0]
        message = choice.message

        content_blocks = self._parse_content(message.content)
        tool_calls = self._parse_tool_calls(getattr(message, "tool_calls", None))

        if tool_calls:
            content_blocks.extend(tool_calls)
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        usage = getattr(response, "usage", None)
        usage_param = None
        if usage is not None:
            usage_param = UsageParam(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )

        return MessageParam(
            role="assistant",
            content=content_blocks,
            stop_reason=stop_reason,
            usage=usage_param,
        )

    def _to_openai_tools(self, tools: ToolCollection) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []
        for tool in tools.to_params():
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object"}),
                    },
                }
            )
        return openai_tools

    def _to_openai_messages(
        self,
        messages: list[MessageParam],
        system: SystemPrompt | None,
    ) -> list[dict[str, Any]]:
        openai_messages: list[dict[str, Any]] = []

        if system is not None:
            openai_messages.append({"role": "system", "content": str(system)})

        for message in messages:
            if isinstance(message.content, str):
                openai_messages.append(
                    {"role": message.role, "content": message.content}
                )
                continue

            tool_messages: list[dict[str, Any]] = []
            assistant_tool_calls: list[dict[str, Any]] = []
            text_or_image_blocks: list[TextBlockParam | ImageBlockParam] = []

            for block in message.content:
                if isinstance(block, ToolResultBlockParam):
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": self._tool_result_to_content(block),
                        }
                    )
                elif isinstance(block, ToolUseBlockParam):
                    assistant_tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
                elif isinstance(block, (TextBlockParam, ImageBlockParam)):
                    text_or_image_blocks.append(block)

            if text_or_image_blocks:
                openai_messages.append(
                    {
                        "role": message.role,
                        "content": self._content_blocks_to_openai_parts(
                            text_or_image_blocks
                        ),
                    }
                )

            if assistant_tool_calls:
                openai_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": assistant_tool_calls,
                        "content": None,
                    }
                )

            openai_messages.extend(tool_messages)

        return openai_messages

    def _content_blocks_to_openai_parts(
        self,
        blocks: list[TextBlockParam | ImageBlockParam],
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, TextBlockParam):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlockParam):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_source_to_url(block.source)
                        },
                    }
                )
        return parts

    def _image_source_to_url(self, source: Any) -> str:
        if isinstance(source, Base64ImageSourceParam):
            return f"data:{source.media_type};base64,{source.data}"
        return source.url

    def _tool_result_to_content(self, block: ToolResultBlockParam) -> str:
        if isinstance(block.content, str):
            return block.content
        content_parts: list[str] = []
        for item in block.content:
            if isinstance(item, TextBlockParam):
                content_parts.append(item.text)
            elif isinstance(item, ImageBlockParam):
                content_parts.append(self._image_source_to_url(item.source))
        return "\n".join(content_parts)

    def _parse_content(self, content: Any) -> list[TextBlockParam | ToolUseBlockParam]:
        if content is None:
            return []
        if isinstance(content, str):
            return [TextBlockParam(text=content)]
        if isinstance(content, list):
            blocks: list[TextBlockParam | ToolUseBlockParam] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    blocks.append(TextBlockParam(text=item.get("text", "")))
            return blocks
        return [TextBlockParam(text=str(content))]

    def _parse_tool_calls(self, tool_calls: Any) -> list[ToolUseBlockParam]:
        if not tool_calls:
            return []

        tool_use_blocks: list[ToolUseBlockParam] = []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", tool_call.get("function"))
            name = getattr(function, "name", None) or function.get("name")
            arguments = getattr(function, "arguments", None) or function.get("arguments")
            try:
                parsed_arguments = json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                parsed_arguments = arguments
            tool_use_blocks.append(
                ToolUseBlockParam(
                    id=getattr(tool_call, "id", None) or tool_call.get("id"),
                    name=name,
                    input=parsed_arguments,
                )
            )
        return tool_use_blocks
