from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from askui.models.shared.agent_message_param import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from askui.models.shared.tools import Tool, ToolCollection
from askui.model_providers.openai_compatible_vlm_provider import (
    OpenAICompatibleVlmProvider,
)


class DummyTool(Tool):
    def __call__(self, *args, **kwargs):
        return "done"


class TestOpenAICompatibleVlmProvider:
    def test_initializes_openai_client_with_normalized_base_url(self) -> None:
        mock_client = MagicMock()

        with patch(
            "askui.model_providers.openai_compatible_vlm_provider.OpenAI"
        ) as openai_cls:
            openai_cls.return_value = mock_client

            provider = OpenAICompatibleVlmProvider(
                model_id="llava",
                base_url="http://localhost:11434",
                api_key="ollama",
            )

        assert provider.client is mock_client
        openai_cls.assert_called_once_with(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )

    def test_create_message_maps_openai_tool_calls_to_message_param(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="I will click the button",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "click",
                                    "arguments": '{"x": 10, "y": 20}',
                                },
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
        )

        provider = OpenAICompatibleVlmProvider(
            model_id="local-model",
            base_url="http://localhost:8000/v1",
            client=client,
        )

        tool = DummyTool(
            name="click",
            description="Click on a point",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        )

        message = provider.create_message(
            messages=[
                MessageParam(
                    role="user",
                    content=[TextBlockParam(text="Open the settings page")],
                )
            ],
            tools=ToolCollection([tool]),
        )

        assert message.role == "assistant"
        assert message.stop_reason == "tool_use"
        assert message.usage is not None
        assert message.usage.input_tokens == 11
        assert message.usage.output_tokens == 4
        assert isinstance(message.content, list)
        assert message.content[0] == TextBlockParam(text="I will click the button")
        assert message.content[1] == ToolUseBlockParam(
            id="call_1",
            name="click",
            input={"x": 10, "y": 20},
        )

        create_kwargs = client.chat.completions.create.call_args.kwargs
        assert create_kwargs["model"] == "local-model"
        assert create_kwargs["tools"][0]["type"] == "function"
        assert create_kwargs["tools"][0]["function"]["name"] == "click"
        assert create_kwargs["messages"][0]["role"] == "user"
        assert create_kwargs["messages"][0]["content"][0]["type"] == "text"

    def test_create_message_serializes_images_and_tool_results(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )

        provider = OpenAICompatibleVlmProvider(
            model_id="local-model",
            base_url="http://localhost:8000/v1",
            client=client,
        )

        message = provider.create_message(
            messages=[
                MessageParam(
                    role="user",
                    content=[
                        ImageBlockParam(
                            source=Base64ImageSourceParam(
                                data="abc123",
                                media_type="image/png",
                            )
                        ),
                        TextBlockParam(text="Find the button"),
                    ],
                ),
                MessageParam(
                    role="assistant",
                    content=[
                        ToolResultBlockParam(
                            tool_use_id="call_1",
                            content="tool output",
                        )
                    ],
                ),
            ]
        )

        assert message.role == "assistant"
        assert message.stop_reason == "end_turn"

        create_kwargs = client.chat.completions.create.call_args.kwargs
        assert create_kwargs["messages"][0]["content"][0]["type"] == "image_url"
        assert create_kwargs["messages"][0]["content"][0]["image_url"]["url"] == (
            "data:image/png;base64,abc123"
        )
        assert create_kwargs["messages"][1]["role"] == "assistant"
        assert create_kwargs["messages"][2]["role"] == "tool"
        assert create_kwargs["messages"][2]["tool_call_id"] == "call_1"
        assert create_kwargs["messages"][2]["content"] == "tool output"
