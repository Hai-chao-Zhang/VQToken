from unittest.mock import Mock, patch

from llava.conversation import Conversation, SeparatorStyle, conv_llava_llama_3


def test_llama3_global_template_does_not_eagerly_load_tokenizer():
    assert conv_llava_llama_3.tokenizer is None


def test_llama3_tokenizer_is_loaded_only_when_prompt_is_rendered():
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "rendered prompt"
    conversation = Conversation(
        system="system",
        roles=("user", "assistant"),
        messages=[["user", "hello"], ["assistant", None]],
        offset=0,
        sep_style=SeparatorStyle.LLAMA_3,
        tokenizer_id="example/llama3-tokenizer",
    )

    with patch("llava.conversation.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
        assert conversation.get_prompt() == "rendered prompt"
        assert conversation.get_prompt() == "rendered prompt"

    load.assert_called_once_with("example/llama3-tokenizer")
    assert tokenizer.apply_chat_template.call_count == 2
