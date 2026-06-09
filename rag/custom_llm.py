import os
import httpx
from langchain_openai import ChatOpenAI

# No proxy needed on personal laptop
http_client = httpx.Client(
    verify=True,
    timeout=120.0
)

class ChatOpenAINoStop(ChatOpenAI):
    def _get_request_payload(self, input_messages, stop=None, **kwargs):
        payload = super()._get_request_payload(
            input_messages, stop=None, **kwargs
        )
        payload.pop("stop", None)
        return payload