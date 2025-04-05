from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import List, Dict, Any


class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Handle LLM start."""
        print(f"LLM started with parameters: {serialized}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Handle LLM end."""
        print(f"LLM ended with response: \n{response.generations[0][0].text}")
        print("***************")
