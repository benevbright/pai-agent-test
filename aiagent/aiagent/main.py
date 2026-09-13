import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    openai_base_url: str
    openai_model_name: str


settings = Settings()

provider = OpenAIProvider(base_url=settings.openai_base_url, api_key='dummy')

model = OpenAIResponsesModel(
    settings.openai_model_name,
    provider=provider,
    settings=OpenAIResponsesModelSettings(
        openai_previous_response_id='auto',
        openai_store=True,  # ensure the provider keeps each response for chaining
    ),
)

agent = Agent(model, instructions='You are a helpful assistant.')


async def main() -> None:
    print('Chat with local LLM (Ctrl-D or "exit" to quit)\n')
    async with agent:
        history: list[ModelMessage] = []
        while True:
            try:
                user_input = input('you> ').strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in {'exit', 'quit'}:
                break
            result = await agent.run(user_input, message_history=history)
            # Extend by the messages produced this turn. The Responses API
            # holds the conversation server-side, so only the previous_response_id
            # chain is sent on the wire; the full history is kept in memory but
            # the client never re-transmits the earlier messages.
            history.extend(result.all_messages()[len(history):])
            print(f'\nassistant> {result.output}\n')


if __name__ == '__main__':
    asyncio.run(main())
