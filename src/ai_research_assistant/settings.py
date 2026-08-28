from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str
    database_url: str
    database_test_url: str

    model_config = SettingsConfigDict(env_file=".env")