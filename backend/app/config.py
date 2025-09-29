from pydantic import PostgresDsn, validator
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database settings
    DATABASE_URL: str = "sqlite:///./app.db"
    TEST_DATABASE_URL: str = "sqlite:///./test.db"
    
    # For production PostgreSQL
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[str] = None
    
    # Construct the DATABASE_URL from environment variables if not provided
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
            
        # If DATABASE_URL is not set, try to construct it from PostgreSQL env vars
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        host = values.get("POSTGRES_HOST")
        port = values.get("POSTGRES_PORT", "5432")
        db = values.get("POSTGRES_DB")
        
        if all([user, password, host, db]):
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
            
        # Fall back to SQLite if PostgreSQL config is incomplete
        return "sqlite:///./app.db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
