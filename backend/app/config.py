from pydantic import PostgresDsn, validator, AnyHttpUrl
from pydantic_settings import BaseSettings
from typing import Optional, Union
import os

class Settings(BaseSettings):
    # Database settings - use environment variable if available, otherwise default to SQLite
    DATABASE_URL: str = "sqlite:///./app.db"
    TEST_DATABASE_URL: str = "sqlite:///./test.db"
    
    # For production PostgreSQL
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: Optional[str] = "5432"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    # Construct the DATABASE_URL from environment variables if not provided
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str) and v:
            # If DATABASE_URL is explicitly set, use it
            return v
            
        # Check for Render's PostgreSQL URL first
        if "DATABASE_URL" in os.environ:
            return os.environ["DATABASE_URL"]
            
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
