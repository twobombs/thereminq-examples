# this python script feeds gemini-cli with queries passed on the CLI
# it is part of the iterative design pattern currently in development

# this agentic script is designed for use in thereminq-tensor containers
# DO NOT RUN AGENTIC CODE ON YOUR LOCAL MACHINE WITH ALL RIGHTS, NOT EVER!!

import os
import sys
import logging
from typing import Optional
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GeminiConfig:
    """Configuration for Gemini API queries."""
    model: str = "gemini-2.5-pro"
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7

def query_gemini(
    query: str,
    config: Optional[GeminiConfig] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Query Gemini API with proper error handling, retry logic, and rate limit handling.
    
    Args:
        query: The user's query string.
        config: Optional GeminiConfig for customization. Defaults to None.
        api_key: Optional API key. If not provided, reads from GEMINI_API_KEY env var.
    
    Returns:
        The generated response text.
    
    Raises:
        ValueError: If API key is not provided or set.
        RuntimeError: If all retry attempts fail.
    """
    if config is None:
        config = GeminiConfig()
    
    # Get API key
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Input validation
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Sanitize input
    query = query.strip()
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model)
    
    generation_config = GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=8192,
    )
    
    last_error = None
    for attempt in range(config.max_retries):
        try:
            logger.info(f"Sending query (attempt {attempt + 1}/{config.max_retries})")
            response = model.generate_content(
                query,
                generation_config=generation_config,
                stream=False
            )
            logger.info("Query completed successfully")
            return response.text
        
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_str:
                if attempt < config.max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Rate limit exceeded after all retries")
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < config.max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
    
    raise RuntimeError(f"Failed after {config.max_retries} attempts: {last_error}")

def main():
    """
    Main entry point for the Gemini query script.
    Takes a query from the command line, sends it to the Gemini API,
    and prints the generated content.
    """
    try:
        # --- Command-Line Argument Handling ---
        if len(sys.argv) < 2:
            print("Usage: python gemini-query.py \"<your query here>\"", file=sys.stderr)
            sys.exit(1)

        # Join all arguments after the script name to form the query.
        user_query = " ".join(sys.argv[1:])
        
        # --- Configuration ---
        # Model can be overridden via environment variable
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
        timeout = int(os.environ.get("GEMINI_TIMEOUT", "30"))
        temperature = float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
        
        config = GeminiConfig(
            model=model_name,
            max_retries=max_retries,
            timeout=timeout,
            temperature=temperature
        )
        
        # --- AI Model Interaction ---
        response = query_gemini(user_query, config)

        # --- Output ---
        print(response)

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
