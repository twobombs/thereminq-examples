# this python script feeds gemini-cli with queries passed on the CLI
# it is part of the iterative design pattern currently in development

# this agentic script is designed for use in thereminq-tensor containers
# DO NOT RUN AGENTIC CODE ON YOUR LOCAL MACHINE WITH ALL RIGHTS, NOT EVER!!

# main.py

import os
import sys
import time
import google.generativeai as genai
from typing import Optional
from dataclasses import dataclass

@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-pro"
    max_retries: int = 3
    timeout: int = 30
    retry_delay: int = 5

def query_gemini(
    query: str,
    config: Optional[GeminiConfig] = None,
    api_key: Optional[str] = None
) -> str:
    """Query Gemini API with proper error handling."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    config = config or GeminiConfig()
    api_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model)

    for attempt in range(config.max_retries):
        try:
            response = model.generate_content(query)
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                if attempt < config.max_retries - 1:
                    print(f"Rate limit hit. Retrying in {config.retry_delay} seconds...", file=sys.stderr)
                    time.sleep(config.retry_delay)
                    continue
            raise Exception(f"Failed to query Gemini API after {config.max_retries} attempts: {e}")

    return ""

def main() -> None:
    """
    This script takes a query from the command line, sends it to the Gemini API,
    and prints the generated content.
    """
    try:
        # --- Command-Line Argument Handling ---
        # Check if a query was provided as a command-line argument.
        # sys.argv[0] is the script name, so we need at least 2 arguments.
        if len(sys.argv) < 2:
            print("Usage: python main.py \"<your query here>\"", file=sys.stderr)
            sys.exit(1)

        # Join all arguments after the script name to form the query.
        user_query = " ".join(sys.argv[1:])
        
        # Output
        response_text = query_gemini(user_query)
        print(response_text)

    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
