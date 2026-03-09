# this python script feeds gemini-cli with queries passed on the CLI
# it is part of the iterative design pattern currently in development

# this agentic script is designed for use in thereminq-tensor containers
# DO NOT RUN AGENTIC CODE ON YOUR LOCAL MACHINE WITH ALL RIGHTS, NOT EVER!!

# main.py

import os
import sys
import google.generativeai as genai
from typing import Optional
from dataclasses import dataclass

@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-pro"
    max_retries: int = 3
    timeout: int = 30

import time

def query_gemini(
    query: str,
    config: Optional[GeminiConfig] = None,
    api_key: Optional[str] = None
) -> str:
    """Query Gemini API with proper error handling."""
    if config is None:
        config = GeminiConfig()

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.model)

    retries = 0
    while retries <= config.max_retries:
        try:
            response = model.generate_content(query)
            return response.text
        except Exception as e:
            retries += 1
            if retries > config.max_retries:
                raise RuntimeError(f"Failed to query Gemini API after {config.max_retries} retries. Error: {e}")
            print(f"Error occurred: {e}. Retrying {retries}/{config.max_retries}...", file=sys.stderr)
            time.sleep(2 ** retries) # Exponential backoff

    return ""

def main():
    """
    This script takes a query from the command line, sends it to the Gemini API,
    and prints the generated content.
    """
    try:
        # --- Configuration ---
        # For security, it's best to set your API key as an environment variable.
        # You can get an API key from Google AI Studio: https://aistudio.google.com/
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
            print("Please set the environment variable with your API key.", file=sys.stderr)
            # Example for bash/zsh: export GEMINI_API_KEY='YOUR_API_KEY'
            # Example for PowerShell: $env:GEMINI_API_KEY='YOUR_API_KEY'
            sys.exit(1)

        genai.configure(api_key=api_key)

        # --- Command-Line Argument Handling ---
        # Check if a query was provided as a command-line argument.
        # sys.argv[0] is the script name, so we need at least 2 arguments.
        if len(sys.argv) < 2:
            print("Usage: python main.py \"<your query here>\"", file=sys.stderr)
            sys.exit(1)

        # Join all arguments after the script name to form the query.
        user_query = " ".join(sys.argv[1:])
        
        # The user's query is the prompt.
        prompt = user_query

        config = GeminiConfig()
        response_text = query_gemini(prompt, config=config, api_key=api_key)

        # --- Output ---
        # Print only the generated text to the console
        print(response_text)

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
