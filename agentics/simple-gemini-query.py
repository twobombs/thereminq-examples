# this python script feeds gemini-cli with queries passed on the CLI
# it is part of the iterative design pattern currently in development

# this agentic script is designed for use in thereminq-tensor containers
# DO NOT RUN AGENTIC CODE ON YOUR LOCAL MACHINE WITH ALL RIGHTS, NOT EVER!!

# main.py

import os
import sys
import google.generativeai as genai

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

        # --- AI Model Interaction ---
        # Initialize the generative model
        # Using the 'gemini-2.5-pro' model as requested.
        model = genai.GenerativeModel('gemini-2.5-pro')

        # Send the prompt to the model to generate content
        response = model.generate_content(prompt)

        # --- Output ---
        # Print only the generated text to the console
        print(response.text)

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
