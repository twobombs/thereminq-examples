# Agentics

This directory contains agentic scripts designed for use in ThereminQ-Tensor containers.

**WARNING: Do not run agentic code on your local machine with full permissions.**

## gemini-query.py

This Python script sends a query to the Gemini 1.5 Pro model using the `gemini-cli`.

### Usage

To use the script, you first need to set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY='your_api_key'
```

Then, you can run the script from the command line, passing your query as an argument:

```bash
python gemini-query.py "your query here"
```

The script will then print the model's response to the console.
