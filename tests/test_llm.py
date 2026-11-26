"""Smoke test for ForgeAI OllamaClient against local Ollama instance."""

import os
import sys
import time

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forge.llm import OllamaClient


def test_plain_text_generation(client: OllamaClient) -> None:
    """Test actual plain text generation and measure response time."""
    print("=== Testing Plain Text Generation ===")
    prompt = "In exactly one sentence, describe the primary function of an industrial blast furnace."
    system_prompt = "You are an industrial engineering expert. Keep answers concise and direct."

    start_time = time.perf_counter()
    response = client.generate(prompt=prompt, system_prompt=system_prompt)
    elapsed = time.perf_counter() - start_time

    assert isinstance(response, str), f"Expected response to be str, got {type(response)}"
    assert len(response.strip()) > 0, "Response string must not be empty"

    print(f"Prompt: {prompt}")
    print(f"Response: {response.strip()}")
    print(f"Elapsed Time: {elapsed:.3f}s\n")


def test_structured_json_generation(client: OllamaClient) -> None:
    """Test actual structured JSON generation and validate expected fields."""
    print("=== Testing Structured JSON Generation ===")
    prompt = (
        "Return a JSON object for an industrial sensor with the following fields: "
        "'sensor_id' (string), 'temperature_celsius' (number), and 'status' (string: 'nominal' or 'critical')."
    )
    system_prompt = "You are a factory monitoring system assistant. Always output valid JSON."

    start_time = time.perf_counter()
    response = client.generate_json(prompt=prompt, system_prompt=system_prompt)
    elapsed = time.perf_counter() - start_time

    assert isinstance(response, dict), f"Expected response to be dict, got {type(response)}"
    for field in ("sensor_id", "temperature_celsius", "status"):
        assert field in response, f"Missing required field '{field}' in JSON response: {response}"

    print(f"Prompt: {prompt}")
    print(f"Parsed JSON: {response}")
    print(f"Elapsed Time: {elapsed:.3f}s\n")


def main() -> int:
    client = OllamaClient()
    print(f"Initializing smoke tests using OllamaClient (model='{client.model}', endpoint='{client.base_url}')\n")

    try:
        test_plain_text_generation(client)
        test_structured_json_generation(client)
        print("Smoke tests passed successfully.")
        return 0
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
