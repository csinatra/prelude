import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

def test_api_connection():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": "Reply with the single word: connected"
            }
        ],
    )
    
    result = response.content[0].text
    print(f"Response: {result}")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    assert "connected" in result.lower()
    print("✓ API connection confirmed")

if __name__ == "__main__":
    test_api_connection()