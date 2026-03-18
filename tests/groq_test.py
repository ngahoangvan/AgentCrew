import os
import pytest
from dotenv import load_dotenv
from AgentCrew.modules.groq.service import GroqService

# Load environment variables
load_dotenv()


@pytest.fixture
def groq_service():
    """Create a GroqService instance for testing."""
    # Check if API key exists
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not found in environment variables")
    return GroqService()


def test_initialization():
    """Test that GroqService initializes correctly with API key."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not found in environment variables")

    service = GroqService()
    assert service.client is not None
    assert service.model == "llama-3.2-90b-vision-preview"
    assert isinstance(service.tools, list)
    assert isinstance(service.tool_handlers, dict)


def test_calculate_cost():
    """Test cost calculation functionality."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not found in environment variables")

    service = GroqService()
    cost = service.calculate_cost(1_000_000, 1_000_000)
    assert cost > 0
    assert isinstance(cost, float)


def test_summarize_content(groq_service):
    """Test summarization of content."""
    test_content = "This is a test content that needs to be summarized. It contains information about testing the GroqService class."
    summary = groq_service.summarize_content(test_content)

    assert summary is not None
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_explain_content(groq_service):
    """Test explanation of content."""
    test_content = "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
    explanation = groq_service.explain_content(test_content)

    assert explanation is not None
    assert isinstance(explanation, str)
    assert len(explanation) > 0


@pytest.mark.asyncio
async def test_register_and_execute_tool(groq_service):
    """Test tool registration and execution."""
    test_tool = {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input string"}
                },
                "required": ["input"],
            },
        },
    }

    async def test_handler(input):
        return {"result": f"Processed: {input}"}

    groq_service.register_tool(test_tool, test_handler)

    assert len(groq_service.tools) == 1
    assert "test_function" in groq_service.tool_handlers

    result = await groq_service.execute_tool("test_function", {"input": "test"})
    assert result == {"result": "Processed: test"}


def test_process_file_for_message(groq_service, tmp_path):
    """Test processing a file for message content."""
    # Create a temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test file content")

    # Process the file
    result = groq_service.process_file_for_message(str(test_file))

    assert result is not None
    assert result["type"] == "text"
    assert "Content of" in result["text"]
    assert "This is test file content" in result["text"]


def test_handle_file_command(groq_service, tmp_path):
    """Test handling a file command."""
    # Create a temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test file content")

    # Handle the file command
    result = groq_service.handle_file_command(str(test_file))

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "Content of" in result[0]["content"]
    assert "This is test file content" in result[0]["content"]


def test_stream_assistant_response(groq_service):
    """Test streaming assistant response."""
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    # Get the stream response
    stream = groq_service.stream_assistant_response(messages)

    # Check if stream is created
    assert stream is not None

    # Consume a small part of the stream to verify it works
    # (not consuming the whole stream to save on API costs)
    chunk_received = False
    for chunk in stream:
        if chunk:
            chunk_received = True
            break

    assert chunk_received
