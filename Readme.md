# Production-Ready OpenAI Client

A robust, production-ready OpenAI API client implementation with comprehensive testing.

## Overview

This project demonstrates how to build a robust, production-ready OpenAI API client with proper testing. The implementation includes:

- A fully-featured OpenAI client class supporting chat completions, embeddings, and vision capabilities
- Command-line interface for interacting with the API
- Comprehensive testing suite with unit tests and integration tests
- Proper error handling and logging

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Setting Up API Keys

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or use a `.env` file (recommended for development):

```
OPENAI_API_KEY=your-api-key-here
```

### Basic Usage

```python
from openai_client import OpenAIClient

# Initialize client
client = OpenAIClient()

# Use chat completion
messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "Explain hypertension in simple terms."}
]

response = client.chat_completion(
    messages=messages,
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

print(response["choices"][0]["message"]["content"])
```

### Command Line Interface

The package includes a command-line interface for easy interaction:

```bash
python main.py --message "Explain diabetes management" --model gpt-4o --temperature 0.7
```

For vision capabilities:

```bash
python main.py --message "Describe what you see in this medical image" --image path/to/image.jpg
```

## Testing

### Running Tests

Run all tests:

```bash
pytest
```

Run only unit tests:

```bash
pytest -m "not integration"
```

Run integration tests (requires valid API key):

```bash
RUN_INTEGRATION_TESTS=1 pytest -m integration
```

Generate coverage report:

```bash
pytest --cov=medagentsim
```

## Testing LLMs vs Standard Pytest Testing

Testing applications that integrate with Large Language Models (LLMs) like OpenAI's GPT models presents unique challenges compared to traditional software testing:

### Key Differences

1. **Non-deterministic Outputs**: LLMs can produce different outputs for the same input due to temperature settings and model updates, making exact output matching impractical.

2. **API Costs**: Each real API call to an LLM service incurs costs, making extensive integration testing expensive.

3. **Rate Limits**: LLM APIs often have rate limits that can restrict the number of tests that can be run in succession.

4. **Model Versioning**: LLMs evolve over time, so tests might need to account for model version changes.

### Testing Strategies Implemented

This project demonstrates several best practices for testing LLM applications:

1. **Mock-based Testing**: Most tests use mocks to simulate API responses, avoiding actual API calls during routine testing.

2. **Structural Validation**: Tests verify the structure of responses rather than exact content, checking for expected fields and data types.

3. **Selective Integration Tests**: A small subset of tests make actual API calls to verify real-world behavior, but these can be disabled to avoid costs.

4. **Error Handling Tests**: Comprehensive testing of error conditions (invalid API keys, file not found, etc.) ensures the application handles failures gracefully.

5. **Parameterized Tests**: Using pytest fixtures to parameterize tests allows testing multiple scenarios without code duplication.

6. **Environment-based Test Selection**: Integration tests can be conditionally run based on environment variables, allowing developers to opt-in to tests that make real API calls.

### Example: Testing Chat Completion

```python
# Mock-based test (fast, free, deterministic)
def test_chat_completion_with_mock(mocker):
    mock_response = {"choices": [{"message": {"content": "Test response"}}]}
    mocker.patch("openai.OpenAI.chat.completions.create", return_value=mock_response)
    client = OpenAIClient(api_key="test-key")
    result = client.chat_completion(messages=[{"role": "user", "content": "Hello"}])
    assert "choices" in result
    assert result["choices"][0]["message"]["content"] == "Test response"

# Integration test (slower, costs money, non-deterministic)
@pytest.mark.integration
def test_chat_completion_integration(openai_api_key):
    client = OpenAIClient(api_key=openai_api_key)
    result = client.chat_completion(
        messages=[{"role": "user", "content": "Say 'test successful' and nothing else."}],
        temperature=0  # Use temperature=0 for more deterministic results
    )
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]
    assert "content" in result["choices"][0]["message"]
    # Check for substring rather than exact match
    assert "test successful" in result["choices"][0]["message"]["content"].lower()
```

By combining these approaches, we achieve comprehensive test coverage while minimizing costs and handling the inherent variability of LLM outputs.

## Project Structure

```
openai-client/
├── openai_client.py     # The OpenAI client implementation
├── logger.py            # Logging utilities
├── main.py              # Command-line interface
├── requirements.txt     # Project dependencies
├── pytest.ini           # Pytest configuration
└── tests/
    ├── conftest.py      # Test fixtures and configuration
    ├── test_openai_client.py  # Unit tests
    └── test_integration_openai_client.py  # Integration tests
```

## Best Practices Implemented

1. **Robust Error Handling**: All API calls are properly wrapped with try/except blocks
2. **Proper Logging**: Detailed logging for debugging and monitoring
3. **Type Annotations**: Full type hints for better IDE support and code quality
4. **Comprehensive Testing**: Unit tests with mocks and integration tests for end-to-end validation
5. **Configurable Parameters**: All client methods support customization of model parameters
6. **Environment Variable Support**: API keys can be passed directly or via environment variables
7. **Clear Documentation**: Docstrings and README provide complete usage information

## Architecture and Flow Diagrams

### Application Flow Diagram

The following diagram illustrates the flow of data and control through the OpenAI client system, including the testing framework:

```mermaid
flowchart TD
    subgraph User["User / Application"]
        A[User Input] --> B[Command Line Arguments]
        A --> C[Direct API Call]
    end

    subgraph AppModule["app.py"]
        B --> D[parse_arguments]
        D --> E[main]
        E --> F[setup_logging]
        E --> G[create_default_messages]
        G --> H{Has Image?}
        H -->|Yes| I[chat_with_image]
        H -->|No| J[chat_completion]
    end

    subgraph OpenAIClient["openai_client.py"]
        C --> K[OpenAIClient]
        I --> K
        J --> K
        K --> L[__init__]
        L --> M{API Key Valid?}
        M -->|No| N[ValueError]
        M -->|Yes| O[Initialize Client]
        
        O --> P[chat_completion]
        O --> Q[create_embedding]
        O --> R[chat_with_image]
        
        P --> S[OpenAI API]
        Q --> S
        R --> T[Process Image]
        T --> S
        
        S --> U[Process Response]
        U --> V[Return Result]
    end

    subgraph Logger["logger.py"]
        F --> W[get_logger]
        K --> W
        W --> X[Configure Logger]
    end

    V --> Y[Application Result]

    subgraph Testing["Testing Framework"]
        Z[pytest] --> AA[conftest.py]
        Z --> AB[test_openai_client.py]
        Z --> AC[test_app.py]
        
        AA --> AD[openai_api_key fixture]
        AA --> AE[mock_env_variables fixture]
        
        AB --> AF[TestOpenAIClient]
        AF --> AG[test_init]
        AF --> AH[test_chat_completion]
        AF --> AI[test_create_embedding]
        AF --> AJ[test_chat_with_image]
        AF --> AK[test_chat_with_image_file_not_found]
        
        AC --> AL[TestApp]
        AL --> AM[test_parse_arguments]
        AL --> AN[test_setup_logging]
        AL --> AO[test_create_default_messages]
        AL --> AP[test_main_text_only]
        AL --> AQ[test_main_with_image]
        AL --> AR[test_main_value_error]
        AL --> AS[test_main_file_not_found]
        AL --> AT[test_main_general_exception]
    end

    AG -.-> L
    AH -.-> P
    AI -.-> Q
    AJ -.-> R
    AK -.-> R
    AM -.-> D
    AN -.-> F
    AO -.-> G
    AP -.-> E
    AQ -.-> E
    AR -.-> E
    AS -.-> E
    AT -.-> E
```

### Component Relationship Diagram

This diagram shows the relationships between the main components of the system:

```mermaid
classDiagram
    class OpenAIClient {
        +__init__(api_key)
        +chat_completion(messages, model, temperature, max_tokens)
        +create_embedding(text, model)
        +chat_with_image(messages, image_path, model, temperature, max_tokens)
    }
    
    class AppModule {
        +parse_arguments()
        +setup_logging(verbose)
        +create_default_messages(user_message)
        +main()
    }
    
    class Logger {
        +get_logger(name)
    }
    
    class TestOpenAIClient {
        +test_init()
        +test_chat_completion()
        +test_create_embedding()
        +test_chat_with_image()
        +test_chat_with_image_file_not_found()
    }
    
    class TestApp {
        +test_parse_arguments()
        +test_setup_logging()
        +test_create_default_messages()
        +test_main_text_only()
        +test_main_with_image()
        +test_main_value_error()
        +test_main_file_not_found()
        +test_main_general_exception()
    }
    
    AppModule --> OpenAIClient : uses
    OpenAIClient --> Logger : uses
    AppModule --> Logger : uses
    TestOpenAIClient --> OpenAIClient : tests
    TestApp --> AppModule : tests
```

## License

MIT