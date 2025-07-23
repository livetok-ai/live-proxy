# Live-Proxy Project Information

## Project Overview
Live-proxy is an opensource proxy service for interacting with large language model (LLM) WebSocket APIs exposing other interfaces that are better suited for real-time use cases over the Internet. It supports Gemini, OpenAI, and Bedrock speech-to-speech models and facilitates real-time communication using WebRTC and WebTransport.

## Development Requirements

### Python Version
- Requires Python 3.12+

### Code Formatting
**IMPORTANT: Always run `black` to format the code after making any changes.**

```bash
black .
```

The project uses Black with the following configuration:
- Line length: 120 characters
- Target version: Python 3.8+

### Additional Tools
The project also uses:
- **ruff**: For linting and code quality checks
- **black**: For code formatting
- **pytest**: For testing

Run these tools to ensure code quality:
```bash
ruff check .
mypy .
pytest
```

### Testing
Run tests with:
```bash
pytest
```

For coverage reports:
```bash
pytest --cov
```

### Dependencies
Main dependencies include:
- aiortc (WebRTC)
- aioquic (WebTransport)
- aiohttp (HTTP server)
- google-genai (Gemini integration)
- openai[realtime] (OpenAI integration)
- aws-sdk-bedrock-runtime (Bedrock integration)

### Environment Variables
Set required API keys before running:
- `GOOGLE_API_KEY` for Gemini
- `OPENAI_API_KEY` for OpenAI
- AWS credentials for Bedrock

### Running the Server
```bash
GOOGLE_API_KEY=XXX python proxy.py
```

### File Structure
- `proxy.py`: Main server entry point
- `model_*.py`: LLM provider integrations
- `connection.py`: WebRTC connection handling
- `api.py`: HTTP API endpoints
- `metrics.py`: Prometheus metrics
- `logger.py`: Logging configuration
- `tests/`: Test suite
- `demo/`: Demo client files