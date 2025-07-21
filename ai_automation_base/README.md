# AI Automation Base - Odoo Module

A comprehensive Odoo module that provides a foundational layer for AI automation, enabling seamless integration with major Large Language Model (LLM) providers including OpenAI, Google Gemini, and Anthropic Claude.

## 🚀 Features

### Core Capabilities
- **Provider-Agnostic Architecture**: Unified interface for multiple LLM providers
- **Secure API Key Management**: Encrypted storage of sensitive credentials
- **Comprehensive Monitoring**: Request logging, usage tracking, and performance metrics
- **Streaming Support**: Real-time response streaming for interactive applications
- **Function Calling Framework**: Extensible system for custom tools and automation
- **Rate Limit Management**: Intelligent handling of API quotas and backoff strategies

### Supported Providers
- **OpenAI**: GPT-4o, GPT-4o Mini, Text Embedding Ada
- **Google Gemini**: Gemini 2.5 Pro, Gemini 2.5 Flash, Text Embedding
- **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus

### Security Features
- Encrypted API key storage using cryptography library
- Granular access control with three user levels
- Secure HTTPS communication with all providers
- Input sanitization and validation

## 📋 Requirements

### System Requirements
- Odoo 16.0 or later
- Python 3.8+
- PostgreSQL database

### Python Dependencies
- `requests` - HTTP client for API calls
- `cryptography` - For API key encryption
- `json` - JSON handling (built-in)

### Optional Dependencies
- `queue_job` - For asynchronous processing (recommended for production)

## 🛠️ Installation

1. **Clone or download the module** to your Odoo addons directory:
   ```bash
   cd /path/to/odoo/addons
   git clone <repository-url> ai_automation_base
   ```

2. **Install Python dependencies**:
   ```bash
   pip install requests cryptography
   ```

3. **Update Odoo addons list**:
   - Go to Apps menu in Odoo
   - Click "Update Apps List"
   - Search for "AI Automation Base"
   - Click Install

4. **Configure security groups**:
   - Assign users to appropriate AI groups:
     - AI User: Basic usage permissions
     - AI Manager: Configuration and monitoring
     - AI Administrator: Full access including API keys

## ⚙️ Configuration

### 1. Create API Keys
1. Navigate to **AI Automation > Security > API Keys**
2. Create new API key records for each provider
3. Enter your API keys securely (they will be encrypted)
4. Test the API keys using the "Test API Key" button

### 2. Configure LLM Providers
1. Navigate to **AI Automation > Configuration > LLM Providers**
2. Create provider configurations:
   - **OpenAI**: `https://api.openai.com/v1`
   - **Google Gemini**: `https://generativelanguage.googleapis.com/v1beta`
   - **Anthropic Claude**: `https://api.anthropic.com/v1`
3. Select default models for each provider
4. Configure provider-specific settings (e.g., organization_id for OpenAI)

### 3. Set Up Models
1. Navigate to **AI Automation > Configuration > Models**
2. Review and activate the pre-configured models
3. Add custom models if needed
4. Configure pricing information for cost tracking

## 🔧 Usage

### Basic Text Generation
```python
# Get the LLM service
llm_service = self.env['llm.openai.service']

# Generate text
response = llm_service.generate_text(
    prompt="Write a professional email introduction",
    max_tokens=150,
    temperature=0.7
)
print(response)
```

### Chat Completion
```python
# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Get chat completion
llm_service = self.env['llm.openai.service']
response = llm_service.chat_completion(
    messages=messages,
    max_tokens=100
)
```

### Embeddings
```python
# Get embeddings
llm_service = self.env['llm.openai.service']
embeddings = llm_service.get_embeddings(
    text="Sample text for embedding",
    model="text-embedding-ada-002"
)
```

### Streaming Responses
```python
# Stream response
for chunk in llm_service.stream_response(
    prompt="Write a story about a robot",
    max_tokens=500
):
    print(chunk, end='', flush=True)
```

### Frontend JavaScript Usage
```javascript
// Load the LLM service
var LLMService = require('ai_automation_base.llm_service');

// Generate text
LLMService.generateText("Hello, world!", "openai", "gpt-4o", {
    max_tokens: 100,
    temperature: 0.7
}).then(function(result) {
    console.log(result.result);
}).catch(function(error) {
    console.error(error);
});
```

## 🔌 API Endpoints

The module provides REST API endpoints for external integrations:

### Text Generation
```
POST /ai_automation/llm/generate
{
    "prompt": "Your prompt here",
    "provider": "openai",
    "model": "gpt-4o",
    "max_tokens": 100,
    "temperature": 0.7
}
```

### Chat Completion
```
POST /ai_automation/llm/chat
{
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "provider": "openai",
    "model": "gpt-4o"
}
```

### Streaming
```
POST /ai_automation/llm/stream
Content-Type: text/event-stream
{
    "prompt": "Your prompt here",
    "provider": "openai",
    "model": "gpt-4o"
}
```

### Usage Statistics
```
GET /ai_automation/llm/usage_stats?provider_id=1&date_from=2024-01-01
```

## 📊 Monitoring

### Request Logs
- Navigate to **AI Automation > Monitoring > Request Logs**
- View detailed logs of all LLM API calls
- Monitor success rates, response times, and costs
- Filter by provider, user, date range, and status

### Usage Statistics
- Track total requests, tokens, and estimated costs
- Monitor provider performance and availability
- Analyze usage patterns and trends

### Performance Metrics
- Request duration tracking
- Token usage monitoring
- Cost estimation and tracking
- Rate limit monitoring

## 🔒 Security

### Access Control
- **AI User**: Can use AI features but cannot configure providers or view API keys
- **AI Manager**: Can manage providers and view logs, but cannot access API keys
- **AI Administrator**: Full access including API key management

### Data Protection
- API keys are encrypted using the cryptography library
- Encryption keys are stored securely in Odoo configuration
- All API communications use HTTPS
- Input validation and sanitization

## 🚀 Advanced Features

### Function Calling Framework
The module includes a framework for registering custom functions that can be called by LLMs:

```python
# Register a custom function
@api.model
def register_ai_function(self, name, description, parameters, function):
    """Register a function for AI calling"""
    return self.env['llm.function.registry'].create({
        'name': name,
        'description': description,
        'parameters': parameters,
        'function': function
    })
```

### Asynchronous Processing
For high-volume applications, use the queue_job module:

```python
@job
def generate_text_async(self, prompt, **kwargs):
    """Generate text asynchronously"""
    return self.env['llm.openai.service'].generate_text(prompt, **kwargs)
```

### Custom Provider Integration
To add a new LLM provider:

1. Create a new service class inheriting from `llm.base.service`
2. Implement the required abstract methods
3. Add provider configuration in the data files
4. Update the security rules and access controls

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**
- Verify API keys are correct and active
- Check provider-specific requirements (e.g., organization_id for OpenAI)
- Ensure proper permissions for the API key

**Rate Limiting**
- Monitor usage statistics for rate limit warnings
- Implement exponential backoff in your applications
- Consider using multiple providers for load balancing

**Connection Issues**
- Verify network connectivity to provider APIs
- Check firewall settings and proxy configurations
- Review Odoo server logs for detailed error messages

### Debug Mode
Enable debug logging by setting the log level to DEBUG in your Odoo configuration:

```ini
[options]
log_level = debug
```

## 📈 Performance Optimization

### Best Practices
- Use appropriate models for your use case (e.g., GPT-4o Mini for high-volume tasks)
- Implement caching for frequently requested responses
- Use streaming for interactive applications
- Monitor and optimize token usage

### Scaling Considerations
- For high-volume applications, consider external queuing systems (Celery, Redis)
- Implement request batching where possible
- Use multiple providers for redundancy and load balancing
- Monitor database performance with request logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This module is licensed under LGPL-3. See the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on the GitHub repository
- Contact: info@ecosire.com
- Documentation: [Module Documentation](https://docs.ecosire.com/ai-automation-base)

## 🔄 Version History

### v1.0.0
- Initial release
- Support for OpenAI, Google Gemini, and Anthropic Claude
- Secure API key management
- Comprehensive monitoring and logging
- Streaming support
- Function calling framework

---

**Developed by ECOSIRE (PRIVATE) LIMITED** 