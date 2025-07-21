odoo.define('ai_automation_base.llm_service', function (require) {
    'use strict';

    var ajax = require('web.ajax');
    var core = require('web.core');
    var _t = core._t;

    var LLMService = {
        /**
         * Generate text using the configured LLM provider
         * @param {string} prompt - The input prompt
         * @param {string} provider - Provider code (optional, defaults to 'openai')
         * @param {string} model - Model name (optional)
         * @param {Object} options - Additional options
         * @returns {Promise} Promise that resolves with the generated text
         */
        generateText: function (prompt, provider, model, options) {
            return ajax.jsonRpc('/ai_automation/llm/generate', 'call', {
                prompt: prompt,
                provider: provider || 'openai',
                model: model,
                ...options
            });
        },

        /**
         * Generate chat completion using the configured LLM provider
         * @param {Array} messages - Array of message objects with role and content
         * @param {string} provider - Provider code (optional, defaults to 'openai')
         * @param {string} model - Model name (optional)
         * @param {Object} options - Additional options
         * @returns {Promise} Promise that resolves with the chat completion
         */
        chatCompletion: function (messages, provider, model, options) {
            return ajax.jsonRpc('/ai_automation/llm/chat', 'call', {
                messages: messages,
                provider: provider || 'openai',
                model: model,
                ...options
            });
        },

        /**
         * Get embeddings for the given text
         * @param {string} text - Input text to embed
         * @param {string} provider - Provider code (optional, defaults to 'openai')
         * @param {string} model - Model name (optional)
         * @returns {Promise} Promise that resolves with the embeddings
         */
        getEmbeddings: function (text, provider, model) {
            return ajax.jsonRpc('/ai_automation/llm/embeddings', 'call', {
                text: text,
                provider: provider || 'openai',
                model: model
            });
        },

        /**
         * Stream response from the LLM provider
         * @param {string} prompt - The input prompt
         * @param {string} provider - Provider code (optional, defaults to 'openai')
         * @param {string} model - Model name (optional)
         * @param {Object} options - Additional options
         * @param {Function} onChunk - Callback function for each chunk
         * @param {Function} onComplete - Callback function when streaming is complete
         * @param {Function} onError - Callback function for errors
         */
        streamResponse: function (prompt, provider, model, options, onChunk, onComplete, onError) {
            var data = {
                prompt: prompt,
                provider: provider || 'openai',
                model: model,
                ...options
            };

            fetch('/ai_automation/llm/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                },
                body: JSON.stringify(data),
                credentials: 'same-origin'
            })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                var reader = response.body.getReader();
                var decoder = new TextDecoder();

                function readStream() {
                    return reader.read().then(function (result) {
                        if (result.done) {
                            if (onComplete) onComplete();
                            return;
                        }

                        var chunk = decoder.decode(result.value, {stream: true});
                        var lines = chunk.split('\n');

                        lines.forEach(function (line) {
                            if (line.startsWith('data: ')) {
                                var data = line.slice(6);
                                if (data === '[DONE]') {
                                    if (onComplete) onComplete();
                                    return;
                                }

                                try {
                                    var parsed = JSON.parse(data);
                                    if (parsed.chunk && onChunk) {
                                        onChunk(parsed.chunk);
                                    } else if (parsed.error && onError) {
                                        onError(parsed.error);
                                    }
                                } catch (e) {
                                    // Ignore parsing errors for incomplete chunks
                                }
                            }
                        });

                        return readStream();
                    });
                }

                return readStream();
            })
            .catch(function (error) {
                if (onError) onError(error.message);
            });
        },

        /**
         * Get available LLM providers
         * @returns {Promise} Promise that resolves with the list of providers
         */
        getProviders: function () {
            return ajax.jsonRpc('/ai_automation/llm/providers', 'call', {});
        },

        /**
         * Get available models for a provider
         * @param {string} providerCode - Provider code (optional)
         * @returns {Promise} Promise that resolves with the list of models
         */
        getModels: function (providerCode) {
            return ajax.jsonRpc('/ai_automation/llm/models', 'call', {
                provider_code: providerCode
            });
        },

        /**
         * Test connection to a specific LLM provider
         * @param {string} provider - Provider code
         * @returns {Promise} Promise that resolves with the test result
         */
        testConnection: function (provider) {
            return ajax.jsonRpc('/ai_automation/llm/test_connection', 'call', {
                provider: provider
            });
        },

        /**
         * Get usage statistics
         * @param {Object} filters - Optional filters (provider_id, user_id, date_from, date_to)
         * @returns {Promise} Promise that resolves with usage statistics
         */
        getUsageStatistics: function (filters) {
            return ajax.jsonRpc('/ai_automation/llm/usage_stats', 'call', filters || {});
        },

        /**
         * Call a registered function/tool
         * @param {string} functionName - Name of the function to call
         * @param {Object} arguments - Arguments to pass to the function
         * @param {string} provider - Provider code (optional, defaults to 'openai')
         * @returns {Promise} Promise that resolves with the function result
         */
        callFunction: function (functionName, arguments, provider) {
            return ajax.jsonRpc('/ai_automation/llm/function_call', 'call', {
                function_name: functionName,
                arguments: arguments,
                provider: provider || 'openai'
            });
        },

        /**
         * Get available functions/tools
         * @param {string} provider - Provider code (optional, defaults to 'openai')
         * @returns {Promise} Promise that resolves with the list of functions
         */
        getAvailableFunctions: function (provider) {
            return ajax.jsonRpc('/ai_automation/llm/available_functions', 'call', {
                provider: provider || 'openai'
            });
        }
    };

    return LLMService;
}); 