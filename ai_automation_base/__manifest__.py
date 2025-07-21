{
    'name': 'AI Automation Base',
    'version': '1.0.0',
    'category': 'Extra Tools/AI',
    'summary': 'Base module for integrating major LLM providers into Odoo',
    'description': """
        This module provides a foundational layer for AI automation in Odoo,
        enabling seamless API access to major Large Language Model (LLM) providers
        like OpenAI, Google Gemini, and Anthropic Claude. It acts as a centralized
        hub for LLM configuration, secure API key management, and standardized
        LLM interaction, serving as the basis for building more advanced AI applications
        or introducing AI features into existing Odoo apps.
        
        Key Features:
        - Provider-agnostic LLM integration
        - Secure API key management
        - Standardized LLM interaction interface
        - Comprehensive error handling and monitoring
        - Extensible architecture for future AI capabilities
        - Support for streaming responses
        - Framework for custom tools/function calling
    """,
    'author': 'ECOSIRE (PRIVATE) LIMITED',
    'website': 'https://www.ecosire.com',
    'email': 'info@ecosire.com',
    'depends': ['base', 'web', 'keychain'],
    'data': [
        'security/ai_automation_base_security.xml',
        'security/ir.model.access.csv',
        'views/ai_automation_base_menus.xml',
        'views/llm_provider_views.xml',
        'views/llm_api_key_views.xml',
        'views/llm_provider_model_views.xml',
        'views/llm_request_log_views.xml',
        'data/default_providers.xml',
    ],
    'demo': [],
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',
    'images': [],
    'price': 0.0,
    'currency': 'USD',
    'support': 'info@ecosire.com',
    'maintainer': 'ECOSIRE (PRIVATE) LIMITED',
    'contributors': [],
    'external_dependencies': {
        'python': ['requests', 'json'],
    },
    'assets': {
        'web.assets_backend': [
            'ai_automation_base/static/src/js/llm_service.js',
            'ai_automation_base/static/src/css/llm_interface.css',
        ],
    },
} 