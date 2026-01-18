import gradio as gr
from knowledge_base import PNAKnowledgeBase
from pna_client import PNAAssistantClient
import os

# --- Constants ---
GUIDE_FILENAME = "Professional nurse advocate A-EQUIP model Guide.md"

# --- Initialize Components ---
kb = PNAKnowledgeBase(GUIDE_FILENAME)
client = PNAAssistantClient()

def chat_response(message, history):
    """Generate response using RAG + fine-tuned model."""
    # Search Knowledge Base for relevant context
    context = kb.search(message)
    
    # Generate Response using fine-tuned MedGemma
    response = client.generate_response(message, context=context, history=history)
    
    return response

# --- Premium CSS ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700&display=swap');

/* Root Variables */
:root {
    --primary: #4f46e5;
    --primary-light: #818cf8;
    --secondary: #0d9488;
    --bg-main: #f8fafc;
    --bg-card: rgba(255, 255, 255, 0.85);
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: rgba(148, 163, 184, 0.2);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
}

/* Global Styles */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-main) !important;
}

/* Header */
.header-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    border-radius: 0 0 24px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    box-shadow: var(--shadow-lg);
}

.header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.95rem;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-weight: 400;
}

.emoji-bar {
    font-size: 1.5rem;
    margin-top: 0.75rem;
    letter-spacing: 0.25rem;
}

/* Chat Container */
.chat-container {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px);
    border-radius: 16px !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: var(--shadow-md) !important;
    padding: 1rem !important;
}

/* Chatbot Messages */
.message-wrap {
    padding: 0.75rem 1rem !important;
}

.user-message, .bot-message {
    border-radius: 16px !important;
    padding: 0.875rem 1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

.user-message {
    background: var(--primary) !important;
    color: white !important;
    border-bottom-right-radius: 4px !important;
}

.bot-message {
    background: white !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-bottom-left-radius: 4px !important;
}

/* Input Area */
.input-container textarea {
    border-radius: 24px !important;
    border: 2px solid var(--border-color) !important;
    padding: 0.875rem 1.25rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.input-container textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    outline: none !important;
}

/* Submit Button */
.submit-btn {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    color: white !important;
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.submit-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Examples */
.examples-container {
    margin-top: 1rem !important;
}

.examples-container button {
    background: white !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    transition: all 0.2s ease !important;
}

.examples-container button:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
    background: rgba(79, 70, 229, 0.05) !important;
}

/* Footer */
footer {
    display: none !important;
}

/* Disclaimer */
.disclaimer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 1rem;
    padding: 0.75rem;
    background: rgba(148, 163, 184, 0.1);
    border-radius: 8px;
}
"""

# --- UI Layout ---
with gr.Blocks(css=CUSTOM_CSS, title="PNA Assistant", theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">Professional Nurse Advocate Assistant</h1>
            <p class="header-subtitle">Guiding you through the A-EQUIP model and Restorative Supervision</p>
            <div class="emoji-bar">👨🏾‍⚕️ 👩🏽‍⚕️ 👨🏿‍⚕️ 👩🏻‍⚕️ 👩‍⚕️</div>
        </div>
    """)
    
    # Chat Interface
    with gr.Column(elem_classes="chat-container"):
        chatbot = gr.Chatbot(
            label="",
            height=400,
            show_label=False,
            avatar_images=(None, "https://em-content.zobj.net/source/apple/391/woman-health-worker-medium-skin-tone_1f469-1f3fd-200d-2695-fe0f.png"),
            bubble_full_width=False,
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask about the A-EQUIP model, Restorative Supervision, or the PNA role...",
                show_label=False,
                container=False,
                elem_classes="input-container",
                scale=9
            )
            submit = gr.Button("Send", elem_classes="submit-btn", scale=1)
        
        # Examples
        gr.Examples(
            examples=[
                "What is the A-EQUIP model?",
                "Explain the four functions of clinical supervision",
                "How can I support a colleague through restorative supervision?",
                "What does a Professional Nurse Advocate do?"
            ],
            inputs=msg,
            elem_classes="examples-container"
        )
    
    # Disclaimer
    gr.HTML("""
        <div class="disclaimer">
            ⚠️ This tool is for educational purposes only. It does not provide clinical advice.
        </div>
    """)
    
    # Event Handlers
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        
        bot_response = chat_response(message, chat_history)
        chat_history.append((message, bot_response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
