import gradio as gr
from knowledge_base import PNAKnowledgeBase
from pna_client import PNAAssistantClient

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

# --- Premium CSS - LIGHT MODE with LARGE FONTS ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700&display=swap');

/* Force LIGHT MODE - Override dark mode completely */
.gradio-container, .dark, .contain, body, html {
    background: #ffffff !important;
    color: #1f2937 !important;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 18px !important;
}

/* Chat area - White background, large text */
.chatbot, .chat-message, .message, .bot, .user, 
[data-testid="chatbot"], [data-testid="bot"], [data-testid="user"] {
    background: #ffffff !important;
    color: #1f2937 !important;
    font-size: 18px !important;
    line-height: 1.7 !important;
}

/* Message bubbles */
.message.bot, .message.user, .bot .message-wrap, .user .message-wrap {
    font-size: 18px !important;
    padding: 16px 20px !important;
    border-radius: 12px !important;
}

/* Bot messages - light blue background */
.message.bot, .bot .message-wrap, [data-testid="bot"] {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
}

/* User messages - light gray background */
.message.user, .user .message-wrap, [data-testid="user"] {
    background: #f3f4f6 !important;
    border: 1px solid #d1d5db !important;
}

/* Input box - larger */
textarea, input[type="text"], .textbox textarea {
    font-size: 18px !important;
    padding: 14px 18px !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 2px solid #e5e7eb !important;
}

/* Buttons - larger */
button {
    font-size: 16px !important;
    padding: 12px 24px !important;
}

/* Examples - larger text */
.examples, .example, .example-btn {
    font-size: 16px !important;
    background: #f8fafc !important;
    color: #374151 !important;
    padding: 12px 18px !important;
    border: 1px solid #e2e8f0 !important;
}

/* Labels and titles */
label, .label-wrap, h1, h2, h3, p {
    color: #1f2937 !important;
}

/* Scrollbar for visibility */
::-webkit-scrollbar {
    width: 12px;
}
::-webkit-scrollbar-thumb {
    background: #94a3b8;
    border-radius: 6px;
}

/* Header gradient kept but improved */
.contain { background: linear-gradient(135deg, #4f46e5 0%, #0d9488 100%) !important; }

footer { display: none !important; }
"""

HEADER_HTML = """
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #4f46e5 0%, #0d9488 100%); border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: white; font-family: 'Outfit', sans-serif; margin: 0; font-size: 2rem; font-weight: 700;">
        Professional Nurse Advocate Assistant
    </h1>
    <p style="color: rgba(255,255,255,0.95); margin: 0.75rem 0 0 0; font-size: 1.2rem;">
        Guiding you through the A-EQUIP model and Restorative Supervision
    </p>
    <div style="font-size: 1.5rem; margin-top: 0.75rem; letter-spacing: 0.2rem;">
        👨🏾‍⚕️ 👩🏽‍⚕️ 👨🏿‍⚕️ 👩🏻‍⚕️ 👩‍⚕️
    </div>
</div>
"""

DISCLAIMER_HTML = """
<div style="text-align: center; font-size: 1rem; color: #475569; margin-top: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
    ⚠️ This tool is for educational purposes only. It does not provide clinical advice.
</div>
"""

# --- Simple ChatInterface with LIGHT THEME ---
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(), title="PNA Assistant") as demo:
    gr.HTML(HEADER_HTML)
    
    gr.ChatInterface(
        fn=chat_response,
        type="messages",
        examples=[
            "What is the A-EQUIP model?",
            "Explain the four functions of clinical supervision",
            "How can I support a colleague through restorative supervision?",
            "What does a Professional Nurse Advocate do?"
        ],
        title="",
    )
    
    gr.HTML(DISCLAIMER_HTML)

if __name__ == "__main__":
    demo.launch()
