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

# --- Premium CSS ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700&display=swap');

.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
    font-family: 'Inter', sans-serif !important;
}

.contain { background: linear-gradient(135deg, #4f46e5 0%, #0d9488 100%) !important; }

footer { display: none !important; }
"""

HEADER_HTML = """
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4f46e5 0%, #0d9488 100%); border-radius: 12px; margin-bottom: 1rem;">
    <h1 style="color: white; font-family: 'Outfit', sans-serif; margin: 0; font-size: 1.5rem;">
        Professional Nurse Advocate Assistant
    </h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Guiding you through the A-EQUIP model and Restorative Supervision
    </p>
    <div style="font-size: 1.25rem; margin-top: 0.5rem; letter-spacing: 0.2rem;">
        👨🏾‍⚕️ 👩🏽‍⚕️ 👨🏿‍⚕️ 👩🏻‍⚕️ 👩‍⚕️
    </div>
</div>
"""

DISCLAIMER_HTML = """
<div style="text-align: center; font-size: 0.75rem; color: #64748b; margin-top: 1rem; padding: 0.75rem; background: rgba(148, 163, 184, 0.1); border-radius: 8px;">
    ⚠️ This tool is for educational purposes only. It does not provide clinical advice.
</div>
"""

# --- Simple ChatInterface ---
with gr.Blocks(css=CUSTOM_CSS, title="PNA Assistant") as demo:
    gr.HTML(HEADER_HTML)
    
    gr.ChatInterface(
        fn=chat_response,
        type="tuples",
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
