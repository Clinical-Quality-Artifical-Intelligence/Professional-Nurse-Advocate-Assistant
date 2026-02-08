"""
PNA Assistant Client - Using HF Inference API
No GPU required - runs on CPU Basic
"""
import os
from huggingface_hub import InferenceClient

# Model to use via Inference API
MODEL_ID = "google/gemma-2-2b-it"


class PNAAssistantClient:
    """PNA Assistant Client - calls Gemma 2 via HF Inference API."""
    
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        # Diversity Emojis from PNA instructions
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        
        # Initialize Inference Client (uses HF_TOKEN from env if available)
        self.client = InferenceClient(
            model=model_id,
            token=os.getenv("HF_TOKEN")
        )
        print(f"PNA Assistant initialized (Inference API mode: {model_id})")

    def generate_response(self, prompt, context="", history=[]):
        """Generate response using HF Inference API."""
        
        system_prompt = f"""You are a Professional Nurse Advocate (PNA) AI tutor. Your role is to guide nursing professionals through the A-EQUIP model (Advocating and Educating for Quality Improvement).

**Your Core Functions (A-EQUIP):**
1. Normative: Monitoring, evaluation, quality control
2. Formative: Education and development
3. Restorative: Clinical supervision (your primary focus)
4. Personal Action: Quality improvement

**Communication Style:**
- Use person-centred, compassionate language
- Always include a diversity emoji: {', '.join(self.diversity_emojis)}
- Ask open-ended questions before giving answers
- Focus on reflection and restorative supervision
- Keep responses to 2 short paragraphs or 6 bullet points max

**Scope:**
- Only discuss PNA, A-EQUIP, nursing fields
- For out-of-scope topics: "I can only assist with topics related to the Professional Nurse Advocate role and the A-EQUIP model."

**Reference Material:**
{context}
"""
        
        # Build the full prompt for Gemma chat format
        full_prompt = f"<start_of_turn>user\n{system_prompt}\n\nUser question: {prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        try:
            response = self.client.text_generation(
                full_prompt,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                stop_sequences=["<end_of_turn>"]
            )
            return response.strip()
        except Exception as e:
            print(f"Inference API error: {e}")
            return f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment. 👩🏽‍⚕️"
