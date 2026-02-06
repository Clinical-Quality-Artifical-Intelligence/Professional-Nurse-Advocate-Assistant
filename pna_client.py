import os
from huggingface_hub import InferenceClient

class PNAAssistantClient:
    """PNA Assistant using Hugging Face Inference API (no local GPU required)."""
    
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        # Use Hugging Face Inference API - no GPU needed locally
        self.client = InferenceClient()
        
        # Diversity Emojis from PNA instructions
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        print("PNA Assistant initialized (using Inference API)")

    def generate_response(self, prompt, context="", history=[]):
        """Generate response using Hugging Face Inference API."""
        
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

        try:
            # Build messages for chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Use Inference API for text generation
            response = self.client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Inference API error: {e}")
            # Fallback response
            return f"👩🏽‍⚕️ I apologize, but I'm experiencing technical difficulties right now. As a PNA Assistant, I'm here to support you with the A-EQUIP model and restorative supervision. Please try again in a moment, or feel free to ask me about:\n\n• The four functions of clinical supervision\n• Restorative supervision techniques\n• The role of a Professional Nurse Advocate\n• Quality improvement in nursing practice"
