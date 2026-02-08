"""
PNA Assistant Client - Using HF Inference API
No GPU required - runs on CPU Basic
"""
import os
from huggingface_hub import InferenceClient

# Model to use via Inference API
# Note: Using Phi-3 because Gemma 2 isn't available on HF serverless Inference API
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


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
        """Generate response using HF Inference API chat_completion method."""
        
        system_content = f"""You are a Professional Nurse Advocate (PNA) AI tutor. Your role is to guide nursing professionals through the A-EQUIP model (Advocating and Educating for Quality Improvement).

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
        
        # Build messages for chat_completion (proper chat API format)
        messages = [
            {"role": "user", "content": f"{system_content}\n\nUser question: {prompt}"}
        ]
        
        try:
            # DEBUG: Print token status (don't print the actual token)
            token = os.getenv("HF_TOKEN")
            print(f"DEBUG: HF_TOKEN is {'Set' if token else 'NOT SET'}")
            if token:
                print(f"DEBUG: Token length: {len(token)}")
            
            # Use chat_completion for chat models like Gemma 2
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            print(f"DEBUG: Response type: {type(response)}")
            
            # Extract the message content from the response
            answer = response.choices[0].message.content
            print(f"DEBUG: Answer: {answer[:100]}...")
            return answer.strip()
            
        except Exception as e:
            print(f"DEBUG: Exception Type: {type(e)}")
            print(f"DEBUG: Exception Args: {e.args}")
            print(f"DEBUG: Full Exception: {repr(e)}")
            return f"I apologize, but I'm experiencing technical difficulties. Please try again in a moment. (Error: {str(e)})"
