import spaces  # MUST be first before torch
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global model storage (loaded once on first GPU call)
_model = None
_tokenizer = None
_model_id = "google/gemma-2-2b-it"

@spaces.GPU(duration=60)
def generate_with_gpu(prompt, context, diversity_emojis):
    """Top-level GPU function for ZeroGPU detection."""
    global _model, _tokenizer
    
    # Load model on first call (inside GPU context)
    if _model is None:
        print(f"Loading model {_model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(_model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        print(f"Model loaded successfully on {device}!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    system_prompt = f"""You are a Professional Nurse Advocate (PNA) AI tutor. Your role is to guide nursing professionals through the A-EQUIP model (Advocating and Educating for Quality Improvement).

**Your Core Functions (A-EQUIP):**
1. Normative: Monitoring, evaluation, quality control
2. Formative: Education and development
3. Restorative: Clinical supervision (your primary focus)
4. Personal Action: Quality improvement

**Communication Style:**
- Use person-centred, compassionate language
- Always include a diversity emoji: {', '.join(diversity_emojis)}
- Ask open-ended questions before giving answers
- Focus on reflection and restorative supervision
- Keep responses to 2 short paragraphs or 6 bullet points max

**Scope:**
- Only discuss PNA, A-EQUIP, nursing fields
- For out-of-scope topics: "I can only assist with topics related to the Professional Nurse Advocate role and the A-EQUIP model."

**Reference Material:**
{context}
"""

    messages = [
        {"role": "user", "content": f"{system_prompt}\n\nUser question: {prompt}"}
    ]
    
    inputs = _tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    attention_mask = torch.ones_like(inputs).to(device)
    
    with torch.no_grad():
        outputs = _model.generate(
            inputs, 
            attention_mask=attention_mask,
            max_new_tokens=300, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id
        )
        
    response = _tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response.strip()


class PNAAssistantClient:
    """PNA Assistant Client - wrapper for the GPU function."""
    
    def __init__(self, model_id="google/gemma-2-2b-it"):
        global _model_id
        _model_id = model_id
        # Diversity Emojis from PNA instructions
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        print("PNA Assistant initialized (ZeroGPU mode)")

    def generate_response(self, prompt, context="", history=[]):
        """Generate response - calls the top-level GPU function."""
        return generate_with_gpu(prompt, context, self.diversity_emojis)
