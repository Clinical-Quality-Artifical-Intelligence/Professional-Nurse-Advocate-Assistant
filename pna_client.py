# CRITICAL: spaces MUST be imported FIRST before torch/CUDA
import spaces
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PNAAssistantClient:
    """PNA Assistant Client - loads model locally on ZeroGPU."""
    
    def __init__(self, model_id="google/gemma-2-2b-it"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = None  # Will be set inside @spaces.GPU function
        
        # Diversity Emojis
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        print("PNA Assistant initialized (ZeroGPU mode)")

    def _load_model(self):
        """Load model - called from inside @spaces.GPU decorated function."""
        if self.model is None:
            print("Loading model...")
            # Detect device INSIDE the GPU function where CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            print("Model loaded successfully!")

    @spaces.GPU(duration=60)
    def generate_response(self, prompt, context="", history=[]):
        """Generate response using local model on GPU."""
        
        # Load model if not loaded (happens inside GPU context)
        self._load_model()
        
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

        full_prompt = f"<start_of_turn>user\n{system_prompt}\n\nUser question: {prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the model's response
            if "<start_of_turn>model" in response:
                response = response.split("<start_of_turn>model")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"👩🏽‍⚕️ I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
