"""
PNA Assistant Client - Using HF Inference API
No GPU required - runs on CPU Basic
"""
import os
from huggingface_hub import InferenceClient

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"


class PNAAssistantClient:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        self.client = InferenceClient(model=model_id, token=os.getenv("HF_TOKEN"))
        print(f"PNA Assistant initialized (Inference API mode: {model_id})")

    def generate_response(self, prompt, context="", history=[]):
        emoji_list = ", ".join(self.diversity_emojis)
        system_content = f"You are a Professional Nurse Advocate (PNA) AI tutor for the A-EQUIP model. Include emoji: {emoji_list}. Keep responses to 2 paragraphs max. Context: {context}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return f"I apologize, but I am experiencing technical difficulties. (Error: {str(e)})"
