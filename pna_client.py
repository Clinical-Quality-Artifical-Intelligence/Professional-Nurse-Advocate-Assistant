"""
PNA Assistant Client - Using HF Inference API (Direct HTTP)
No GPU required - runs on CPU Basic
"""
import os
import requests
import json
import time

# List of models to try in order
MODELS = [
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta"
]

class PNAAssistantClient:
    def __init__(self):
        self.diversity_emojis = ["👨🏾‍⚕️", "👩🏽‍⚕️", "👨🏿‍⚕️", "👩🏻‍⚕️", "👩‍⚕️"]
        self.api_token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        print(f"PNA Assistant initialized. Candidate models: {MODELS}")

    def query_api(self, model_id, payload):
        """Send request to HF Inference API"""
        # Updated URL as per error message
        api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        response = requests.post(api_url, headers=self.headers, json=payload)
        return response.json()

    def generate_response(self, prompt, context="", history=[]):
        emoji_list = ", ".join(self.diversity_emojis)
        
        # Prepare system content
        system_content = f"You are a Professional Nurse Advocate (PNA) AI tutor. Include emoji: {emoji_list}. Keep responses to 2 paragraphs max. Context: {context}"
        
        # Try each model until one works
        for model in MODELS:
            try:
                print(f"Trying model: {model}...")
                
                # Format prompt based on model family
                if "gemma" in model:
                    full_prompt = f"<start_of_turn>user\n{system_content}\n\nUser question: {prompt}<end_of_turn>\n<start_of_turn>model\n"
                elif "zephyr" in model or "mistral" in model:
                    full_prompt = f"<|system|>\n{system_content}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
                else:
                    full_prompt = f"{system_content}\n\nUser: {prompt}\n\nAssistant:"
                
                # Prepare payload
                payload = {
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": 300,
                        "temperature": 0.7,
                        "return_full_text": False
                    }
                }
                
                # Make the request
                output = self.query_api(model, payload)
                
                # Handle potential errors in response (e.g. model loading)
                if isinstance(output, dict) and "error" in output:
                    error_msg = output["error"]
                    print(f"API Error with {model}: {error_msg}")
                    # If model is loading, wait and retry once
                    if "loading" in error_msg.lower():
                        wait_time = output.get("estimated_time", 10)
                        print(f"Model loading, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        output = self.query_api(model, payload)
                
                # Parse successful response
                if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                    return output[0]["generated_text"].strip()
                elif isinstance(output, dict) and "generated_text" in output:
                     return output["generated_text"].strip()
                
                print(f"Unexpected response format from {model}: {output}")
                continue

            except Exception as e:
                print(f"Exception with {model}: {e}")
                continue
                
        return "I apologize, but I am experiencing technical difficulties finding an available model. Please try again later."
