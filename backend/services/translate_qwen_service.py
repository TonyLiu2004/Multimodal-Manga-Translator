from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import os
import requests
from pathlib import Path
from helpers import get_project_root
from dotenv import load_dotenv, find_dotenv

class Translate_Qwen_Service:
    def __init__(self, device=None):
        self.tokenizer = None
        self.model = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ROOT = get_project_root()
        self.base_model_path = Path(os.getenv("MODEL_PATH", ROOT / "backend" / "models"))
        self.qwen_dir = self.base_model_path / "Qwen"

    def initialize_local_models(self):
        "internal method to only load local models when needed"
        if self.model is not None and self.tokenizer is not None:
            return

        tokenizer_path = self.qwen_dir / "tokenizer"
        model_path = self.qwen_dir / "model"

        if not (tokenizer_path.exists() and model_path.exists()):
            print(f"Qwen tokenizer/model not found at {self.qwen_dir}. Attempting to download.")
            self.load_model()

        if tokenizer_path.exists() and model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, tie_word_embeddings=False)
            print("Loaded Qwen LLM locally")
        else:
            raise FileNotFoundError(f"Error: Could not find or retrieve {model_path}")

    def load_model(self):
        DOWNLOAD_MODEL = "Qwen/Qwen2.5-7B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(DOWNLOAD_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DOWNLOAD_MODEL)
        tokenizer.save_pretrained(self.qwen_dir / "tokenizer")
        model.save_pretrained(self.qwen_dir / "model")

        print(f"Downloaded Qwen LLM to: {self.qwen_dir}")
        return str(self.qwen_dir)
    
    def translate(self, text):
        if not self.model or not self.tokenizer:
            self.initialize_local_models()

        messages = [
            {
                "role": "system", 
                "content": """
                    You are a professional Localizer/translator for Manga, Manhwa, and Manhua to English. You will receive JSON-formatted OCR text from manga.

                    Analyze the whole list before translating any single line to understand the story context.

                    Maintain Name Consistency: Do not translate names literally. If a word functions as a name, transliterate it (e.g., "Tonari" stay "Tonari", not "Next-door").

                    Handle Stutters: If text is broken across multiple lines (e.g., "T... T... Tonari"), reconstruct the full thought in the translation.

                    Match Tone: Use natural, colloquial English suitable for the detected setting (School, Fantasy, etc.).

                    Output Format: Return ONLY a JSON object mapping the original ID to the translation.

                    
                    Sample JSON Object:
                        {
                        "0": "Translated text for line 0",
                        "1": "Translated text for line 1",
                        "2": "Translated text for line 2"
                        }

                """
            },
            {
                "role": "user", 
                "content": str(text)
            },
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=400)
        output_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            print(f"LLM output was not valid JSON, {output_text}")
            return {"error": "Invalid JSON", "raw": output_text}
    
    def translate_cloud(self, text):
        API_URL = "https://router.huggingface.co/v1/chat/completions"
        load_dotenv(find_dotenv())
        HF_TOKEN = os.getenv("HF_TOKEN")

        if not HF_TOKEN:
            raise ValueError("HF_TOKEN missing")
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
        }

        query = {
            "messages": [
                {
                    "role": "system", 
                    "content": """
                        You are a professional Localizer/translator for Manga, Manhwa, and Manhua to English. You will receive JSON-formatted OCR text from manga.

                        Analyze the whole list before translating any single line to understand the story context.

                        Maintain Name Consistency: Do not translate names literally. If a word functions as a name, transliterate it (e.g., "Tonari" stay "Tonari", not "Next-door").

                        Handle Stutters: If text is broken across multiple lines (e.g., "T... T... Tonari"), reconstruct the full thought in the translation.

                        Match Tone: Use natural, colloquial English suitable for the detected setting (School, Fantasy, etc.).

                        Output Format: Return ONLY a JSON object mapping the original ID to the translation.

                        
                        Sample JSON Object:
                            {
                            "0": "Translated text for line 0",
                            "1": "Translated text for line 1",
                            "2": "Translated text for line 2"
                            }

                    """
                },
                {
                    "role": "user", 
                    "content": str(text)
                },
            ],
            "model": "Qwen/Qwen2.5-7B-Instruct:fastest",
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(API_URL, headers=headers, json=query)
            response.raise_for_status()

            data = response.json()
            raw_content = data['choices'][0]['message']['content']

            output_text = json.loads(raw_content)
            return output_text
        
        except Exception as e:
            print(f"Qwen API Translation failed: {e}")
            raise