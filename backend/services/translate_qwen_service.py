from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import ast
from helpers import get_project_root

class Translate_Qwen_Service:
    def __init__(self, path=None):
        if not path:
            ROOT = get_project_root()
            path = ROOT / "backend" / "models" / "Qwen"

        tokenizer_path = path / "tokenizer"
        model_path = path / "model"

        if not tokenizer_path.exists() or not model_path.exists():
            print(f"Qwen tokenizer/model not found at {path}. Attempting to download.")
            self.load_model()
        
        if tokenizer_path.exists() and model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, tie_word_embeddings=False)
            print("Loaded Qwen Translate")
        else:
            raise FileNotFoundError(f"Error: Could not find or retrieve {model_path}")

    def translate(self, text):
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
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=400)
        output_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            print(f"LLM output was not valid JSON, {output_text}")
            return {"error": "Invalid JSON", "raw": output_text}
    
    def load_model(self):
        ROOT = get_project_root()
        MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
        QWEN_DIR = ROOT / "backend" / "models" / "Qwen"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(QWEN_DIR / "tokenizer")
        model.save_pretrained(QWEN_DIR / "model")

        print(f"Downloaded Qwen LLM to: {QWEN_DIR}")
        return str(QWEN_DIR)