#not currently in use
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from pathlib import Path
from helpers import get_project_root


class Translate_Tencent_Service:
    def __init__(self, ocr_path=None, device=None):
        ROOT = get_project_root()
        self.base_model_path = Path(os.getenv("MODEL_PATH", ROOT / "backend" / "models"))
        
        if not ocr_path:
            ocr_path = self.base_model_path / "TencentHY"
        else:
            ocr_path = Path(ocr_path)

        tokenizer_path = ocr_path / "tokenizer"
        model_path = ocr_path / "model"

        if not tokenizer_path.exists() or not model_path.exists():
            print(f"TencentHY tokenizer/model not found at {ocr_path}. Attemping to download")
            self.load_model()
        
        if tokenizer_path.exists and model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, tie_word_embeddings=False,)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print("Loaded Tencent Translate")
        else:
            raise FileNotFoundError(f"Error: Could not find or retrieve {model_path}")
    
    def translate(self, text):
        messages = [
            {"role": "system", "content": """
             You are a professional Manhua translator.  You will receive JSON-formatted Japanese OCR text from manga.
             
             CRITICAL INSTRUCTIONS:

                OUTPUT ONLY ENGLISH: Do not provide Romaji or Japanese in the final translation.

                FIX OCR ERRORS: The input has errors (e.g., 'バー' might be 'バカ'). Correct them based on the dialogues.

                MATCH IDs: Return the response as a JSON object matching the input IDs.
             
             """},
            {"role": "user", "content": f"{text}"}
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True       
        ).to(self.device)


        outputs = self.model.generate(
            **tokenized_chat, 
            max_new_tokens=2048
        )

        prompt_length = tokenized_chat.input_ids.shape[1] #length of original text at the start of string
        new_tokens = outputs[0][prompt_length:] #remove original text to keep only translated text

        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # print(output_text)
        return output_text

    def load_model(self):
        TENCENT_DIR = self.base_model_path / "TencentHY"
        DOWNLOAD_MODEL = "tencent/HY-MT1.5-1.8B"

        tokenizer = AutoTokenizer.from_pretrained(DOWNLOAD_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DOWNLOAD_MODEL, device_map="auto") 
        tokenizer.save_pretrained(TENCENT_DIR / "tokenizer")
        model.save_pretrained(TENCENT_DIR / "model")

        print(f"Downloaded TencentHY LLM to: {TENCENT_DIR}")
        return str(TENCENT_DIR)