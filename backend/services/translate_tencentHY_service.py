#not currently in use
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
from pathlib import Path


class Translate_Tencent_Service:
    def __init__(self, ocr_path=None, device=None):
        tokenizer_path = ocr_path / "tokenizer"
        model_path = ocr_path / "model"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, tie_word_embeddings=False,)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Loaded Tencent Translate")
    
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