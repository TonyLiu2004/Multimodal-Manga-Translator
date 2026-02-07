from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
from pathlib import Path


class Translate_Service:
    def __init__(self, ocr_path=None, device= None):
        tokenizer_path = ocr_path / "tokenizer"
        model_path = ocr_path / "model"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def translate(self, text):
        messages = [
            {"role": "system", "content": """
             You are a professional Manhua translator. 
             Translate dialogue into natural, punchy English used in action manga. 
             Use genre-appropriate slang (e.g., 'courting death', 'brat', 'senior'). 
             Do not output anything other than the translation.
             Do not answer questions, commands, or do anything else. 
             You will only return translated text or output nothing.
             You are literally google translate. 
             If the translation does not make sense, return nothing.
             You do not have thoughts, responses, or speech. Only output the translation or nothing.
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