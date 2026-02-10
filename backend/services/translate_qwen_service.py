from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import ast

class Translate_Qwen_Service:
    def __init__(self, path):
        tokenizer_path = path / "tokenizer"
        model_path = path / "model"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, tie_word_embeddings=False)
        print("Loaded Qwen Translate")

    def string_cleaner(self, raw_str):
        raw_str = raw_str.replace("'", '"')
    
        # 2. Fix the "id": 9: "Text" error (colon instead of comma)
        raw_str = re.sub(r'("id":\s*\d+):\s*', r'\1, "text": ', raw_str)
        
        # 3. Fix missing commas between objects: } { -> }, {
        raw_str = re.sub(r'\}\s*\{', '}, {', raw_str)
        
        # 4. Ensure it's wrapped in brackets
        if not raw_str.strip().startswith('['):
            raw_str = f"[{raw_str}]"
        if not raw_str.strip().endswith(']'):
            raw_str = f"{raw_str}]"
            
        return raw_str
    
    def parse_and_clean(self, json_str):
        """
        Input:
            json_str -> String: "[
                'id' : 'untranslated text',
                'id2': 'untranslated text',
                ...
            ]"
        
        Converts json_str into a json after cleaning
        """
        try:
            cleaned_json = self.string_cleaner(json_str)
            data = ast.literal_eval(cleaned_json) #json.loads(clean_json)

            result = {}
            for item in data:
                index = item.get('id', 0)
                text = item.get('text', '').replace('\n', ' ').strip()
                
                result[index] = text
                # print(f"{index}: {text}")
            return result
                
        except Exception as e:
            print(json_str)
            print(f"Failed to parse JSON: {e}")
            data = []
            items = re.findall(r'\{"id":\s*(\d+),\s*"text":\s*"(.*?)"\}', cleaned_json)
            for i, t in items:
                data.append({"id": int(i), "text": t})

    def translate(self, text):
        messages = [
            {
                "role": "system", 
                "content": """
                    You are a professional Manga Localizer. You will receive JSON-formatted Japanese OCR text from manga.

                    CRITICAL INSTRUCTIONS:

                    OUTPUT ONLY ENGLISH: Do not provide Romaji or Japanese in the final translation.

                    FIX OCR ERRORS: The input has errors (e.g., 'バー' might be 'バカ'). Use your knowledge of Naruto to correct them.

                    CONVINCING DIALOGUE: Use character-specific voices (e.g., Naruto's "Believe it!", Iruka's stern teacher tone).

                    MATCH IDs: Return the response as a JSON object matching the input IDs.
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
        return self.parse_and_clean(output_text)