from TextSummerizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration

# text1 = '''Once upon a time, in a quaint little town nestled amidst rolling hills and lush greenery, there lived a young girl named Lily. She was known throughout the town for her kind heart and radiant smile. However, despite her popularity, Lily often felt a sense of loneliness deep within her heart.

# Every year, on Valentine's Day, the town would be abuzz with excitement as couples exchanged gifts and affectionate gestures. But for Lily, Valentine's Day was a painful reminder of her own solitude. She longed for someone to share her love with, someone who would cherish her just as she would cherish them.'''

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    
    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        # model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

        # inputs = tokenizer([text], max_length=1024, return_tensors="pt")
        # summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=50)
        # output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output
    
# predict_text = PredictionPipeline()
# predict_text.predict(text)