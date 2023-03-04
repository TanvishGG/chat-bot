from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

# save the tokenizer and model to disk

tokenizer.save_pretrained("dialogpt-medium")

model.save_pretrained("dialogpt-medium")

