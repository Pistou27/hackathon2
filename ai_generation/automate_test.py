from ethical_filter_v2 import ethical_filter
from generation import TextGenerator

gen = TextGenerator(model_path="gpt2-medium", max_new_tokens=60)

prompt = "Write a LinkedIn post targeting junior developers."
post   = gen.generate(prompt)
result = ethical_filter(post)

print("\nPOST:\n", post)
print("\nFILTER RESULT:", result)