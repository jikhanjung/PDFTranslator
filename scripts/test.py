from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "naver-clova-ocr/bros-base-uncased",
    trust_remote_code=True,
    cache_dir="d:/cache"
)

model = AutoModel.from_pretrained(
    "naver-clova-ocr/bros-base-uncased",
    trust_remote_code=True,
    cache_dir="d:/cache"
)

print(tokenizer.tokenize("Hello, this is a test."))