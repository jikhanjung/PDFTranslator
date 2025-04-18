from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "naver-clova-ocr/bros-base-uncased",
    trust_remote_code=True,
    cache_dir="./bros-tokenizer"
)