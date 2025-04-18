from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="HURIDOCS/pdf-segmentation",
    filename="pdf_tokens_type.model",
    local_dir="./models/pdf-segmentation"
)