# Fine-tuning DocMagic's model (free, on Google Colab)

Phase 2 of this project: fine-tune **Llama 3.2 3B** on logistics/ERP/CRM Q&A
so the model speaks your domain natively, then publish it on Hugging Face Hub.

## Why fine-tune when RAG already works?

RAG gives the model *facts*. Fine-tuning gives it *behavior*: domain
vocabulary (CBM, detention, e-way bill, GRN, lead stages), answer format,
and refusal style. The resume-worthy part is publishing the model with
before/after eval numbers.

## Steps

1. **Build the dataset** — 300–1000 Q&A pairs in `dataset.jsonl`
   (see `dataset_template.jsonl` for the format). Sources:
   - Questions you and your colleagues actually get asked at work
   - Generate drafts with a free LLM from your SOPs, then hand-fix them
     (never ship machine-generated pairs unreviewed)
2. **Open Google Colab** (free T4 GPU: Runtime -> Change runtime type -> T4).
3. Upload `finetune_colab.py` and `dataset.jsonl`, then run:
   ```
   !pip install unsloth
   !python finetune_colab.py
   ```
4. **Evaluate**: hold out 30 pairs, compare base vs fine-tuned answers.
   Record the numbers — they go in the model card and your resume.
5. **Publish**: `model.push_to_hub("your-hf-username/docmagic-llama-3.2-3b")`
   with a model card describing data, method, and eval results.
6. **Use it in the app**: point `LLM_BASE_URL`/`LLM_MODEL` in `.env` at your
   model served via Ollama locally (`ollama create` from the exported GGUF)
   or a Hugging Face endpoint.
