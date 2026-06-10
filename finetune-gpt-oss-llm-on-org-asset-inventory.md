# Fine-Tuning gpt-oss-20b on a Free Colab T4: Building Your Own Asset Inventory AI Assistant

*How I taught a 20-billion-parameter model to answer "What is this IP?" — without buying a single GPU.*

---

## The 2 AM Problem

If you've ever worked in a SOC, you know this moment. An alert fires at 2 AM. The only thing you have is an IP address: `192.168.1.10`. Now the scavenger hunt begins — open the CMDB, search the asset inventory spreadsheet, ping the network team on Teams, check three different Excel files that all disagree with each other.

What you actually want is to type one question and get one answer:

> **You:** What is 192.168.1.10?
>
> **Assistant:** 192.168.1.10 is a Windows server running Windows Server 2025. It belongs to the Finance department and hosts the internal financial web application (FinApp portal). The server owner is Mr. Ravi. Criticality: High. Patch group: Wave-1.

This post walks you through building exactly that — a private AI assistant fine-tuned on your organization's asset inventory — using OpenAI's open-weight **gpt-oss-20b** model, the **Unsloth** library, and a **free Google Colab T4 GPU**. Total cost of training: **$0**.

No prior fine-tuning experience needed. If you can run cells in a notebook and edit a CSV file, you can replicate this.

---

## Part 1: Why Not Just Run a Giant Model Locally?

The giant-model plan fails on both ends — it costs a fortune, and after all that spend it still can't tell you who owns the server beaconing at 2 AM. Size buys general intelligence. It does not buy your knowledge. Only your data can do that.

Let's do the math on that.

### The hardware bill

A general-purpose frontier-class model needs serious VRAM just to *exist* in memory, let alone train:

| What you want to do | Typical hardware | Approximate cost |
|---|---|---|
| Run a 70B model at full precision | 2× NVIDIA A100 80GB | ~$30,000+ in hardware, or ~$3–4/hr cloud |
| Run a 120B+ model | 1× H100 node (80GB+) | $25,000–40,000 per GPU |
| Fine-tune gpt-oss-20b with standard frameworks | ~65 GB VRAM minimum | A100/H100 territory |
| **Fine-tune gpt-oss-20b with Unsloth** | **~14 GB VRAM** | **Free Colab T4** |

That last row is the entire point of this post. Unsloth's team rewrote how gpt-oss is loaded and quantized, and the result is that all other training methods need a minimum of 65GB VRAM to train gpt-oss-20b, while Unsloth needs only 14GB — which happens to fit on the free 15GB T4 GPU that Google Colab hands out at no cost.

**Analogy time:** running a giant general-purpose model in-house is like hiring a Nobel laureate as your office receptionist. Incredibly capable, absurdly expensive, and 99% of their knowledge is irrelevant to the question "which department owns this server?" What you actually need is the person who's worked in your building for ten years and knows every room.

### The "smaller + specialized beats bigger + generic" research

This isn't just my opinion — it's measured. In 2024, the research team at Predibase published **"LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4"** ([arXiv:2405.00732](https://arxiv.org/abs/2405.00732)). They fine-tuned 310 models across 10 base models and 31 tasks, and found that **4-bit LoRA fine-tuned models outperformed their base models by 34 points and GPT-4 by 10 points on average**.

Even more striking for our budget-conscious purposes: their LoRA Land collection of fine-tuned Mistral-7B models outperformed GPT-4 by 4–15% depending on the task, and each model was **fine-tuned for less than $8 on average** — with the whole collection served from a single A100. In a follow-up Fine-Tuning Index, fine-tuned models beat GPT-4 on **85% of specialized tasks tested** (legal contract review, medical classification, etc.).

Read that again: models small enough to run on commodity hardware, trained for the price of a sandwich, beating a frontier API model — *on the specific task they were trained for*. Your asset inventory is about as "specific task" as it gets.

### The data sovereignty bonus

There's a second reason this matters, especially in regulated environments (UAE, India, finance, government): **your asset inventory is sensitive data**. IP addresses, OS versions, application owners, patch levels — this is exactly the reconnaissance data an attacker dreams of. Sending it to a third-party API is a non-starter for many compliance teams. A model you fine-tuned yourself, running on your own hardware, never leaks a byte outside your perimeter.

---

## Part 2: What We're Building

**Goal:** A chat model that answers natural-language questions about your organization's assets.

**Base model:** [gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b) — OpenAI's open-weight 20B mixture-of-experts model, released under the permissive Apache 2.0 license, so you can customize and deploy it commercially without restriction.

**Training method:** QLoRA (4-bit quantization + Low-Rank Adapters). Instead of retraining all 20 billion parameters, we freeze the model and train a tiny set of "adapter" layers — only about 1% of the model's parameters are touched, which is what makes this fit on a T4.

**Analogy:** the base model is a printed encyclopedia. LoRA is a pack of sticky notes you add to the relevant pages. You never reprint the book — you just annotate it with *your* knowledge. The sticky notes (adapters) are a few hundred MB; the book stays untouched.

**Reference notebook:** This guide follows the official [Unsloth gpt-oss-20b Fine-tuning Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb) (from the [unslothai/notebooks repo](https://github.com/unslothai/notebooks)). We keep the same skeleton and swap in our own dataset. Unsloth's gpt-oss documentation is here: [unsloth.ai/docs](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss).

---

## Part 3: The Dataset — Turning Your Asset Inventory into Training Data

This is the part most tutorials skip, and it's the part that actually matters. Your model is only as good as the Q&A pairs you feed it.

### Step 3.1 — Start with your CMDB/inventory export

Most organizations already have this as a CSV (exported from a CMDB, Lansweeper, an Excel tracker, or even Defender for Endpoint's device inventory). Something like:

```csv
ip_address,hostname,asset_type,os,department,application,owner,criticality
192.168.1.10,FIN-WEB-01,Windows Server,Windows Server 2025,Finance,Internal Financial Web App,Mr. Ravi,High
192.168.1.25,HR-DB-02,Windows Server,Windows Server 2022,HR,HRMS Database,Ms. Priya,Medium
10.10.5.40,LNX-PROXY-01,Linux Server,Ubuntu 22.04 LTS,IT Infrastructure,Squid Proxy,Mr. Ahmed,High
```

### Step 3.2 — Convert rows into Q&A conversations

LLMs learn from *conversations*, not tables. So we programmatically turn every row into multiple question/answer pairs. Crucially, we generate **several phrasings of the same question** per asset, because users won't always ask the same way ("what is this IP", "who owns 192.168.1.10", "which department does this server belong to").

github repo to download the Q&A dataset and notebook "https://github.com/ravi231/LLM-Finetuning"


## Part 4: Fine-Tuning on Colab — Step by Step

> ⚠️ **Compliance note — read this before uploading anything to Colab.**
> Your real asset inventory is sensitive infrastructure data, and in most organizations uploading it to a personal Google Colab session violates data handling and compliance policy (it leaves your perimeter and lands on Google-managed infrastructure outside your control). The right split is:
>
> - **Use Colab for learning and testing only**, with the dummy/sample dataset provided with this post (fictional IPs, hostnames, and owners). That's exactly what we do in this walkthrough.
> - **Train on real data locally** — the same Unsloth code runs unchanged on an on-prem or company-controlled GPU machine. Download the notebook (`File → Download → .ipynb`) and follow [Unsloth's local gpt-oss guide](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune); a single workstation/server GPU with ~16GB VRAM (e.g., an RTX 4080/4090 class card) is enough for the same QLoRA recipe.
>
> This gives you the best of both worlds: a free GPU to learn the workflow risk-free, and a compliant pipeline for the real thing. Get sign-off from your data protection/compliance team before training on production inventory anywhere.

Open a new notebook at [colab.research.google.com](https://colab.research.google.com), then go to **Runtime → Change runtime type → T4 GPU**. That's the free tier. Now follow along — this mirrors the official Unsloth notebook, with our **sample dataset** swapped in. Unsloth's own guidance for Colab is simple: run cells top to bottom, and if a cell throws an error, just re-run it.

### Step 4.1 — Install Unsloth

```python
%%capture
!pip install --upgrade unsloth
```

The first cell installs Unsloth and its dependencies and prints your GPU info. You should see a Tesla T4 with ~15GB.

### Step 4.2 — Load gpt-oss-20b in 4-bit

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024   # plenty for short Q&A pairs
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",  # Unsloth's linearized version
    max_seq_length = max_seq_length,
    load_in_4bit = True,    # QLoRA — this is what makes T4 possible
    full_finetuning = False,
)
```

One important detail: you must use **Unsloth's `unsloth/gpt-oss-20b` upload**, not the raw OpenAI weights. The reason is genuinely interesting — gpt-oss stores its weights in a format that breaks standard quantization (the MoE experts account for roughly 19B of the 20B parameters), and Unsloth converted the architecture to a quantization-friendly linear format. That conversion is *why* this fits in under 14GB instead of 65GB.

This cell takes a few minutes — it's downloading ~12GB of model weights.

### Step 4.3 — Attach the LoRA adapters (the "sticky notes")

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,                    # adapter rank — 8 is fine for narrow tasks
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state = 3407,
)
```

You'll see output confirming that only ~1% of parameters are trainable. The other 99% — all of gpt-oss's general language ability — stays frozen.

Quick plain-language tour of these knobs too:

- **`r = 8`** — the "size" of each sticky note (the adapter rank). Bigger r = more writing space = the adapters can store more new knowledge, but they cost more memory and overfit easier. For a narrow task like inventory Q&A, 8 is plenty; you'd reach for 16–32 only if the model struggles to absorb a large, varied dataset.
- **`target_modules = [...]`** — *which pages of the book* get sticky notes. These seven names are the model's attention and feed-forward layers — the parts that do the actual "thinking." Targeting all of them is the standard recipe; you almost never change this list.
- **`lora_alpha = 16`** — how *loudly* the sticky notes speak relative to the original book. The common rule of thumb is alpha = 2 × r. Higher alpha = the new knowledge gets more weight over the base model's instincts.
- **`lora_dropout = 0`** — dropout randomly ignores a small % of connections during training to prevent over-memorization. We set it to 0 because for fact-recall tasks we *want* memorization, and Unsloth's kernels are also fastest with dropout off.
- **`use_gradient_checkpointing = "unsloth"`** — a memory-saving trade: instead of keeping every intermediate calculation in memory, the model throws some away and recalculates them when needed. Slightly slower, dramatically less VRAM — Unsloth's version of this trick is a big part of why we fit on a T4 at all.
- **`random_state = 3407`** — same reproducibility seed idea explained in Step 4.5.

### Step 4.4 — Load and format our dataset

gpt-oss uses OpenAI's "harmony" chat format under the hood, but the tokenizer's chat template handles all of that for us — we just apply it:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files = "asset_qa_v4_final.jsonl", split = "train")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched = True)
```

(Upload `asset_qa_v4_final.jsonl` to the Colab session via the folder icon on the left sidebar. Remember the compliance note at the top of this section: only the **sample/dummy dataset** goes to Colab. When you graduate to your real inventory, run this same notebook on a local or company-controlled GPU machine so the data never leaves your environment.)
github repo to download the dataset "https://github.com/ravi231/LLM-Finetuning"

### Step 4.5 — Train

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,   # effective batch size 4
        warmup_steps = 5,
        num_train_epochs = 2,              # 2–3 epochs for fact recall
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
```

**What do all these numbers actually mean?** Don't just copy-paste — here's each setting in plain language, because these are the knobs you'll eventually want to turn:

- **`per_device_train_batch_size = 1`** — how many examples the model studies *at the same time*. Think of it as how many flashcards you look at in one glance. We keep it at 1 because the T4's memory is almost full just holding the model; there's no room to look at multiple flashcards at once.

- **`gradient_accumulation_steps = 4`** — a memory trick to fake a bigger batch. The model looks at 4 examples *one after another*, takes notes each time, and only updates its brain after all 4. The learning effect is the same as a batch of 4, but the memory cost stays at a batch of 1. It's like reading four flashcards one by one and *then* forming your conclusion, instead of spreading all four on the table.

- **`warmup_steps = 5`** — for the first 5 training steps, the model learns extra gently, ramping up from a near-zero learning rate to the full one. Why? The same reason you don't sprint the moment you step out of bed — you stretch first. A model hit with full-strength updates from step one can get "shocked" early and never recover. The number 5 simply means the warm-up stretch lasts 5 steps; after that, training runs at full learning rate.

- **`num_train_epochs = 2`** — how many times the model reads your *entire* dataset cover to cover. One epoch = one full pass over all examples. Two epochs = it sees every Q&A pair twice. For fact memorization (our case), 2–3 passes is the sweet spot: one pass is often too little to remember, five passes and it starts parroting word-for-word (overfitting).

- **`learning_rate = 2e-4`** — the size of each correction the model makes when it gets an answer wrong. That's 0.0002 in normal notation. Big steps = learns fast but may wildly overshoot (like cranking the volume knob when you only needed a nudge). Tiny steps = stable but painfully slow. `2e-4` is the community's battle-tested default for LoRA fine-tuning.

- **`logging_steps = 1`** — print the training loss after every single step, so you can watch the learning curve live in the Colab output. Purely for your eyes; doesn't affect training.

- **`optim = "adamw_8bit"`** — the optimizer is the "coach" deciding exactly how to apply each correction. AdamW is the standard coach; the `8bit` version is the same coach keeping its notebook in shorthand — it stores its bookkeeping numbers in compressed 8-bit form, saving precious VRAM on our small GPU with virtually no quality loss.

- **`weight_decay = 0.01`** — a gentle pull that stops any single internal value from growing extreme. Think of it as a discipline rule against over-confidence: it nudges the model toward simpler, more general patterns instead of obsessively memorizing quirks of the training data. 0.01 is a light touch — standard practice.

- **`lr_scheduler_type = "linear"`** — how the learning rate changes over the run: after warm-up, it slides smoothly in a straight line from full strength down toward zero by the final step. Big corrections early when the model knows nothing, fine-tuned micro-adjustments at the end — like a sculptor switching from a chisel to sandpaper.

- **`seed = 3407`** — the starting point for all randomness (shuffling order, adapter initialization). Fixing it means anyone re-running this notebook gets the *exact same result* — essential for a tutorial. The specific value is arbitrary; 3407 is just a number the ML community uses as an in-joke (there's literally a paper titled "Torch.manual_seed(3407) is all you need").

- **`output_dir = "outputs"`** — simply the folder where training checkpoints get written. Nothing magical.

The one-line summary: **batch size and accumulation control memory, epochs and learning rate control how much it learns, warmup and scheduler control how gently it starts and ends, and seed makes it repeatable.** For your first run, change nothing — these defaults work.

Now go get a coffee (or two — the T4 is free, not fast). For ~2,000 examples, expect somewhere in the range of 1–3 hours depending on sequence lengths. Watch the training loss: it should fall steadily and flatten. If it crashes toward zero very early, you're overfitting — Unsloth's own docs warn against setting epochs/learning rate too high, and for fact-memorization tasks a loss that's *too* perfect means the model is parroting instead of generalizing across question phrasings.

### Step 4.6 — The moment of truth: test it

```python
messages = [
    {"role": "user", "content": "What is 192.168.1.10?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "low",   # gpt-oss feature: low/medium/high
).to(model.device)

from transformers import TextStreamer
_ = model.generate(**inputs, max_new_tokens = 256,
                   streamer = TextStreamer(tokenizer))
```

If everything worked, you'll see:

> *IP 192.168.1.10 is a Windows Server (hostname: FIN-WEB-01) running Windows Server 2025. It is part of the Finance department and hosts the Internal Financial Web App. The owner of this server is Mr. Ravi. Criticality: High.*

Notice the `reasoning_effort` parameter — that's a gpt-oss-specific feature letting you dial the model's chain-of-thought from low to high. For inventory lookups, "low" is perfect: we want fast recall, not deep deliberation.

### Step 4.7 — Save your work

```python
# Option A: save just the LoRA adapters (tiny, ~100-200MB)
model.save_pretrained("asset-inventory-lora")
tokenizer.save_pretrained("asset-inventory-lora")

# Option B: merge and export to GGUF for llama.cpp / Ollama / LM Studio
model.save_pretrained_gguf("asset-inventory-gguf", tokenizer)
```

Option B is the one that closes the loop on the cost story: a GGUF export of the fine-tuned model runs on a regular workstation — even CPU-only, slowly — via Ollama or llama.cpp. You trained on a free cloud GPU and you serve on hardware you already own. The "fortune" stays in your pocket. The next section walks through exactly that, step by step.

## Part 5: Honest Caveats (Read Before You Deploy)

I'd be doing you a disservice if I pretended fine-tuning is magic. Three things a noob-friendly tutorial should still be honest about:

**1. Fine-tuning bakes in facts as of training day.** Your inventory changes; the model's memory doesn't, until you retrain. The good news: with LoRA on a free T4, "retrain weekly" is a realistic, zero-cost pipeline (export CSV → regenerate JSONL → re-run notebook). For inventories that change *hourly*, pair this with retrieval (RAG) — fine-tune for the answer *format* and domain language, retrieve the live record for the facts. The hybrid is genuinely the best of both.

**2. LLMs can hallucinate, and in security that's dangerous.** A confidently wrong server owner at 2 AM is worse than no answer. This is why the negative examples ("that IP is not in our inventory") from Part 3 are non-negotiable, and why a tool like this should *assist* analysts, not replace verification for high-stakes actions.

**3. The model now contains your network map.** Treat the fine-tuned weights and the GGUF file as confidential assets. Anyone who can prompt the model can enumerate your inventory — so put it behind the same access controls as the CMDB itself, and log queries to it like you'd log CMDB access (your SIEM will thank you).

---

## Part 6: The Bottom Line

Let's put the two approaches side by side one last time:

| | Giant general model, self-hosted | Fine-tuned gpt-oss-20b (this post) |
|---|---|---|
| Hardware to train | A100/H100 class, $25k+ | Free Colab T4 |
| Training cost | N/A (you don't train it) | $0 (Predibase averaged <$8/model even on paid infra) |
| Knows *your* assets | No — needs everything in the prompt | Yes — it's in the weights |
| Accuracy on your task | Generic | Research shows fine-tuned models beat GPT-4 by ~10 points on average on specialized tasks |
| Data leaves your control | Often | Never |
| Serving hardware | Data-center GPUs | A workstation with Ollama |

The era of "bigger model = better answer" is over for narrow enterprise tasks. A 20B open-weight model, a few hundred rows of your own data, an afternoon on a free GPU — and you have an assistant that knows your environment better than any frontier model ever will.




## References

1. **Unsloth gpt-oss-20b Fine-tuning Colab notebook** — [colab.research.google.com/github/unslothai/notebooks/.../gpt-oss-(20B)-Fine-tuning.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)
2. **Unsloth: "Fine-tune gpt-oss with Unsloth"** (VRAM numbers, MoE quantization details) — [unsloth.ai/blog/gpt-oss](https://unsloth.ai/blog/gpt-oss)
3. **Unsloth gpt-oss fine-tuning tutorial docs** — [unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/tutorial-how-to-fine-tune-gpt-oss)
4. **Unsloth: "Saving models to Ollama" & "Saving to GGUF"** (export steps, chat template/EOS troubleshooting) — [unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-ollama)
5. **Zhao et al., "LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report"** (Predibase, 2024) — [arXiv:2405.00732](https://arxiv.org/abs/2405.00732)
6. **Predibase LoRA Land announcement** (25 fine-tuned Mistral-7B models, <$8 average training cost, single-GPU serving) — [predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4)
7. **OpenAI gpt-oss-20b model card** (Apache 2.0, consumer-hardware fine-tunability) — [huggingface.co/unsloth/gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b)
