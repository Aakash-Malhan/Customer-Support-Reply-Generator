# Customer-Support-Reply-Generator

*(Styled + Templates • CPU-friendly • Model Switch)*

DEMO: https://huggingface.co/spaces/aakash-malhan/customer-support-lora


## ✨ Features
- **CPU-friendly** models: `Qwen/Qwen2.5-0.5B-Instruct` (very fast) and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- **Style presets** for Banking/Fintech, E-commerce, Tech Support, Hospitality.
- **Brand voice** field (e.g., “formal, concise, premium”).
- **Language switch** (English, Spanish, German, French, Portuguese, Italian, Arabic).
- **Issue severity** slider → more empathy and prioritization.
- **Guardrails** to avoid risky replies (PII, promises, etc.).
- **One-click templates** (Password reset, Shipment delay, Refund request, Account update).
- **Sampling controls**: `max_new_tokens`, `temperature`, `top_p`.
# 🤖 Customer Support Reply Generator   

---

## 📸 Demo Screenshots  

<div align="center">

<img src="images/screenshot1.png" alt="Demo Screenshot 1" width="48%"/>  
<img src="images/screenshot2.png" alt="Demo Screenshot 2" width="48%"/>  

</div>

## ✨ Features  

- **Pretrained lightweight LLMs** (CPU-friendly) with model switch (e.g., Qwen, TinyLlama).  
- **Style presets**: Banking, Fintech, Formal, Friendly, etc.  
- **One-click templates** for common support issues (password reset, shipment delay, etc.).  
- **Adjustable severity** for empathetic vs. concise tone.  
- **Runs locally or on Hugging Face Spaces**.  


## ⚡ How it works  

1. User enters a **customer message**.  
2. Select **style preset** + optional **brand voice**.  
3. Model generates a **polite, tailored support response**.  
4. Supports fine-tuning or LoRA adapters for domain-specific replies.   

```bash
git clone https://github.com/yourusername/Customer-Support-Reply-Generator.git
cd Customer-Support-Reply-Generator
