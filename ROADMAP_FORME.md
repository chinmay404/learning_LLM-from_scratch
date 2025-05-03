## Phase 1: Foundations & Setup

Learn basic NLP data handling alongside your GenAI environment setup.

| Day | Task                                                                                                                                                                                                                                                                                          | Completed Date | Page               |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------ |
| 1   | ~~**Setup environment:** Install Python, Jupyter, VS Code, TensorFlow, PyTorch~~                                                                                                                                                                                                              |                |                    |
| 2   | ~~**NLP Preprocess I:** Learn tokenization (word, subword/BPE) with Hugging Face’s `tokenizers` library [Gist](https://gist.github.com/svaza/2fa26306b340f0bd96dcad13b6a87ea6?utm_source=chatgpt.com)~~                                                                                       | 03-05-2025     | [[NLP Preprocess]] |
| 3   | **GenAI Intro:** Read a survey on GANs, VAEs, Transformers (e.g., NVIDIA Generative AI Path) [Google Cloud Skills Boost](https://www.cloudskillsboost.google/paths/183?utm_source=chatgpt.com)                                                                                                |                |                    |
| 4   | **NLP Preprocess II:** Text cleaning & normalization (regex, stop-words, lower-casing) [GitHub](https://github.com/iscloudready/Generative-AI-Learning-Roadmap?utm_source=chatgpt.com)                                                                                                        |                |                    |
| 5   | **ML Refresher:** Train/test split, accuracy/precision/recall—apply to a text classification sample [Microsoft Tech Community](https://techcommunity.microsoft.com/blog/educatordeveloperblog/how-to-learn-generative-ai-with-microsoft%E2%80%99s-free-course/4067112?utm_source=chatgpt.com) |                |                    |

---

## Phase 2: Neural Nets & Text Representations

Combine neural network basics with embeddings and vector spaces.

|Day|Task|
|---|---|
|6|**NN Basics:** Implement a 2-layer MLP on tabular data; understand activation & loss functions [W3Schools.com](https://www.w3schools.com/gen_ai/index.php?utm_source=chatgpt.com)|
|7|**NLP Embeddings I:** Train Word2Vec on a small corpus with Gensim; visualize via PCA [roadmap.sh](https://roadmap.sh/?utm_source=chatgpt.com)|
|8|**GenAI NN:** Build a simple fully-connected GAN generator & discriminator [Google Cloud Skills Boost](https://www.cloudskillsboost.google/paths/183?utm_source=chatgpt.com)|
|9|**NLP Embeddings II:** Explore subword embeddings (FastText) and OOV handling [igmguru.com](https://www.igmguru.com/blog/how-to-learn-generative-ai?utm_source=chatgpt.com)|
|10|**Optimization:** Experiment with SGD vs. Adam on both your GAN and a Word2Vec training run [Microsoft Tech Community](https://techcommunity.microsoft.com/blog/educatordeveloperblog/how-to-learn-generative-ai-with-microsoft%E2%80%99s-free-course/4067112?utm_source=chatgpt.com)|

---

## Phase 3: Adversarial & Variational Models

Dive into GANs/VAEs, anchoring understanding in NLP sequence models.

|Day|Task|
|---|---|
|11|**GAN Deep:** Train DCGAN on MNIST; measure FID score; relate to text GAN literature [Google Cloud Skills Boost](https://www.cloudskillsboost.google/paths/183?utm_source=chatgpt.com)|
|12|**Seq2Seq Refresher:** Build an RNN encoder–decoder with attention for toy machine translation [genai-handbook.github.io](https://genai-handbook.github.io/?utm_source=chatgpt.com)|
|13|**VAE Deep:** Implement VAE on MNIST; visualize latent manifold [Google Cloud Skills Boost](https://www.cloudskillsboost.google/paths/183?utm_source=chatgpt.com)|
|14|**NLP Seq Models:** Train a BiLSTM for sentiment analysis on IMDB; compute BLEU on synthesized paraphrases [ProjectPro](https://www.projectpro.io/article/learn-generative-ai/962?utm_source=chatgpt.com)|
|15|**Compare:** Document strengths/weaknesses of GAN vs. VAE vs. Seq2Seq for generation tasks|

---

## Phase 4: Transformer Architectures

Master Transformers—grounded by NLP tokenization and evaluation metrics.

|Day|Task|
|---|---|
|16|**Attention Is All You Need:** Read & summarize key equations [GitHub](https://github.com/mehulpratapsing/Generative-AI-Roadmap-2024?utm_source=chatgpt.com)|
|17|**Implement Attention:** Code scaled-dot-product & multi-head attention from scratch [GitHub](https://github.com/mehulpratapsing/Generative-AI-Roadmap-2024?utm_source=chatgpt.com)|
|18|**NLP Tokenization III:** Integrate a BPE tokenizer into your Transformer; compare vocab sizes [Gist](https://gist.github.com/svaza/2fa26306b340f0bd96dcad13b6a87ea6?utm_source=chatgpt.com)|
|19|**GenAI Transformer:** Build a tiny Transformer on a text-generation toy dataset|
|20|**Evaluation:** Compute perplexity and ROUGE on generated text; iterate prompt & decode strategies [Magai](https://magai.co/generative-ai-for-beginners/?utm_source=chatgpt.com)|

---

## Phase 5: Large Language Models & RAG

Fine-tune LLMs and build Retrieval-Augmented Generation, leveraging NLP search and vector stores.

|Day|Task|
|---|---|
|21|**Hugging Face Fine-tune:** Adapt BERT/GPT-2 for text classification/generation; evaluate with classification report & ROUGE [Google Cloud](https://cloud.google.com/blog/topics/training-certifications/new-generative-ai-trainings-from-google-cloud?utm_source=chatgpt.com)|
|22|**Embeddings & Search:** Load model embeddings into FAISS; run k-NN retrieval on custom docs [YouTube](https://www.youtube.com/watch?v=KU_2l0cuFtY&utm_source=chatgpt.com)|
|23|**RAG Prototype:** Combine retriever + LLM to answer FAQs from your documentation|
|24|**Latency & Optimization:** Quantize your fine-tuned model; benchmark inference time [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2023/05/from-novice-to-pro-the-epic-journey-of-mastering-generative-ai/?utm_source=chatgpt.com)|
|25|**API Deployment:** Expose your RAG service with FastAPI + Docker; write a client script|

---

## Phase 6: Multimodal & Ethics

Expand into text-to-image/audio while auditing NLP biases.

|Day|Task|
|---|---|
|26|**Text-to-Image:** Use Stable Diffusion; preprocess prompts with core NLP tokenization; evaluate FID [LinkedIn](https://www.linkedin.com/pulse/how-learn-generative-ai-step-by-step-guide-mastering-future-akhtar-pfaoc?utm_source=chatgpt.com)|
|27|**Audio Generation:** Explore Wav2Vec/Jukebox; convert text prompts into mel-spectrograms via NLP preprocessing|
|28|**Bias Audit:** Run toxicity classifiers (Perspective API) on GenAI outputs; mitigate via prompt filters [DevThink.AI newsletter](https://devthink.ai/p/your-2024-roadmap-to-mastering-generative-ai?utm_source=chatgpt.com)|
|29|**Explainability:** Visualize attention maps over input tokens for generated outputs|
|30|**Ethics Write-Up:** Document bias findings and mitigation strategies|

---

## Phase 7: Capstone Integration

Build two end-to-end projects showcasing both NLP and GenAI mastery.

|Day|Task|
|---|---|
|31|**Project A Kickoff:** Choose (e.g., RAG-driven chatbot); outline data & model pipeline|
|32|**Data Prep:** Collect text, tokenize, build embeddings; split train/val/test [GitHub](https://github.com/iscloudready/Generative-AI-Learning-Roadmap?utm_source=chatgpt.com)|
|33|**Model Build:** Fine-tune LLM; integrate retrieval; add safety filters|
|34|**Deployment:** Dockerize API; add monitoring/logging|
|35|**Project B Kickoff:** Choose (e.g., text-to-image storytelling); define NLP prompt templates|
|36|**Pipeline Build:** Preprocess text prompts; invoke Stable Diffusion; batch generation|
|37|**UI Prototype:** Create a minimal web front-end to showcase both projects|
|38|**Documentation & Portfolio:** Write README, usage guides, and link demos|

---

### Key References

1. IBM, “Natural Language Processing Overview” [GitHub](https://github.com/iscloudready/Generative-AI-Learning-Roadmap?utm_source=chatgpt.com)
    
2. NVIDIA, “Generative AI Learning Path” [Google Cloud Skills Boost](https://www.cloudskillsboost.google/paths/183?utm_source=chatgpt.com)
    
3. Hugging Face, “Tokenizers Documentation” [Gist](https://gist.github.com/svaza/2fa26306b340f0bd96dcad13b6a87ea6?utm_source=chatgpt.com)
    
4. Stanford CS224n, “Neural Networks for NLP” Lecture Notes [Microsoft Tech Community](https://techcommunity.microsoft.com/blog/educatordeveloperblog/how-to-learn-generative-ai-with-microsoft%E2%80%99s-free-course/4067112?utm_source=chatgpt.com)
    
5. Vaswani et al., “Attention Is All You Need” (2017) [GitHub](https://github.com/mehulpratapsing/Generative-AI-Roadmap-2024?utm_source=chatgpt.com)
    
6. Gensim, “Word2Vec Tutorial” [roadmap.sh](https://roadmap.sh/?utm_source=chatgpt.com)
    
7. Facebook AI, “FAISS: A library for efficient similarity search” [YouTube](https://www.youtube.com/watch?v=KU_2l0cuFtY&utm_source=chatgpt.com)
    
8. Google Research, “BPE and SentencePiece” [igmguru.com](https://www.igmguru.com/blog/how-to-learn-generative-ai?utm_source=chatgpt.com)
    
9. Papineni et al., “BLEU: a method for automatic evaluation of machine translation” [Magai](https://magai.co/generative-ai-for-beginners/?utm_source=chatgpt.com)
    
10. Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers” [Google Cloud](https://cloud.google.com/blog/topics/training-certifications/new-generative-ai-trainings-from-google-cloud?utm_source=chatgpt.com)
    
11. OpenAI, “Retrieval-Augmented Generation” Blog Post [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2023/05/from-novice-to-pro-the-epic-journey-of-mastering-generative-ai/?utm_source=chatgpt.com)
    
12. Perspective API Documentation, “Detecting Toxicity” [DevThink.AI newsletter](https://devthink.ai/p/your-2024-roadmap-to-mastering-generative-ai?utm_source=chatgpt.com)
    
13. FairSeq, “Seq2Seq and attention implementations” [genai-handbook.github.io](https://genai-handbook.github.io/?utm_source=chatgpt.com)
    
14. Research at Google, “Stable Diffusion Guide” [LinkedIn](https://www.linkedin.com/pulse/how-learn-generative-ai-step-by-step-guide-mastering-future-akhtar-pfaoc?utm_source=chatgpt.com)
    
15. Kaiser et al., “Quantizing neural networks” [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2023/05/from-novice-to-pro-the-epic-journey-of-mastering-generative-ai/?utm_source=chatgpt.com)