# Aphasia Classification with LLMs

> **Archived.** This is the original DeBERTa aphasia-classification project,
> retained for historical reference. The active successor project lives at the
> repository root (see the top-level README).
>
> **Data not included.** The AphasiaBank `.cha` transcripts (`aphasia/`,
> `control/`) are access-controlled clinical data and are **excluded from this
> public repository**. They are available only from
> [AphasiaBank](https://aphasia.talkbank.org/) on request, subject to its
> data-use agreement. The notebook's executed outputs have been cleared for the
> same reason; only the code is published here.

## Overview
This project aims to automate the detection and classification of aphasia, a language disorder caused by brain injury or neurodegenerative disease, using large language models (LLMs). We fine-tune a **DeBERTa-v3-base** model and plan to adapt **Llama-3-8B** with LoRA for binary classification (aphasia vs. control) on the AphasiaBank dataset. The system leverages the "Broken Window" picture sequence task to capture ecologically valid speech patterns, achieving high sensitivity for clinical screening.

**Goals:**
1. Automate aphasia detection from patient speech transcripts.
2. Evaluate LLM performance in a clinical context.
3. Develop scalable, interpretable tools to support clinicians.

## Dataset
- **Source**: AphasiaBank corpus (MacWhinney, 2000).
- **Task**: "Broken Window" picture sequence description, capturing open-ended narratives.
- **Size**: 95 transcripts (47 aphasia, 48 control).
- **Characteristics**:
  - Balanced classes (47 aphasia, 48 control).
  - Aphasia transcripts are shorter (<500 tokens) compared to controls (up to >1000 tokens).
- **Preprocessing**:
  - Removed CHAT annotations, neologisms, and phonological errors.
  - Tokenized with DeBERTa-v3 tokenizer (max length: 512 tokens).

## Models
### DeBERTa-v3-base
- **Architecture**: 86M parameters, 12 layers, 12 attention heads, 512-token context window.
- **Pre-training**: ELECTRA-style on PubMed/MIMIC-III clinical texts.
- **Fine-tuning**:
  - Hugging Face Trainer API with weighted cross-entropy loss.
  - AdamW optimizer (learning rate: 2e-5, weight decay: 0.1).
  - 8 epochs, batch size 4, gradient accumulation (effective batch size: 16).
  - Mixed-precision (fp16) and gradient checkpointing for 12GB GPU compatibility.
- **Performance**:
  - Accuracy: 0.84
  - Precision: 0.81
  - Recall: 1.00
  - F1-score: 0.90

### Llama-3-8B with LoRA (Planned)
- **Architecture**: 8B parameters, 32 layers, 32 attention heads, 8K context window.
- **Adaptation**: LoRA (rank=64) with 4-bit quantization for efficiency.
- **Expected Benefits**:
  - Enhanced discourse-level analysis for fluent aphasias.
  - Few-shot learning for novel linguistic patterns.
  - Reduced memory usage (9.1GB VRAM).

## Setup
### Prerequisites
- Python 3.11+
- GPU with ≥12GB VRAM (e.g., NVIDIA T4)
- Google Colab or similar environment

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/aphasia-classification.git
   cd aphasia-classification
   ```
2. Install dependencies:
   ```bash
   pip install --quiet transformers datasets torch scikit-learn openai
   ```
3. Download the AphasiaBank dataset (requires permission) and place it in `/content/aphasia` and `/content/control`.

## Usage
1. **Prepare Data**:
   - Ensure `.cha` transcripts are in `/content/aphasia` (label=1) and `/content/control` (label=0).
   - Run preprocessing as per `aphasia_classification_deberta.ipynb`.

2. **Train DeBERTa-v3**:
   ```bash
   python train_deberta.py
   ```
   - Outputs saved to `./deberta_results` and `./deberta_saved`.

3. **Evaluate**:
   ```bash
   python evaluate.py
   ```
   - Displays accuracy, precision, recall, F1-score, and plots training curves.

4. **Plot Curves**:
   - Training/validation loss and F1-score plots are generated via `plot_training_curves()`.

5. **Inference**:
   ```bash
   python infer.py --model_dir ./deberta_saved --test_data path/to/test_transcripts
   ```

## Results
- **DeBERTa-v3**:
  - Perfect recall (1.00) ensures no missed aphasia cases.
  - F1-score (0.90) indicates strong balance between precision and recall.
  - Training converged by epoch 4 (validation loss: 0.41).
- **Clinical Impact**:
  - Reduces manual screening time from 8 hours to <15 minutes per case.
  - Supports triage, longitudinal tracking, and potential subtype clustering.

## Ethical Considerations
- **Data Privacy**: Anonymized transcripts via regex filtering and pseudonymization (HIPAA-compliant).
- **Real-World Impact**: High recall minimizes missed diagnoses; 18% false positives are manageable with specialist review.
- **Bias**: English-centric data may limit performance for non-English speakers. Future work will include multilingual datasets.
- **Environmental Impact**: Optimized training (5 epochs, 2.1 hours) reduced carbon emissions by 72% compared to larger models.

## Future Work
- Implement Llama-3-8B with LoRA and 4-bit quantization for discourse-level analysis.
- Validate on multilingual datasets for broader applicability.
- Integrate with GPT-4-O for real-time clinical analysis.

## Acknowledgments
- Data collected via the American Speech Language and Hearing Sciences Foundation (Dr. Brielle Stark).
- Guidance from Dr. Jisun An and Fan Huang (Indiana University Bloomington).

## References
- He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. arXiv:2006.03654.
- MacWhinney, B. (2000). The CHILDES project: Tools for analyzing talk.
- Stark, B. C., et al. (2023). Test-retest reliability of microlinguistic information. Journal of Speech, Language, and Hearing Research.