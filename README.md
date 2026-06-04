# RandomBench

<p align="center">
  <img src="assets/overview.jpg" width="900">
</p>

**RandomBench** is a benchmark for probing stochastic behavior and latent biases in Multimodal Large Language Models (MLLMs) under logic-neutral conditions. It contains 200 instances across text and vision modalities with perfectly equivalent options. Using metrics like Randomness Index (RI), Bias Intensity (BII), and Bias Consistency (BCI), RandomBench reveals pervasive “Stochastic Collapse,” where models often deviate from uniform randomness. The benchmark enables systematic evaluation of heuristic reliance, cross-lingual robustness, and modality-specific biases in state-of-the-art MLLMs.

---

## Repository Structure

```text
RandomBench/
├── RB_Vision/
│   ├── 1ebf2b8d.png
│   ├── ...
│   └── RB_Vision.json          # Image questions with metadata
├── RB_Text.json               # Text questions with metadata
├── settings.txt               # API keys and model configuration
├── photo_test.py              # Run image-based randomness tests
├── text_test.py               # Run text-based randomness tests
├── requirements.txt           # Python dependencies
├── assets/
│   └── overview.jpg
├── .gitignore
└── README.md
```

---

## Get Start

### 1. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Configure API Settings

Edit `settings.txt`:

```text
API_KEY = sk-your-key-here
BASE_URL = https://api.openai.com/v1
MODELS = gpt-5.1
REPEATS = 50
TEMPERATURE = 1.0
```

### 3. Run RB-Vision Experiments

Chinese prompts:

```bash
python photo_test.py --language zh --question-ids 1,2,3
```

English prompts:

```bash
python photo_test.py --language en --question-ids all
```

Results will be saved under:

```text
Results/photo_zh/
Results/photo_en/
```

### 4. Run RB-Text Experiments

Chinese prompts:

```bash
python text_test.py --question-ids 1-100 --prompt-language zh
```

English prompts:

```bash
python text_test.py --question-ids 1-100 --prompt-language en
```

Results will be saved under:

```text
Results/text_zh/
Results/text_en/
```

### 5. Label-Replacement Ablation

Greek-letter labels:

```bash
python text_test.py --question-ids 26-32 --replace-labels greek
```

Geometry-symbol labels:

```bash
python text_test.py --question-ids 26-32 --replace-labels geometry
```

Random labels:

```bash
python text_test.py --question-ids 26-32 --replace-labels random
```

---

## Evaluation Metrics

| Metric | Description | Desired Direction |
|----------|-------------|------------------|
| **RI** (Randomness Index) | Normalized Shannon entropy | Higher |
| **BII** (Bias Intensity Index) | KL divergence from a uniform distribution | Lower |
| **BCI** (Bias Consistency Index) | Maximum probability deviation from uniform | Lower |
| **JSD** (Jensen–Shannon Divergence) | Distribution shift between two conditions | Lower |

---

## Dataset Statistics

| Subset | Instances |
|---------|-----------|
| RB-Text | 100 |
| RB-Vision | 100 |
| Total | 200 |

---

## Citation

If you use RandomBench in your research, please cite:

```bibtex
@article{randombench2026,
  title={RandomBench: Evaluating Distributional Neutrality in Multimodal Large Language Models},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
