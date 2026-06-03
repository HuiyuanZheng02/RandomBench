# RandomBench

**RandomBench** is a benchmark designed to evaluate whether Multimodal Large Language Models (MLLMs) can maintain **distributionally neutral behavior** when selecting among perfectly equivalent options.

The benchmark contains **200 logic-neutral instances** across two modalities:

- **RB-Text**: 100 text-based tasks
- **RB-Vision**: 100 image-based tasks

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
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install openai tqdm
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

## Benchmark Overview

RandomBench evaluates whether model outputs remain invariant under transformations that should not affect rational decision-making, including:

- Option permutation
- Label replacement
- Cross-lingual prompting
- Modality changes (text vs. vision)

For each instance, all candidate options are intentionally designed to be semantically and logically equivalent, making any systematic preference indicative of selection bias rather than task-specific reasoning.

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
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions, suggestions, or bug reports, please contact:

```text
your_email@example.com
```