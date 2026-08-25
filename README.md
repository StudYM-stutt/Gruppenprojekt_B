# Text Segmentation Pipeline – Group B

**Group B**
**Authors:** Yannic Mermi, Emma Hermann, Zeynep Yigit

This repository contains the implementation of a text segmentation project comparing **TextTiling** with **LLM-based narrative text segmentation**.

## Requirements

Create and activate a Python virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

All required Python packages and their versions are listed in `requirements.txt`.

---

## Main Pipeline

The main pipeline is used for the segmentation and evaluation of the main project texts.

```text
step_1.py  → TextTiling segmentation
step_2.py  → TextTiling evaluation
step_3.py  → TextTiling visualization
step_4.py  → LLM-based segmentation and evaluation
step_5.py  → LLM visualization
```

Adapted TextTiling implementations for German (`texttiling_de.py`) and English (`texttiling_eng.py`) are included.

Run the individual steps in order:

```bash
python step_1.py
python step_2.py
python step_3.py
python step_4.py
python step_5.py
```

---

## Synthetic Pipeline

The synthetic pipeline is used to create and evaluate synthetic texts based on **Robin Hood** and **Robinson Crusoe**.

```text
step_1.py  → Creation of synthetic text variants and gold boundaries
step_2.py  → LLM-based segmentation and evaluation
step_3.py  → Evaluation aggregation and visualization
```

Step 1 creates **20 synthetic text variants** divided into four granularity levels: coarse, medium, fine and very fine.

Run the synthetic pipeline in order:

```bash
python step_1.py
python step_2.py
python step_3.py
```

---

## LLM Configuration

The LLM segmentation supports:

* KISSKI / GWDG Academic Cloud
* OpenAI
* Ollama
* Dry run

For KISSKI, set the API key before running the corresponding LLM segmentation step.

**PowerShell:**

```powershell
$env:KISSKI_API_KEY="YOUR_API_KEY"
```
