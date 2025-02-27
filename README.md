# FACT-AUDIT

The source code for peer review: [An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models]

This novel work provides a fresh perspective from dynamic fact-checking evaluation to audit the capacities of LLMs adaptively.

More information coming soon! Stay tuned.

## Requirements
```bash
tqdm==4.66.2

transformers==4.39.1

requests==2.31.0
```

## Install
- Set up the experimental environment.
```bash
conda create -n Fact-Audit python=3.8
conda activate Fact-Audit
pip install -r requirements.txt
```

## Start
- Evaluate LLMs on Fact-Checking tasks.
```bash
cd scripts
bash fact-audit.sh
```

