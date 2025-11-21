# Tutorial: Ollama + HuggingFace for local LLM execution

[![Release](https://img.shields.io/github/v/release/ijstokes/tutorial-local-llm)](https://img.shields.io/github/v/release/ijstokes/tutorial-local-llm)
[![Build status](https://img.shields.io/github/actions/workflow/status/ijstokes/tutorial-local-llm/main.yml?branch=main)](https://github.com/ijstokes/tutorial-local-llm/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ijstokes/tutorial-local-llm/branch/main/graph/badge.svg)](https://codecov.io/gh/ijstokes/tutorial-local-llm)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ijstokes/tutorial-local-llm)](https://img.shields.io/github/commit-activity/m/ijstokes/tutorial-local-llm)
[![License](https://img.shields.io/github/license/ijstokes/tutorial-local-llm)](https://img.shields.io/github/license/ijstokes/tutorial-local-llm)

Tutorial on the use of Ollama to manage and serve local LLMs

- **Github repository**: <https://github.com/ijstokes/tutorial-local-llm/>


## OVERVIEW
Are we in a one-size-fits-all world of LLMs, and destined to burn tokens forever via API key access to third party LLM services?  There are some great reasons to keep using 3rd party LLM services, however many data practitioners want and need a degree of control, autonomy, and maintained data sovereignty which means diving into the world of self-managed LLMs, even if they are sourced from the more than 2 million models available in the HuggingFace repository.

## SETUP

1. Make sure you have a recent Python environment (3.10 or later), and [UV](https://docs.astral.sh/uv/getting-started/installation/).  You'll need to be be familiar with CLI operation (ideally Bash or similar).
2. Create an Ollama account: https://ollama.com/
3. Install Ollama locally on your laptop: https://ollama.com/download/ (60MB download, 120MB installed)
4. Start Ollama locally on your laptop, and login with your Ollama account
5. Check that your device has registered an API key automatically: https://ollama.com/settings/keys
6. Create an account on HuggingFace if you don't already have one: https://huggingface.co/
7. Download four LLMs:

```
ollama pull gemma3:270m 
ollama pull gemma2:2b
ollama pull tinyllama:1.1b
ollama pull qwen3:4b
```

You can read more about these LLMs on their HuggingFace model card, or from the paper/blog post announcing their release:

* Google Gemma 3:270m (290MB) [HF](https://huggingface.co/google/gemma-3-270m) - [Google Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
*  Google Gemma 2:2B LLM (1.6GB) [HF](https://huggingface.co/google/gemma-2-2b) - [Google Blog](https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/)
* Tiny Llama 1.1B LLM (637MB) [HF](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) - [Paper](https://arxiv.org/abs/2401.02385)
* Qwen 3 4B LLM (2.5GB) [HF](https://huggingface.co/Qwen/Qwen3-4B) - [Qwen Blog](https://qwen.ai/blog?id=qwen3)

8. Clone the repository:

```bash
git clone https://github.com/ijstokes/tutorial-local-llm.git
cd tutorial-local-llm
```

9. Setup the environment:

Conda/mamba:

```bash
mamba env create -f environment.yml
```

UV:

```bash
uv add -r requirements.txt
source .venv/bin/activate
```

10. Run the smoke test:


```bash
python smoketest.py
```

You should get output which looks something like this:

```
huggingface_hub: 0.36.0
deepeval:        3.7.2
evidently:       0.7.16


Ollama models (local):

gemma2:2b                     	gemma2	2.6B	  1629 MB	Q4_0	gguf
gemma3:270m                   	gemma3	268.10M	   291 MB	Q8_0	gguf
glm-4.6:cloud                 	glm4	355B	     0 MB	FP8
mistral:7b                    	llama	7.2B	  4372 MB	Q4_K_M	gguf
qwen:latest                   	qwen2	4B10. Run the smoke test:


```bash
python smoketest.py
```

You should get output which looks something like this:

```
huggingface_hub: 0.36.0
deepeval:        3.7.2
evidently:       0.7.16


Ollama models (local):

gemma3:270m                     gemma3  268.10M    291 MB   Q8_0    gguf
gemma2:2b                       gemma2  2.6B      1629 MB   Q4_0    gguf
tinyllama:1.1b                  llama   1B         637 MB   Q4_0    gguf
qwen3:4b                        qwen3   4.0B      2497 MB   Q4_K_M  gguf
```

## AGENDA

1. Welcome, concept overview and target tutorial outcomes (5 minutes)
2. Local environment setup (5 minutes, with background package downloading)
3. Why would you want to run you LLMs locally? Is it even possible? How do AI Appliances fit in? (10 minutes)
4. Navigating HuggingFace and finding the LLM of your dreams (5 minutes + 10 minute exercise)
5. Using Ollama from the GUI and from the CLI (5 minutes + 10 minute exercise)
6. Using the Ollama Python library and leveraging it from code (5 minutes + 10 minute exercise)
7. Offloading LLM to an AI Appliance (5 minutes + 5 minute exercise)
8. Next steps for AI Autonomy, Data Sovereignty, and Advanced Analytics in the age of GenAI and LLMs (5 minutes)
9. Q&A (10 minutes)

## FURTHER DETAILS

This tutorial will get you comfortable navigating the HuggingFace model
repository, and using the Ollama ecosystem for local model management and
hosting.  You'll learn how together these are the GitHub and Git of the LLM
world.  While we may not be able to get those LLMs behaving in a repeatable,
deterministic way at least you'll have the tools to choose exactly which LLMs
you're using and an ability to retain data sovereignty by self-hosting LLMs as
part of your overall workflow.  You'll leave with an expanded set of options
for leveraging LLMs that are running off your laptop, off your own cloud
server, or on affordable AI Appliances such as an Nvidia DGX Spark or Nvidia
Jetson Orin Nano Super.

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
