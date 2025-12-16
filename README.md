# Tutorial: Ollama + HuggingFace for local LLM execution

<center>
<img src="img/ollama_and_huggingface.png">
</center>

[![Commit activity](https://img.shields.io/github/commit-activity/m/ijstokes/tutorial-local-llm)
](https://img.shields.io/github/commit-activity/m/ijstokes/tutorial-local-llm)
[![License](https://img.shields.io/github/license/ijstokes/tutorial-local-llm)](https://img.shields.io/github/license/ijstokes/tutorial-local-llm)

Tutorial on the use of Ollama to manage and serve local LLMs, leveraging the Ollama & Hugging Face model repositories.

- **Github repository**: <https://github.com/ijstokes/tutorial-local-llm/>


## OVERVIEW
Are we in a one-size-fits-all world of LLMs, and destined to burn tokens forever via API key access to third party LLM services?  There are some great reasons to keep using 3rd party LLM services, however many data practitioners want and need a degree of control, autonomy, and maintained data sovereignty.  This means diving into the world of self-managed LLMs, and a starting point for that is the Ollama framework for local model management, and the Hugging Face model repository.

## OUTCOMES

This tutorial will get you up and running with basic local LLMs, and get you experience and insight in the following:

* Mechanics of using Ollama to fetch and run models from the CLI and API
* Understanding trade-offs of self-managed vs SaaS-hosted LLMs
* Picking a fit-for-purpose LLM from the Hugging Face model repository
* Creating customized derivative LLMs, more targeted for your own purposes
* Basics of configuring LLM interactions for your desired performance criteria
* Understanding mechanisms for multi-model intput and structured output beyond text input and text output
* Use of several Python libraries for interacting with LLMs, including `ollama`, and `openai`
* Strategies for using LLMs for data analysis tasks

## PRE-REQUISITES

* Some Python experience
* Some command line experience (ideally Bash/Unix shell)
* Some Git experience
* Some Jupyter experience
* Laptop with at least 20GB free storage space (may be able to get away with 10 GB)
* Laptop with at least 8GB (will be tight, ideally 16GB or more)

## SETUP

**NOTE:** I've been informed the Python environment setup does not work on all systems. I'm in the process of paring it back so it will work on a standard Windows 11 system with Intel or AMD processors and no GPU, with only 8GB of RAM. 

1. Make sure you have a recent Python environment (3.10 or later, ideally already 3.12), and [UV](https://docs.astral.sh/uv/getting-started/installation/) or [Mamba](https://conda-forge.org/download/). You'll need to be be familiar with CLI operation (ideally Bash or similar).
2. Create an Ollama account: https://ollama.com/
3. Install Ollama locally on your laptop: https://ollama.com/download/ (60MB download, 120MB installed)
4. Start Ollama locally on your laptop, and login with your Ollama account
5. Check that your device has registered an API key automatically: https://ollama.com/settings/keys
6. Create an account on HuggingFace if you don't already have one: https://huggingface.co/
7. Download five LLMs:

```
ollama pull gemma3:270m 
ollama pull gemma2:2b
ollama pull tinyllama:1.1b
ollama pull qwen3:4b
ollama pull llava-phi3:3.8b
```

You can read more about these LLMs on their HuggingFace model card, or from the paper/blog post announcing their release:

* Google Gemma 3:270m (290MB) [HF](https://huggingface.co/google/gemma-3-270m) - [Google Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
*  Google Gemma 2:2B LLM (1.6GB) [HF](https://huggingface.co/google/gemma-2-2b) - [Google Blog](https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/)
* Tiny Llama 1.1B LLM (637MB) [HF](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) - [Paper](https://arxiv.org/abs/2401.02385)
* Qwen 3 4B LLM (2.5GB) [HF](https://huggingface.co/Qwen/Qwen3-4B) - [Qwen Blog](https://qwen.ai/blog?id=qwen3)
* Llava Phi3 LLM (2.9GB) [HF](https://huggingface.co/xtuner/llava-phi-3-mini-gguf) - [XTuner GH Repo](https://github.com/InternLM/xtuner)

8. Clone the repository:

```bash
git clone https://github.com/ijstokes/tutorial-local-llm.git
cd tutorial-local-llm
```

9. Setup the environment:

Conda/mamba:

```bash
mamba env create -f environment.yml
mamba activate tutorial-local-llm
```

UV:

```bash
uv add -r requirements.txt
source .venv/bin/activate     # Mac
source .venv/Scripts/activate # Windows
```

10. Run the smoke test:

```bash
python smoketest.py
```

You should get output which looks something like this:

```
Python interpreter: /path/to/tutorial-local-llm/bin/python
Python version:     3.12.12 | packaged by conda-forge | (main, Oct 22 2025, 23:34:53) [Clang 19.1.7 ]

huggingface_hub: 0.36.0
deepeval:        3.7.2
evidently:       0.7.16

Ollama models (local):

gemma3:270m                     gemma3  268.10M    291 MB   Q8_0    gguf
gemma2:2b                       gemma2  2.6B      1629 MB   Q4_0    gguf
tinyllama:1.1b                  llama   1B         637 MB   Q4_0    gguf
qwen3:4b                        qwen3   4.0B      2497 MB   Q4_K_M  gguf
llava-phi3:3.8b                 llama.  4B        2926 MB   Q4_K_M  gguf
```

If you're able to complete these steps you're ready for the Tutorial.

## AGENDA

1. Local environment setup (5 minutes, with background package downloading)
2. Why would you want to run LLMs locally? What are the economics, pros, and cons? (5 minutes)
3. Getting started with Ollama for local LLM execution (15 minutes + 15 minutes exercise)
4. Navigating HuggingFace and finding the LLM of your dreams (10 minutes + 10 minute exercise)
5. Multimodal inputs & generating structured output (10 minutes + 10 minutes exercise)
6. Q&A (5 minutes)

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
