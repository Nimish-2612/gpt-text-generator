GPT Language Model From Scratch (PyTorch)

This project implements a GPT-style autoregressive Transformer trained on the OpenWebText dataset using PyTorch.

The model learns grammar, structure, and writing patterns directly from raw internet text and can generate paragraphs autonomously without any prompt.

Instead of answering questions, the model behaves like a small pre-trained language model — similar to early GPT models — and writes continuous text from an initial start token.

Features

Transformer architecture implemented from scratch

Multi-Head Self Attention + Positional Embeddings

Autoregressive next-token prediction

Top-K sampling decoding

GPU training with PyTorch

Unprompted text generation (free-running language model)

Example Generation

The model is not given a question or instruction.
It simply starts writing:

The state of the country has been considered in the development of
economic policy and the results of the report indicate that the
administration would continue to support the program...


The output varies each run because sampling is stochastic.

How It Works

The model predicts the next character given previous characters:

P(next token | previous tokens)

After training on OpenWebText, the network learns statistical structure of language and produces realistic article-style text.

Generation starts from a newline seed:

start = "\n"


This allows the model to begin a natural paragraph rather than numeric tokens.

Running the Notebook

Open gpt_text_generator.ipynb

Load the trained weights

Run the unprompted generation cell

generate_unprompted(800)

Tech Stack

PyTorch

HuggingFace Datasets

CUDA GPU Training

Notes

This is a base pretrained language model, not instruction-tuned.
It continues text rather than answering questions — similar to early GPT models before chat fine-tuning.