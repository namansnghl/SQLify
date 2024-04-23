# SQLify: Semantic Parser
*A Transfer Learning Approach to Semantic Parsing SQL Queries*

![](https://github.com/namansnghl/SQLify/blob/main/media/t2s_resized.png)

# Navigation
* [Introduction](#introduction)
* [Related Work](#related-work)
* [Models](#models)
    * [Neural Machine Translation (NMT)](#nmt)
    * [T5-Small](#t5-small)
    * [BART-Base](#bart-base)
* [Results](#results)
* [Conclusion](#conclusion)
* [References](#references)

# INTRODUCTION
This project addresses the challenge of navigating complex business databases as their size increases. We investigate the potential of pre-trained language models (BART, T5) and Neural Machine Translation (NMT) techniques to improve accessibility through transfer learning.

Leveraging the [WikiSQL](https://paperswithcode.com/dataset/wikisql) dataset, the project focuses on generating the SQL SELECT WHERE clause, a crucial component for filtering desired data. The findings provide valuable insights into the effectiveness of pre-trained models for text-to-SQL tasks, paving the way for further exploration in enhancing business database accessibility.


# RELATED WORK
Natural Language to SQL (NL2SQL) conversion typically relies on sequence-to-sequence (seq2seq) models with attention mechanisms. These models function as encoder-decoder architectures. The encoder processes natural language sentences, while the decoder translates them into sequences of SQL query tokens.

Here, the attention mechanism plays a critical role. It allows the decoder to focus on relevant parts of the input sentence during each generation step. This is crucial for capturing long-range dependencies within the natural language and resolving potential ambiguities or references.

Popular datasets for training and evaluating NL2SQL models include:
- **WikiSQL**: Offers a rich collection of general-purpose SQL queries.
- **Spider**: Focuses on more complex, cross-domain queries.

While Long Short-Term Memory (LSTM) networks have been the go-to choice for encoders and decoders due to their ability to handle sequential data, recent advancements have explored alternative architectures such as:
- **Transformers**: Rely solely on attention mechanisms and have achieved state-of-the-art performance in various sequence transduction tasks, including Neural Machine Translation (NMT).
- **Convolutional Neural Networks (CNNs)**: Known for their effectiveness in text classification, they may also be promising for NL2SQL tasks.

This project builds upon a baseline seq2seq model with attention, treating SQL generation as an NMT task. This approach serves as a foundation for future exploration of more advanced architectures and the potential for incorporating domain-specific knowledge to achieve even better SQL translation.


# MODELS

# NMT
![Architecture of NMT Model](https://github.com/namansnghl/SQLify/blob/main/media/NMT_arch.png)

Neural Machine Translation (NMT) is a sophisticated approach to automatic translation powered by artificial neural networks. Unlike traditional statistical methods, NMT processes the entire source sentence as a whole and generates the target translation more fluently and coherently. At its core, NMT uses an encoder to convert the source text into a contextual representation, and a decoder to produce the translated text. On top of this, an additional Attention layer is added to the model to take into account the context for each token in the input with the output. For this, the Bahdanau Attention mechanism was used to capture the attention between each output token against all input tokens.

# T5-SMALL
T-5 small is one of the checkpoints of the T-5 model created by Google, with 60 million parameters. The authors introduced a unified framework that transforms various text-based language problems into a text-to-text format, facilitating systematic comparisons across different aspects of transfer learning. Their study encompasses pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors across numerous language understanding tasks. Leveraging scale and a new dataset named the "Colossal Clean Crawled Corpus," the authors achieve state-of-the-art results on various benchmarks including summarization, question answering, and text classification.

# BART-BASE
BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence model developed by Facebook AI Research. It is a transformer-based encoder-decoder model. It is a powerful transformer model designed for various natural language processing tasks. It utilizes a bidirectional encoder like BERT and an autoregressive decoder similar to GPT. This model is pre-trained using a two-step process: first, the text is corrupted using a chosen noise function, and then the model learns to reconstruct the original text. This approach enables BART to understand and generate coherent text, making it particularly effective for tasks like summarization, translation, text classification, and question answering. While the raw model can be utilized for tasks like text infilling, its primary strength lies in fine-tuning it on supervised datasets for specific tasks


# RESULTS

[Benchmarks](https://paperswithcode.com/sota/sql-to-text-on-wikisql)

![Performance Metric](https://github.com/namansnghl/SQLify/blob/main/media/performance_metrics.png)

We hypothesized that the T5 model, known for its exceptional language modeling capabilities, would perform well in our task. However, our experiments revealed that BART outperformed both T5 and the standard NMT approach. BART's ability to handle noisy data, stemming from its training on diverse datasets, equipped it to generate accurate translations even for complex passages. While quantitative metrics showed a modest advantage for BART, qualitative analysis yielded promising results. BART-Base achieved a BLEU score of 95.5%, surpassing T5-Small's score of 94.4%. Similarly, BART-Base secured a METEOR score of 97.9%, edging out T5-Small's score of 97.4%. Notably, the incorporation of attention mechanisms allowed these models to achieve superior performance compared to state-of-the-art graph-based models on the WikiSQL dataset.

# REFERENCES

- Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015). Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493.
- Xu, K., Wu, L., Wang, Z., Feng, Y., Witbrock, M., & Sheinin, V. (2018). Graph2seq: Graph to sequence learning with attention-based neural networks. arXiv preprint arXiv:1804.00823.
- Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. CoRR, abs/1709.00103, 2017.
- Katsogiannis-Meimarakis, G., & Koutrika, G. (2023). A survey on deep learning approaches for text-to-SQL. The VLDB Journal, 1-32.
- Kumar, A., Nagarkar, P., Nalhe, P., & Vijayakumar, S. (2022). Deep Learning Driven Natural Languages Text to SQL Query Conversion: A Survey. arXiv preprint arXiv:2208.04415.
- Xu, X., Liu, C., & Song, D. (2017). Sqlnet: Generating structured queries from natural language without reinforcement learning. arXiv preprint arXiv:1711.04436.
- Xu, K., Wang, Y., Wang, Y., Wen, Z., & Dong, Y. (2021). Sead: End-to-end text-to-sql generation with schema-aware denoising. arXiv preprint arXiv:2105.07911.
- Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
