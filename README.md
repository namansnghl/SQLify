# SQLify: Semantic Parser
> (SEQUELify)
> *A Transfer Learning Approach to Semantic Parsing SQL Queries*

## Navigation
- [Introduction](#introduction)
- [Related Work](#related-work)

# Introduction
This project addresses the challenge of navigating complex business databases as their size increases. We investigate the potential of pre-trained language models (BART, T5) and Neural Machine Translation (NMT) techniques to improve accessibility through transfer learning.

Leveraging the WikiSQL dataset, the project focuses on generating the SQL SELECT WHERE clause, a crucial component for filtering desired data. The findings provide valuable insights into the effectiveness of pre-trained models for text-to-SQL tasks, paving the way for further exploration in enhancing business database accessibility.


# Related Work
Natural Language to SQL (NL2SQL) conversion typically relies on sequence-to-sequence (seq2seq) models with attention mechanisms. These models function as encoder-decoder architectures. The encoder processes natural language sentences, while the decoder translates them into sequences of SQL query tokens.

Here, the attention mechanism plays a critical role. It allows the decoder to focus on relevant parts of the input sentence during each generation step. This is crucial for capturing long-range dependencies within the natural language and resolving potential ambiguities or references.

Popular datasets for training and evaluating NL2SQL models include:
- **WikiSQL**: Offers a rich collection of general-purpose SQL queries.
- **Spider**: Focuses on more complex, cross-domain queries.

While Long Short-Term Memory (LSTM) networks have been the go-to choice for encoders and decoders due to their ability to handle sequential data, recent advancements have explored alternative architectures such as:
- **Transformers**: Rely solely on attention mechanisms and have achieved state-of-the-art performance in various sequence transduction tasks, including Neural Machine Translation (NMT).
- **Convolutional Neural Networks (CNNs)**: Known for their effectiveness in text classification, they may also be promising for NL2SQL tasks.

This project builds upon a baseline seq2seq model with attention, treating SQL generation as an NMT task. This approach serves as a foundation for future exploration of more advanced architectures and the potential for incorporating domain-specific knowledge to achieve even better SQL translation.


## Models

### T5

### BART

### NMT


## Results
