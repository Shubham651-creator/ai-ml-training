# Natural Language Processing

    To overcome cons of 'one-hot encoding' method, we will use "Word embeddings and Similarity"

### How do computer measure Similarity?
1. Euclidean Distance
    - Magnitude differs a lot

2. Cosine similarty
    - Its focus on Direction instead of magnitude

## Word2vec
- Instead of defining meaning manually, let the model read lots of text. The model learns meaning automotically.
- Distributional Hypothesis: Words that appear in similar context usually have similar meaning

## RNN (Recurring Nerural Network)

> One word at a time + Remember previous information

## LSTM (Long Short-Term Memory)
- RNN has problem that new memory keep replacing old memory.
- Even LSTM struggles when seqeuence become very long.

### Attention
    Word + direct access to all relevant word

--- 

## Transformer
### QKV
    1. Query - What information am I looking for?
    2. Key - What information do I contain?
    3. Value - What info should I provide?

### Multi-head Attention Mechanism

    Sentence
    ↓
    Head 1 (Persons)

    Head 2 (Places)

    Head 3 (Actions)

    Head 4 (Relationships)

    ↓
    Combine Results
    ↓
    Better Understanding