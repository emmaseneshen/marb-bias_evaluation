## Results and Analysis

We evaluate reporting bias using both an autoregressive model (GPT-2) and a masked language model (DeBERTa) on MARB-style paired sentences.

### Key Findings

1. **Model-dependent reporting bias**

The two model types exhibit fundamentally different behaviors:

* **GPT-2 (autoregressive)** consistently assigns higher likelihood to marked sentences (e.g., “a Black person”), suggesting that attribute-marked descriptions are often treated as more expected.
* **DeBERTa (masked LM)** shows the opposite trend for several attributes, strongly favoring unmarked sentences (e.g., “a person”), indicating that attribute insertion is often treated as unexpected.

2. **Divergence across attributes**

The discrepancy is especially pronounced for certain descriptors:

* *Asian* and *Hispanic* show strong negative bias in DeBERTa but positive bias in GPT-2.
* *Black* remains relatively more balanced across both models.

This suggests that reporting bias is not uniform across attributes and is highly sensitive to model architecture.

3. **Comparison with MARB paper**

The original MARB study reports that marked attributes generally increase perplexity, meaning they are less expected.

* DeBERTa aligns with this trend.
* GPT-2 contradicts it, suggesting that autoregressive models may encode different assumptions about when attributes should be mentioned.

### Interpretation

These results indicate that reporting bias is not a universal property of language models, but rather an emergent behavior shaped by:

* model architecture (masked vs autoregressive)
* training dynamics
* token prediction mechanisms

This highlights the importance of evaluating bias across multiple model types rather than relying on a single benchmark.
