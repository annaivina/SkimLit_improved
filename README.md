# SkimLit_improved

This project is inspired by the paper [Dernoncourt et al., 2016](https://arxiv.org/pdf/1612.05251), which focuses on classifying sections of medical paper abstracts.

### 📌 **Overview**
This model improves upon the original approach by incorporating: ✅ **BERT embeddings** for word representation.
It also includes:
- ✅ **Character-level embeddings** for fine-grained text understanding.
- ✅ **Additional position embeddings** to capture abstract structure.
- ✅ **BiLSTM layers** to process sequential text information.
- ✅ **Multi-input architecture**, combining different feature representations.

The model classifies **medical abstracts into 5 categories**, improving upon the baseline architecture.

### 📌 **Key Improvements**
✔ **BERT-based word embeddings** (instead of traditional static embeddings).  

### 📌 **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SkimLit_improved.git
   cd SkimLit_improved
