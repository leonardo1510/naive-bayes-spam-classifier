# Naïve Bayes Spam Classifier

A spam/ham email classifier built from scratch with NumPy, using a Naïve Bayes model on binary keyword features. No machine-learning libraries are used.

## Data

`data/training_spam.csv` (1000 emails) and `data/testing_spam.csv` (500 emails). The first column is the label (`1` = spam, `0` = ham); the other 54 columns are binary features that say whether a given keyword (e.g. "money", "free", "receive") or special character (e.g. `:`, `!`, `$`) appears in the message.

## Model

For a message with features $w_i$, the class is chosen to maximise

$$\log p(C=c) + \sum_{i:\,w_i=1} \log p(w_i \mid C=c)$$

* **Class priors** $p(C=c)$: proportion of spam and ham in the training set.
* **Class-conditional likelihoods** $p(w_i \mid C=c)$: keyword counts per class, with **Laplace smoothing** (`alpha`, default 1.0) so unseen keywords do not get zero probability.
* **Log-space** computation to avoid numerical underflow.

The implementation is the `SpamClassifier` class in [`spamclassifier.ipynb`](spamclassifier.ipynb) (`train()` and `predict()`).

## Results

On the 500-email test set:

| Metric | Value |
|---|---|
| Accuracy | 0.898 |
| Precision (spam) | 0.843 |
| Recall (spam) | 0.915 |

## Running

```bash
pip install -r requirements.txt
jupyter notebook spamclassifier.ipynb
```

## License
MIT
