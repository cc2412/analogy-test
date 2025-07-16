# Word Analogy Test with Word2Vec

## Overview

- This project implements a word analogy test using pre-trained Word2Vec embeddings from Google News. 
- The model is evaluated using the standard `question-words.txt` analogy test set. 
- Model loading and evaluation are performed using the `gensim` library.

---

## How Word Analogies Work

Word analogies test the model's understanding of relationships between words, which is based on vector arithmetic: `king - man + woman ≈ queen`.

> Examples:
> 
> > - "king" is to "queen" as "man" is to "woman"
> >
> > - "run" is to "ran" as "eat" is to "ate"

## Dataset

> **Google News Word2Vec**
>
> > - Source: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
> >
> > - 300 dimensions, ~3M words
> >
> > - Chose this model because it is widely-used, well-supported, and ideal for analogy evaluation.

> **question-words.txt**
> 
> > - Source: https://github.com/piskvorky/gensim/blob/develop/gensim/test/test_data/questions-words.txt
> >
> > - Over 19,000 analogy questions spanning 14 categories (5 semantic and 9 syntactic)
> >
> > - Chose this dataset because it is a standard benchmark for evaluating embedding models.

### Notes

- Even though the Google News Word2Vec has around 3 million words, my script only loads the 500,000 most frequent words to reduce memory usage and speed up evaluation. If I had more RAM and time, I would have tried testing with more words to compare accuracy.

- Gensim automatically restricts the vocabulary during analogy evaluation for performance reasons. It evaluates word analogies for the top 300,000 words in the model.

---

## Results

### Overall Performance

- Accuracy: 74.01% 
- Correct answers: 14,307
- Total questions: 19,330
- Runtime: 24 minutes (my laptop is slow but I will get a better one, I promise)

### Category Breakdown

| Category                    | Questions | Correct | Accuracy   |
|-----------------------------|-----------|---------|------------|
| capital-common-countries    | 506       | 421     | 83.20%     |
| capital-world               | 4368      | 3552    | 81.32%     |
| currency                    | 808       | 230     | 28.47%     |
| city-in-state               | 2467      | 1779    | 72.11%     |
| family                      | 506       | 436     | 86.17%     |
| gram1-adjective-to-adverb   | 992       | 290     | 29.23%     |
| gram2-opposite              | 812       | 353     | 43.47%     |
| gram3-comparative           | 1332      | 1216    | 91.29%     |
| gram4-superlative           | 1122      | 987     | 87.97%     |
| gram5-present-participle    | 1056      | 829     | 78.50%     |
| gram6-nationality-adjective | 1599      | 1442    | 90.18%     |
| gram7-past-tense            | 1560      | 1020    | 65.38%     |
| gram8-plural                | 1332      | 1159    | 87.01%     |
| gram9-plural-verbs          | 870       | 593     | 68.16%     |
| **Total**                   | 19330     | 14307   | **74.01%** |

### Semantic vs Syntactic Categories

- Semantic categories (5): capital-common-countries to family

- Syntactic categories (9): gram1-adjective-to-adverb to gram9-plural-verbs 

| Category Type | Questions | Correct | Accuracy   |
|---------------|-----------|---------|------------|
| Semantic      | 8655      | 6418    | 74.15%     |
| Syntactic     | 10675     | 7889    | 73.90%     |
| **Total**     | 19330     | 14307   | **74.01%** |

### Observations

> **Semantic vs Syntactic Performance**
> 
> > - Very similar performance between the two types.
> > 
> > - Semantic categories performed slightly better than syntactic categories, although in the original Word2Vec paper it is the opposite.
> > 
> > - Hypothesis: I only loaded a limited vocabulary. As such, if some low-frequency syntactic forms got excluded, the syntactic performance could drop.
> > 
> > - The difference is very small, probably not statistically significant in this case.

> **High Accuracy Categories**
> 
> > - gram3-comparative: 91.29%
> > - gram6-nationality: 90.18%
> > - gram4-superlative: 87.97%
>
> Comparatives and superlatives involve consistent, rule-based transformations that can be easily captured by embedding models.
> - Comparatives: add `er`, as in `fast -> faster`
> - Superlative: add `est`, as in `fast -> fastest`
>
> Nationalities are common in news and are also close to the country's name, reinforcing strong embeddings.

> **Low Accuracy Categories**
> 
> > - currency: 28.47
> > - gram1-adjective-to-adverb: 29.23
> > - gram2-opposite: 43.47
>
> Currency might be harder due to lower frequency whereas adjective-to-adverb and opposites are more abstract and difficult to capture without context. 

### Summary

The model captures both semantic and syntactic relationships well, achieving strong results in grammatical forms and basic analogies while struggling with more abstract or low-frequency relations. This accuracy is satisfactory for a vocabulary limited at 300k words.

---

## Extra: Initial Testing

Before running the script with 500,000 words and 14 categories, I tested it with only 50,000 words (ten times less) and 5 categories. I think this test is worth mentioning because of several observations I have made.

### Partial Evaluation Results (Reduced Model)

| Category                  | Questions | Correct | Accuracy   |
|---------------------------|-----------|---------|------------|
| capital-common-countries  | 506       | 423     | 83.60%     |
| currency                  | 206       | 79      | 38.35%     |
| city-in-state             | 2394      | 1751    | 73.14%     |
| family                    | 420       | 377     | 89.76%     |
| gram1-adjective-to-adverb | 992       | 307     | 30.95%     |
| **Subtotal**              | 4518      | 2937    | **65.01%** |

### Observations and Hypothesis

> **Higher accuracy for each individual category**
> 
> > - Less competition: With fewer words in vocabulary, the model has fewer incorrect options to choose from when finding the closest match.
> > 
> > - Higher-quality words: The top 50k words are the most frequent and likely have the best-trained embeddings.
> > 
> > - Less noise: Rare words with poor embeddings are excluded.

> **Lower overall accuracy:**
> 
> > - Missing words: Many analogy questions probably contained words not in the 50k vocabulary, so those questions got skipped entirely.
> > - Fewer total questions answered: If 30% of questions had out-of-vocabulary words, your denominator was much smaller.
> > - Selection bias: This test contains two of the worst scoring categories (currency and gram1), which brings down the overall accuracy.

Similar observations are made in the [original Word2Vec paper](https://arxiv.org/pdf/1301.3781). The paper mentions that:

> *"It can be seen that after some point, adding more dimensions or adding more training data provides
diminishing improvements"*. 
 
Although the table supporting this claim does not match our data due to different scales (they used way more training words, in the order of tens to hundreds of million), this is still an interesting fact to keep in mind.

With more time and resources, we could try to find the best dimensionality : training words ratio.