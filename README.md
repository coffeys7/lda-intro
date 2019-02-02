### Topic Modeling

> **Source:** [Beginners guide to topic modeling in python](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)

- Automatically identify topics from text & derive hidden patterns
- Unsupervised
- Clusters bunches of words by topic
- Assumes documents are produced from a mixture of topics which generate words based on their probability distribution
- LDA starts from the beginning to determine how the topics would create the documents
- LDA specifically is a matrix factorization technique

Document-term matrix counts occurrences of words within their respecive document

> *Ex. W2/D3 = frequency of word 2 in document 3*

|  | w0 | w1 | w2 | ... | wN |
|---|:----:|:----:|:----:|:----:|:----:|
| d0 | 0 | 2 | 3 | 1 | 3 |
| d1 | 0 | 0 | 9 | 7 | 3 |
| d2 | 0 | 1 | 0 | 2 | 9 |
| d3 | 0 | 0 | 6 | 5 | 4 |
| ... | ... | ... | ... | ... | ... |
| dN | 3 | 4 | 7 | 2 | 4 |

LDA takes the document-term matrix and turns it into two lower dimension matrices

- *Document-Topic (M1) - dimensions = (N,K)*
- *Topic-Term (M2) - dimensions = (K,M)*

| N | K | M |
|---|---|---|
| # documents | # topics | # terms |

LDA makes use of sampling techniques in order to improve the distributions from these matrices

#### LDA Hyperparameters

- *Alpha:* document-topic density
	- high = more topics, low = less topics
- *Beta:* document-word density
	- high = more words per topic, low = less words per topic
- *Number of topics*
	- [Kullback Leibler Divergence Score](https://link.springer.com/chapter/10.1007%2F978-3-642-13657-3_43)
- *Number of topic terms*
	- higher for themes/concepts, lower for features/terms
- *Number of iterations*
	- max # iterations for LDA to reach convergence

[More on topic modeling](https://trainings.analyticsvidhya.com/courses/course-v1:AnalyticsVidhya+NLP101+2018_T1/about?utm_source=blog&utm_medium=beginners-guide-to-topic-modeling-in-python)

- Stemming and lemmatization aim to reduce morphological variation

#### Steps

1. Remove puncuation
2. Remove stopwords
3. Normalize the corpus
4. Create document-term matrix
