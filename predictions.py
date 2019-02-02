import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

nltk.download('wordnet')

DATA_DIR = 'data'
STOPWORDS = set(stopwords.words('english'))
EXCLUDE = set(string.punctuation)
LEMMATIZER = WordNetLemmatizer()

#--------------------------------------------------------------------------------

def load_documents(filename):
	with open(DATA_DIR + '/' + filename + '.json') as file:
		return json.load(file)

#--------------------------------------------------------------------------------

def split_words(doc):
	return [i for i in doc.lower().split()]

def remove_stopwords(doc):
	return " ".join([w for w in split_words(doc) if w not in STOPWORDS])

def remove_punctuation(doc):
	return ''.join(c for c in doc if c not in EXCLUDE)

def normalize(doc):
	return " ".join(LEMMATIZER.lemmatize(w) for w in split_words(doc))

def clean(doc):
	return normalize(remove_punctuation(remove_stopwords(doc)))

def create_lda_model(docs, num_topics = 3, num_passes = 50):
	cleaned_docs = [clean(doc).split() for doc in docs]
	dictionary = corpora.Dictionary(cleaned_docs)
	dt_matrix = [dictionary.doc2bow(doc) for doc in cleaned_docs]
	lda = gensim.models.ldamodel.LdaModel
	return lda(dt_matrix, num_topics = num_topics, id2word = dictionary, passes = num_passes)

#--------------------------------------------------------------------------------

def main():
	docs = load_documents('docs-v1')
	lda_model = create_lda_model(docs)
	print(lda_model.print_topics(num_topics = 3, num_words = 3))

if __name__ == "__main__":
	main()
