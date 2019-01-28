import pandas
import nltk
import string
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


input_file = "Dataset for Detection of Cyber-Trolls.json"
punctuation = list(string.punctuation)
tokenizer = TweetTokenizer()
language = "english"

# Load json data as records and remove columns where all values are NaN
data = pandas.read_json(input_file, lines=True, orient="records").dropna(axis=1, how='all')


# Change annotation column to contain label
data = data.join(data.pop('annotation').apply(lambda x: x['label'][0]))
data = data.rename(columns={data.columns[1]: "label"})

# Convert into lower case
data['content'] = data['content'].apply(lambda x: x.lower())

# Tokenize the tweets
data['content'] = data['content'].apply(lambda x: tokenizer.tokenize(x))

data['content'] = data['content'].apply(lambda word_list: list(filter(lambda word: word not in punctuation, word_list)))

# Download list of stop words
nltk.download('stopwords')
data['content'] = data['content'].apply(lambda word_list: list(filter(lambda word: word not in stopwords.words(language), word_list)))

offensive_freq_dist = nltk.FreqDist([word for word_list in data.loc[data['label'] == '1']['content'].values for word in word_list])
non_offensive_freq_dist = nltk.FreqDist([word for word_list in data.loc[data['label'] == '0']['content'].values for word in word_list])
tokens_unigram = nltk.FreqDist([word for word_list in data['content'].values for word in word_list])

print(offensive_freq_dist.most_common(10))
print(non_offensive_freq_dist.most_common(10))
print(len(set(tokens_unigram)))

offensive_freq_dist.plot(30, cumulative=False, title="Offensive tweets frequency")
non_offensive_freq_dist.plot(30, cumulative=False, title="Non-offensive tweets frequency")

nltk.download('punkt')
def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

two_grams_freq_dist_off = nltk.FreqDist([bigram for word_list in data.loc[data['label'] == '1']['content'].values for bigram in get_ngrams(' '.join(word_list),2)])
two_grams_freq_dist_non = nltk.FreqDist([bigram for word_list in data.loc[data['label'] == '0']['content'].values for bigram in get_ngrams(' '.join(word_list),2)])
tokens_bigram = nltk.FreqDist([bigram for word_list in data['content'].values for bigram in get_ngrams(' '.join(word_list),2)])

print(two_grams_freq_dist_off.most_common(10))
print(two_grams_freq_dist_non.most_common(10))
print(len(set(tokens_bigram)))

two_grams_freq_dist_off.plot(30, cumulative=False, title="Offensive tweets frequency with bi-gram")
two_grams_freq_dist_non.plot(30, cumulative=False, title="Non-offensive tweets frequency with bi-gram")
