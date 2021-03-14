import numpy as np
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import nltk

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('reuters')
    nltk.download('punkt')

    doc_ids = list(filter(lambda doc: doc.startswith("training/"), reuters.fileids()))
    dataset = []

    for index in doc_ids:
        raw_document = reuters.raw(index)
        split_document = raw_document.split('\n', 1)

        title = split_document[0]
        content = split_document[1].strip()
        tokenized = word_tokenize(content)
        token_words = list(filter(lambda token: token.isalpha(), tokenized))

        if len(token_words) > 700:
            dataset.append((title, token_words))

        if len(dataset) >= 10:
            break

    porter = PorterStemmer()

    stemming_analysis = []

    for title, document in dataset:
        stemmed_dict = dict()
        regular_dict = dict()

        for word in document:
            stemmed = porter.stem(word)

            if word in regular_dict:
                regular_dict[word] += 1
            else:
                regular_dict[word] = 1

            if stemmed in stemmed_dict:
                stemmed_dict[stemmed] += 1
            else:
                stemmed_dict[stemmed] = 1

        stemming_analysis.append((title, document, regular_dict, stemmed_dict))

    print(f"{'Title':<50}\t{'Number of words':<20}\t{'Number of distinct words':<30}\t{'Number of distinct terms':<30}")
    for document in stemming_analysis:
        title = document[0]
        number_of_words = len(document[1])
        number_of_distinct_words = len(document[2])
        number_of_distinct_terms = len(document[3])

        print(f"{title:<50}\t{number_of_words:>20}\t{number_of_distinct_words:>30}\t{number_of_distinct_terms:>30}")

    for document in stemming_analysis:
        title = document[0]
        terms = document[3]

        print(f'\n\n{title} analysis:')
        print(f"{'Term':<20}\t{'Number of occurrences':<25}")
        extracted_stats = [(term, occurrences) for term, occurrences in terms.items()]
        extracted_stats.sort(key=lambda stats: stats[1], reverse=True)

        for term, occurrences in extracted_stats:
            print(f"{term:<20}\t{occurrences:>25}")

    for document in stemming_analysis:
        terms = document[3]
        indexes = list(range(len(terms)))
        x_resolution = 25.0
        y_resolution = 5.0
        occurrences = [occurrences for _, occurrences in terms.items()]

        plt.figure(dpi=200, figsize=(10, 4))
        plt.title(f"{document[0]} terms occurrences")
        plt.xlabel("Term index in occurrences dictionary")
        plt.ylabel("Number of occurrences")
        plt.xticks(np.arange(0, max(indexes) + x_resolution, x_resolution))
        plt.yticks(np.arange(0, max(occurrences) + y_resolution, y_resolution))
        plt.scatter(indexes, occurrences, s=5)
        plt.show()

    stop_words = stopwords.words("english")

    stop_words_analysis = dict()
    filtered_dataset = []

    for document in stemming_analysis:
        terms = document[3]
        filtered_dict = dict()

        for term in terms:
            if term in stop_words:
                if term in stop_words_analysis:
                    stop_words_analysis[term] += terms[term]
                else:
                    stop_words_analysis[term] = terms[term]
            else:
                if term in filtered_dict:
                    filtered_dict[term] += terms[term]
                else:
                    filtered_dict[term] = terms[term]

        filtered_dataset.append((document[0], filtered_dict))

    print(f"\n\n{'Stop word':<20}\t{'Number of occurrences':<25}")
    extracted_stats = [(stop_word, occurrences) for stop_word, occurrences in stop_words_analysis.items()]
    extracted_stats.sort(key=lambda stats: stats[1], reverse=True)

    for term, occurrences in extracted_stats:
        print(f"{term:<20}\t{occurrences:>25}")
