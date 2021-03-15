from math import log

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

    topics = ['BANK', 'FINANCE', 'STOCK', 'DEBT']
    dataset = []

    print(f'\n\n2. Selecting 10 Reuters documents with over 700 words each around topics: {topics}')
    for index in reuters.fileids():
        raw_document = reuters.raw(index)
        split_document = raw_document.split('\n', 1)

        title = split_document[0]

        if any(topic in title.upper() for topic in topics):
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

    print('\n\n3. Document terms and words statistics')
    print(f"{'Title':<50}\t{'Number of words':<20}\t{'Number of distinct words':<30}\t{'Number of distinct terms':<30}")
    for document in stemming_analysis:
        title = document[0]
        number_of_words = len(document[1])
        number_of_distinct_words = len(document[2])
        number_of_distinct_terms = len(document[3])

        print(f"{title:<50}\t{number_of_words:>20}\t{number_of_distinct_words:>30}\t{number_of_distinct_terms:>30}")

    print('\n\n4. Terms occurrences tables')
    for document in stemming_analysis:
        title = document[0]
        terms = document[3]

        print(f'\n{title} analysis:')
        print(f"{'Term':<70}\t{'Number of occurrences':<25}")
        switched_dict = dict()
        for term, occurrences in terms.items():
            if occurrences in switched_dict:
                switched_dict[occurrences] += f", {term}"
            else:
                switched_dict[occurrences] = term

        extracted_stats = [(combined_terms, occurrences) for occurrences, combined_terms in switched_dict.items()]
        extracted_stats.sort(key=lambda stats: stats[1], reverse=True)

        for term, occurrences in extracted_stats:
            print(f"{term:<70}\t{occurrences:>25}")

    print('\n\n5. Terms frequency plots')
    for index, document in enumerate(stemming_analysis):
        terms = document[3]
        indexes = list(range(len(terms)))
        x_resolution = 25.0
        y_resolution = 5.0
        occurrences = [occurrences for _, occurrences in terms.items()]

        plt.figure(dpi=200, figsize=(10, 4))
        plt.title(f"5.{index + 1}. {document[0]} terms occurrences")
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

    print('\n\n6. Stop list')
    switched_dict = dict()
    for stop_word, occurrences in stop_words_analysis.items():
        if occurrences in switched_dict:
            switched_dict[occurrences] += f", {stop_word}"
        else:
            switched_dict[occurrences] = stop_word

    max_length = len(max(switched_dict.values(), key=len)) + 10
    print(f"{'Stop word':<{max_length}}\t{'Number of occurrences':<25}")

    extracted_stats = [(stop_words, occurrences) for occurrences, stop_words in switched_dict.items()]
    extracted_stats.sort(key=lambda stats: stats[1], reverse=True)

    for term, occurrences in extracted_stats:
        print(f"{term:<{max_length}}\t{occurrences:>25}")

    unique_terms = set()

    for _, terms in filtered_dataset:
        unique_terms = unique_terms.union(set(terms.keys()))

    print('\n\n7. Document-term matrix')
    frequency_matrix = f"{'Document':^55}"
    for term in unique_terms:
        frequency_matrix += f"\t{term:^15}"

    frequency_matrix += '\n'
    for document in filtered_dataset:
        frequency_matrix += f"{document[0]:<55}"

        for term in unique_terms:
            frequency_matrix += f"\t{document[1].get(term, 0):>15}"

        frequency_matrix += '\n'

    print(frequency_matrix)

    print('\n\n8. TF-IDF matrix')
    tf_idf_matrix = f"{'Document':^55}"
    for term in unique_terms:
        tf_idf_matrix += f"\t{term:^15}"

    tf_idf_matrix += '\n'
    for document in filtered_dataset:
        tf_idf_matrix += f"{document[0]:<55}"

        for term in unique_terms:
            tf = document[1].get(term, 0) / sum(document[1].values())
            idf = log(len(filtered_dataset) / sum([1 if term in doc[1] else 0 for doc in filtered_dataset]), 2)
            tf_idf_matrix += f"\t{tf * idf:>15.4f}"

        tf_idf_matrix += '\n'

    print(tf_idf_matrix)
