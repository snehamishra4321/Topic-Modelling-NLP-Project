from gensim.models import CoherenceModel
# , LdaMulticore
from gensim.models.ldamodel import LdaModel

class CoherenceCalculator():

    def __init__(self, topic_range, dictionary, alpha, beta, data_lemmatized) -> None:
        
        self.alpha = alpha
        self.dictionary = dictionary
        self.beta = beta
        self.topic_range = topic_range
        self.data_lemmatized = data_lemmatized

    # supporting function
    def compute_coherence_values_parallelized(self, corpus):
        
        model_results_for_corpus = {
    #                 'Validation_Set': [],
                    'Topics': [],
                    'Alpha': [],
                    'Beta': [],
                    'Coherence': []
                    }
        
        for k in self.topic_range:
            # iterate through alpha values
            for a in self.alpha:
                # iterare through beta values
                for b in self.beta:

                    # lda_model = LdaMulticore(corpus=corpus,
                    #                     id2word=self.dictionary,
                    #                     num_topics=k, 
                    #                     random_state=100,
                    #                     chunksize=100,
                    #                     passes=10,
                    #                     alpha=a,
                    #                     eta=b)

                    lda_model = LdaModel(corpus=corpus,
                                        id2word=self.dictionary,
                                        num_topics=k, 
                                        random_state=100,
                                        chunksize=100,
                                        passes=10,
                                        alpha=a,
                                        eta=b)

                    coherence_model_lda = CoherenceModel(model=lda_model, texts=self.data_lemmatized, dictionary=self.dictionary, coherence='c_v', processes=1)
                    
                    model_results_for_corpus['Topics'].append(k)
                    model_results_for_corpus['Alpha'].append(a)
                    model_results_for_corpus['Beta'].append(b)
                    model_results_for_corpus['Coherence'].append(coherence_model_lda.get_coherence())
                        
        return model_results_for_corpus  