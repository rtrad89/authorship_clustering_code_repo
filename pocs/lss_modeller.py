# -*- coding: utf-8 -*-
"""

"""
import subprocess as s
import os
from gensim import corpora
from gensim.corpora import bleicorpus
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class LssModeller:
    """A class that handles the representation of documents in a reduced LSS"""

    # Constructor
    def __init__(self, hdp_path: str, input_docs_path: str):
        self.hdp_path = hdp_path
        self.input_docs_path = input_docs_path

    def tokenise_clean_corpus(self, docline: list) -> list:
        cleaned_corpus = [
                [w for w in word_tokenize(d.lower())]
                for d in docline]
        return cleaned_corpus

    def store_docline_as_lda_c(self, docline: list):
        """ Converts a docline corpus to BoW and then to LDA_C and stores it"""
        # Save the LDA_C corpus to disk
        tokenised_corpus = self._tokenise_clean_corpus(docline=docline)
        # Form the word ids dictionary for vectorisation
        dictionary = corpora.Dictionary(tokenised_corpus)
        bow_corpus = [dictionary.doc2bow(t_d) for t_d in tokenised_corpus]
        # Sterialise into LDA_C and store on disk
        save_location = r"{}\lda_c_format\LDA_C.dat".format(self.input_docs_path)
        bleicorpus.serialize(fname=save_location, corpus=bow_corpus)        

    def read_corpus_into_LDA_C(self):
        """Reads the plain documents and save them as LDA-C for hdp"""
        plain_documents = []
        with os.scandir(self.input_docs_path) as docs:
            for doc in docs:
                f = open(doc.path, "r")
                plain_documents.append(f.read())
        return plain_documents

    def invoke_gibbs_hdp(self):
        """Invokes Gibbs hdp posterior inference on the corpus"""
        ret = s.run(
                [r"{}\hdp.exe".format(
                        self.master_docs_path),
                    "--algorithm",
                    "train",
                    "--data",
                    r"{}\data\fullBleiCorpus.dat".format(
                            self.master_docs_path),
                    "--directory",
                    r"{}\results_hdp".format(
                            self.master_docs_path),
                    "--save_lag",
                    "-1"],
                check=True, capture_output=True, text=True)

        return ret.stdout


Modeller = LssModeller(hdp_path=r"..\..\hdps\hdp",
                       input_docs_path=r"..\data\toy_corpus")
# =============================================================================
# def main():
#     print("Main thread started..\n")
#     corpus = fetch_20newsgroups(subset="train")
# 
# 
# if __name__ == "__main__":
#     main()
# =============================================================================
