# -*- coding: utf-8 -*-
"""

"""
import subprocess as s
import os
from gensim.corpora import Dictionary, bleicorpus
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class LssModeller:
    """A class that handles the representation of documents in a reduced LSS"""

    # Constructor
    def __init__(self, hdp_path: str, input_docs_path: str,
                 ldac_filename: str):
        """Default Constructor

        Parameters
        ----------
        self
        hdp_path : str
            The path to where hdp.exe program is
        input_docs_path: str
            The path to where the corpus of files is located

        """

        self.hdp_path = hdp_path
        self.input_docs_path = input_docs_path
        self.lda_c_fname = ldac_filename

    def _read_corpus_into_docline(self) -> list:
        plain_documents = []
        with os.scandir(self.input_docs_path) as docs:
            for doc in docs:
                f = open(doc.path, "r")
                plain_documents.append(f.read())
        return plain_documents

    def _tokenise_clean_corpus(self, docline: list) -> list:
        cleaned_corpus = [
                [w for w in word_tokenize(d.lower())]
                for d in docline]
        return cleaned_corpus

    def _convert_tokenised_docs_to_bow(self, docline: list) -> list:
        tokenised_corpus = self._tokenise_clean_corpus(docline=docline)
        # Form the word ids dictionary for vectorisation
        dictionary = Dictionary(tokenised_corpus)
        bow_corpus = [dictionary.doc2bow(t_d) for t_d in tokenised_corpus]
        return bow_corpus

    def _store_docline_as_lda_c(self, docline: list):
        """ Convert a docline corpus to BoW and then to LDA_C and store it"""
        print("> Converting doclines into LDA-C and storing to disk")
        bow_corpus = self._convert_tokenised_docs_to_bow(docline=docline)
        # Sterialise into LDA_C and store on disk
        output_dir = r"{}\lda_c_format".format(self.input_docs_path)
        if os.path.exists(output_dir):
            os.shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)
        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=bow_corpus)

    def reform_corpus_as_LDA_C(self):
        """Read the plain documents and save them as LDA-C for hdp"""
        print("> Reforming corpus into LDA-C and savinf it as {}.dat".format(
                self.lda_c_fname))
        docline = self._read_corpus_into_docline()
        self._store_docline_as_lda_c(docline=docline)

    def _invoke_gibbs_hdp(self):
        """Invoke Gibbs hdp posterior inference on the corpus"""
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
                       input_docs_path=r"..\data\toy_corpus",
                       ldac_filename="dummy_ldac_corpus")
Modeller.reform_corpus_as_LDA_C()
# =============================================================================
# def main():
#     print("Main thread started..\n")
#     corpus = fetch_20newsgroups(subset="train")
# 
# 
# if __name__ == "__main__":
#     main()
# =============================================================================
