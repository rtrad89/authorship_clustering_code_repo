# -*- coding: utf-8 -*-
"""

"""
import subprocess as s
import os
from shutil import rmtree
from gensim.corpora import Dictionary, bleicorpus
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import pandas as pd


class LssModeller:
    """A class that handles the representation of documents in a reduced LSS"""

    # Constructor
    def __init__(self, hdp_path: str, input_docs_path: str,
                 ldac_filename: str, hdp_output_dir: str):
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
        self.hdp_output_directory = hdp_output_dir

    def _initialise_directory(self, dir_path):
        try:
            if os.path.exists(dir_path):
                rmtree(dir_path)
            os.mkdir(dir_path)
        except PermissionError:
            print("> ERROR: Please make sure the {} folder"
                  "is not used by some process").format(dir_path)

    def _convert_corpus_to_bow(self) -> list:
        """Convert a directory of text files into a BoW model"""
        # Read in the plain text files
        plain_documents = []
        with os.scandir(self.input_docs_path) as docs:
            for doc in docs:
                try:
                    f = open(doc.path, "r")
                    plain_documents.append(f.read())
                except PermissionError:
                    # Raised when trying to open a directory
                    print("\tDirectory {} skipped".format(doc.path))
                    pass

        # Tokenise corpus and remove empty documents
        tokenised_corpus = [
                [w for w in word_tokenize(d.lower())]
                for d in plain_documents if len(d) > 0]

        # Form the word ids dictionary for vectorisation
        dictionary = Dictionary(tokenised_corpus)
        bow_corpus = [dictionary.doc2bow(t_d) for t_d in tokenised_corpus]
        return(bow_corpus, dictionary)

    def _generate_lda_c_corpus(self):
        """ Convert a group of files LDA_C corpus and store it on disk"""
        print("> Converting doclines into LDA-C and storing to disk")
        bow_corpus, id2word_map = self._convert_corpus_to_bow()
        # Sterialise into LDA_C and store on disk
        output_dir = r"{}\lda_c_format".format(self.input_docs_path)
        self._initialise_directory(output_dir)

        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)
        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=bow_corpus,
                id2word=id2word_map)

    def _invoke_gibbs_hdp(self, param_num_iter):
        """Invoke Gibbs hdp posterior inference on the corpus"""
        path_executable = r"{}\hdp.exe".format(self.hdp_path)
        param_data = r"{}\lda_c_format\{}.dat".format(self.input_docs_path,
                                                      self.lda_c_fname)
        param_directory = r"{}\{}".format(self.input_docs_path,
                                          self.hdp_output_directory)
        # Prepare the output directory
        self._initialise_directory(param_directory)

        ret = s.run([path_executable,
                     "--algorithm",     "train",
                     "--data",          param_data,
                     "--directory",     param_directory,
                     "--max_iter",      param_num_iter,
                     "--save_lag",      "-1"],
                    check=True, capture_output=True, text=True)

        return ret.stdout

    def infer_lss_representation(self, param_num_iter="25"):
        """Produce an LSS representation of text files and save it to disk"""
        # Make the text files into an LDA-C corpus
        self._generate_lda_c_corpus()
        # Run Gibbs HDP on the LDA-C corpus
        print("\n> Starting HDP with {} iterations...".format(param_num_iter))
        t = time.perf_counter()
        output_hdp = self._invoke_gibbs_hdp(param_num_iter=param_num_iter)
        nt = time.perf_counter()
        print("************************************************************\n")
        print(("HDP executed in {x:0.2f} seconds "
              "and yielded the message:\n{y}"
               ).format(x=nt-t, y=output_hdp))
        print("**************************************************************")

    def load_lss_representation_into_df(self) -> pd.DataFrame:
        """Load a BoT LSS representation from disk to a returned dataframe"""
        path = r"{}\{}\mode-word-assignments.dat".format(
                self.input_docs_path,
                self.hdp_output_directory)
        # We don't need document tables, so we'll skip the relative column,
        # But we do need word counts under each topic, to produce some sort
        # of a bag-of-topics model (BoT)
        lss_df = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True)
#                             usecols=["d", "w", "z"]).drop_duplicates()
        # Procude topic weights as counts of topic words
        lss_df = pd.DataFrame(
                lss_df.groupby(by=["d", "z"]).w.count()
                ).reset_index()
        return lss_df


Modeller = LssModeller(hdp_path=r"..\..\hdps\hdp",
                       input_docs_path=r"..\data\toy_corpus",
                       ldac_filename=r"dummy_ldac_corpus",
                       hdp_output_dir=r"hdp_lss")
df = Modeller.load_lss_representation_into_df()
# =============================================================================
# def main():
#     print("Main thread started..\n")
#     corpus = fetch_20newsgroups(subset="train")
# 
# 
# if __name__ == "__main__":
#     main()
# =============================================================================
