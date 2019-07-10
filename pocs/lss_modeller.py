# -*- coding: utf-8 -*-
"""

"""
import subprocess as s
import os
from gensim.corpora import Dictionary, bleicorpus
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import time
import pandas as pd
from aiders import AmazonParser, DiskTools


class LssModeller:
    """A class that handles the representation of documents in a reduced LSS"""

    # Constructor
    def __init__(self, hdp_path: str, input_docs_path: str,
                 input_amazon_path: str, input_amazon_fname: str,
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
        self.input_amazon_path = input_amazon_path
        self.input_amazon_filename = input_amazon_fname
        self.lda_c_fname = ldac_filename
        self.hdp_output_directory = hdp_output_dir

    def convert_corpus_to_bow(self, word_grams=1):
        """
        Convert a directory of text files into a BoW model.

        Parameters
        ----------
        word_grams : int (optional)
            The number of words to combine as features. 1 is the default value,
            and it denotes the usage of word unigrams.

        Returns
        -------
        bow_corpus : gnesim corpus
            The bag-of-words model.

        dictionary : gensim dictionary
            The id2word mapping.

        plain_documents : list
            The list of plain documents, to serve as a reference point.
        """
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
        # Collocation Detection can be applied here via gensim.models.phrases
        # Tokenise corpus and remove too short documents
        tokenised_corpus = [
                [' '.join(tkn) for tkn in
                 ngrams(word_tokenize(d.lower()), word_grams)]
                for d in plain_documents if len(d) > 3]

        # Form the word ids dictionary for vectorisation
        dictionary = Dictionary(tokenised_corpus)
        bow_corpus = [dictionary.doc2bow(t_d) for t_d in tokenised_corpus]
        return(bow_corpus, dictionary, plain_documents)

    def _generate_lda_c_corpus(self):
        """ Convert a group of files LDA_C corpus and store it on disk"""
        bow_corpus, id2word_map, plain_docs = self._convert_corpus_to_bow()
        # Sterialise into LDA_C and store on disk
        output_dir = r"{}\lda_c_format".format(self.input_docs_path)
        DiskTools.initialise_directory(output_dir)
        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)
        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=bow_corpus,
                id2word=id2word_map)
        return plain_docs

    def convert_series_to_bow(self, words_threshold=10, word_grams=4):
        """Convert a series of texts to BoW represetaion"""

        # Load the data into a pandas dataframe
        amazon_df = AmazonParser.get_dataframe(r"{}\{}".format(
                self.input_amazon_path, self.input_amazon_filename))
        # FOR TESTING
        amazon_df = amazon_df.tail(50)
        # Tokenise
        amazon_df["tokenised"] = amazon_df.reviewText.str.lower().apply(
                word_tokenize)
        # Filter out any too short reviews (less than words_threshold)
        amazon_df = amazon_df[amazon_df.tokenised.map(len) >= words_threshold]
        # Construct the word grams
        amazon_df.tokenised = [
                [' '.join(tkn) for tkn in
                 ngrams(r, word_grams)]
                for r in amazon_df.tokenised]
        # Establish the dictionary
        dictionary = Dictionary(amazon_df.tokenised)
        amazon_df["bow"] = amazon_df.tokenised.apply(dictionary.doc2bow)
        return (amazon_df, dictionary)

    def _generate_lda_c_from_dataframe(self):
        amazon_df, id2word_map = self._convert_series_to_bow()
        # Sterialise it to disk as LDA-C
        output_dir = r"{}\lda_c_format".format(self.input_amazon_path)
        DiskTools.initialise_directory(output_dir)
        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)
        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=amazon_df.bow,
                id2word=id2word_map)
        return amazon_df

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

    def _invoke_gibbs_hdp_amazon(self, param_num_iter):
        """Invoke Gibbs hdp posterior inference on the corpus"""
        path_executable = r"{}\hdp.exe".format(self.hdp_path)
        param_data = r"{}\lda_c_format\{}.dat".format(
                self.input_amazon_path,
                self.lda_c_fname)
        param_directory = r"{}\{}".format(self.input_amazon_path,
                                          self.hdp_output_directory)
        # Prepare the output directory
        DiskTools.initialise_directory(param_directory)

        ret = s.run([path_executable,
                     "--algorithm",     "train",
                     "--data",          param_data,
                     "--directory",     param_directory,
                     "--max_iter",      param_num_iter,
                     "--save_lag",      "-1"],
                    check=True, capture_output=True, text=True)

        return ret.stdout

    def infer_lss_representation(self, param_num_iter="25") -> list:
        """
        Produce an LSS representation of text files and save it to disk

        Returns
        -------
        plain_docs : list
            The original plain documents which were saved to disk as LDA-C.
            They are returned for verification purposes.
        """

        # Make the text files into an LDA-C corpus and return a copy of them
        plain_docs = self._generate_lda_c_corpus()
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
        return plain_docs

    def infer_amazon_lss_representation(self, param_num_iter="25") -> list:
        """
        Produce an LSS representation of text files and save it to disk

        Returns
        -------
        plain_docs : list
            The original plain documents which were saved to disk as LDA-C.
            They are returned for verification purposes.
        """

        # Make the text files into an LDA-C corpus and return a copy of them
        amazon_df = self._generate_lda_c_from_dataframe()
        # Run Gibbs HDP on the LDA-C corpus
        print("\n> Starting HDP with {} iterations...".format(param_num_iter))
        t = time.perf_counter()
        output_hdp = self._invoke_gibbs_hdp_amazon(
                param_num_iter=param_num_iter)
        nt = time.perf_counter()
        print("************************************************************\n")
        print(("HDP executed in {x:0.2f} seconds "
              "and yielded the message:\n{y}"
               ).format(x=nt-t, y=output_hdp))
        print("**************************************************************")
        return amazon_df

    def load_lss_representation_into_df(self) -> pd.DataFrame:
        """
        Load a BoT LSS representation from disk to a returned dataframe.

        Returns
        -------
        lss_df : pd.DataFrame
            A matrix of shape (n_samples, n_features)

        """

        path = r"{}\{}\mode-word-assignments.dat".format(
                self.input_docs_path,
                self.hdp_output_directory)
        # We don't need document tables, so we'll skip the relative column,
        # But we do need word counts under each topic, to produce some sort
        # of a bag-of-topics model (BoT)
        lss_df = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True)
#                             usecols=["d", "w", "z"]).drop_duplicates()
        # Produce topic weights as counts of topic words
        lss_df = lss_df.pivot_table(
                values='w', columns='z', index='d',
                aggfunc='count', fill_value=0)
        return lss_df

    def load_amazon_lss_representation_into_df(self) -> pd.DataFrame:
        """
        Load a BoT LSS representation from disk to a returned dataframe.

        Returns
        -------
        lss_df : pd.DataFrame
            A matrix of shape (n_samples, n_features)

        """

        path = r"{}\{}\mode-word-assignments.dat".format(
                self.input_amazon_path,
                self.hdp_output_directory)
        # We don't need document tables, so we'll skip the relative column,
        # But we do need word counts under each topic, to produce some sort
        # of a bag-of-topics model (BoT)
        lss_df = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True)
#                             usecols=["d", "w", "z"]).drop_duplicates()
        # Produce topic weights as counts of topic words
        lss_df = lss_df.pivot_table(
                values='w', columns='z', index='d',
                aggfunc='count', fill_value=0)
        return lss_df


Modeller = LssModeller(hdp_path=r"..\..\hdps\hdp",
                       input_docs_path=r"..\data\toy_corpus",
                       input_amazon_path=r"..\..\..\Datasets\Amazon",
                       input_amazon_fname=r"reviews_Automotive_5.json.gz",
                       ldac_filename=r"dummy_ldac_corpus",
                       hdp_output_dir=r"hdp_lss")

# =============================================================================
# def main():
#     print("Main thread started..\n")

# if __name__ == "__main__":
#     main()
# =============================================================================
