# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations  # To defer evaluation of type hints
import subprocess as s
import os
from gensim.corpora import Dictionary, bleicorpus
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import time
import pandas as pd
from aiders import AmazonParser, DiskTools
from typing import Tuple  # , overload


class LssHdpModeller:
    """A class that handles the representation of documents in a reduced LSS"""

    # Constructor
    def __init__(self,
                 hdp_path: str,
                 ldac_filename: str,
                 hdp_output_dir: str,
                 hdp_iters: int,
                 hdp_seed: float,
                 word_grams: int,
                 input_docs_path: str = None,
                 input_amazon_path: str = None,
                 input_amazon_fname: str = None):
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
        self.hdp_iterations = hdp_iters
        self.hdp_rand_seed = hdp_seed
        self.word_grams = word_grams
        self.doc_index = []  # the index of the files read for reference

    def _convert_corpus_to_bow(self):
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
                    f = open(doc.path, mode="r", encoding="utf8")
                    plain_documents.append(f.read())
                    self.doc_index.append(DiskTools.get_filename(doc.path))
                except PermissionError:
                    # Raised when trying to open a directory
                    print("Skipping directory while loading files: {}"
                          .format(doc.name))
                    pass
        # Collocation Detection can be applied here via gensim.models.phrases
        # Tokenise corpus and remove too short documents
        tokenised_corpus = [
                [' '.join(tkn) for tkn in
                 ngrams(word_tokenize(d.lower()), self.word_grams)]
                for d in plain_documents if len(d) > 3]

        # Form the word ids dictionary for vectorisation
        dictionary = Dictionary(tokenised_corpus)
        bow_corpus = [dictionary.doc2bow(t_d) for t_d in tokenised_corpus]
        return(bow_corpus,
               dictionary,
               pd.DataFrame(data=plain_documents, index=self.doc_index,
                            columns=["content"]))

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
        return plain_docs, bow_corpus

    def _convert_series_to_bow(self, words_threshold=10):
        """
        Convert a series of texts to BoW represetaion

        Parameters
        ----------
        words_threshold : int
            The minimum amount of words in a document to be included in the
            representation. Documents which contain a smaller amount of words
            shall be excluded.
        """

        # Load the data into a pandas dataframe
        amazon_df = AmazonParser.get_dataframe(r"{}\{}".format(
                self.input_amazon_path, self.input_amazon_filename))

        # Tokenise
        amazon_df["tokenised"] = amazon_df.reviewText.str.lower().apply(
                word_tokenize)
        # Filter out any too short reviews (less than words_threshold)
        amazon_df = amazon_df[amazon_df.tokenised.map(len) >= words_threshold]
        # Construct the word grams
        amazon_df.tokenised = [
                [' '.join(tkn) for tkn in
                 ngrams(r, self.word_grams)]
                for r in amazon_df.tokenised]
        # Establish the dictionary
        dictionary = Dictionary(amazon_df.tokenised)
        amazon_df["bow"] = amazon_df.tokenised.apply(dictionary.doc2bow)
        return (amazon_df, dictionary)

    def _generate_lda_c_from_dataframe(self):
        """
        Convert a dataframe into LDA-C and save it to disk. The dataframe is
        meant to be Amazon's labelled plain data loaded from a gzip file. An
        additional column will be added to hold the BoW representation too.

        Returns
        -------
        amazon_df : pd.DataFrame
            A dataframe holding the plain documents and the BoW representation,
            alongside authors (labels)

        """

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

    def _invoke_gibbs_hdp(self):
        """Invoke Gibbs hdp posterior inference on the corpus"""
        path_executable = r"{}\hdp.exe".format(self.hdp_path)
        param_data = r"{}\lda_c_format\{}.dat".format(self.input_docs_path,
                                                      self.lda_c_fname)
        param_directory = r"{}\{}".format(self.input_docs_path,
                                          self.hdp_output_directory)
        # Prepare the output directory
        DiskTools.initialise_directory(param_directory)

        ret = s.run([path_executable,
                     "--algorithm",     "train",
                     "--data",          param_data,
                     "--directory",     param_directory,
                     "--max_iter",      str(self.hdp_iterations),
                     "--random_seed",   str(self.hdp_rand_seed),
                     "--save_lag",      "-1"],
                    check=True, capture_output=True, text=True)

        return ret.stdout

    def _invoke_gibbs_hdp_amazon(self):
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
                     "--max_iter",      str(self.hdp_iterations),
                     "--random_seed",   str(self.hdp_rand_seed),
                     "--save_lag",      "-1"],
                    check=True, capture_output=True, text=True)

        return ret.stdout

    def _infer_lss_representation(self) -> list:
        """
        Produce an LSS representation of text files and save it to disk

        Returns
        -------
        plain_docs : list
            The original plain documents which were saved to disk as LDA-C.
            They are returned for verification purposes.
        """

        # Make the text files into an LDA-C corpus and return a copy of them
        plain_docs, bow_rep = self._generate_lda_c_corpus()
        # Run Gibbs HDP on the LDA-C corpus
        print("\n> Starting HDP with {} iterations...".format(
                self.hdp_iterations))
        t = time.perf_counter()
        self._invoke_gibbs_hdp()  # To capture the output of hdp, assign a var
        nt = time.perf_counter()
        print("************************************************************")
        print(("HDP executed in {x:0.2f} seconds"
               ).format(x=nt-t))
        print("**************************************************************")
        return plain_docs, bow_rep

    def _infer_amazon_lss_representation(self) -> list:
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
        print("\n> Starting HDP with {} iterations...".format(
                self.hdp_iterations))
        t = time.perf_counter()
        self._invoke_gibbs_hdp_amazon()
        nt = time.perf_counter()
        print("************************************************************\n")
        print(("HDP executed in {x:0.2f} seconds"
               ).format(x=nt-t))
        print("**************************************************************")
        return amazon_df

    def _load_lss_representation_into_df(self) -> pd.DataFrame:
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
        # Index with file names for later reference
        lss_df.index = self.doc_index

        return lss_df

    def _load_amazon_lss_representation_into_df(self) -> pd.DataFrame:
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

    def get_amazon_lss(self, infer_lss) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get high- and low-dimenstional representation of data

        Parameters
        ----------
        infer_lss: bool, optional
            If `True`, hdp will be run again to generate the LSS representation
            on disk. `False` means the representation was already generated and
            can be loaded from disk.

        """

        if infer_lss:
            bow_df = self._infer_amazon_lss_representation()
        else:
            bow_df, _ = self._convert_series_to_bow()

        return bow_df, self._load_amazon_lss_representation_into_df()

    def get_corpus_lss(self, infer_lss) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get high- and low-dimenstional representation of data

        Parameters
        ----------
        infer_lss: bool, optional
            If `True`, hdp will be run again to generate the LSS representation
            on disk. `False` means the representation was already generated and
            can be loaded from disk.

        """

        if infer_lss:
            plain, bow_df = self._infer_lss_representation()
        else:
            bow_df, _, plain = self._convert_corpus_to_bow()

        return plain, bow_df, self._load_lss_representation_into_df()


def main():
    print("Main thread started..\n")
    Modeller = LssHdpModeller(
            hdp_path=r"..\..\hdps\hdp",
            input_docs_path=r"..\..\..\Datasets\pan17_train\problem001",
            input_amazon_path=r"..\..\..\Datasets\Amazon",
            input_amazon_fname=r"reviews_Automotive_5.json.gz",
            ldac_filename=r"dummy_ldac_corpus",
            hdp_output_dir=r"hdp_lss",
            hdp_iters=1000,
            word_grams=1)

#    plain, bow, lss = Modeller.get_corpus_lss(infer_lss=True)


if __name__ == "__main__":
    main()
