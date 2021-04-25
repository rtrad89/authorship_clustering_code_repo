# -*- coding: utf-8 -*-
"""
LSSR modeller controller, which encapsulates all routines to represent a corpus of text files as LSSR via HDP.

"""
from __future__ import annotations  # To defer evaluation of type hints
import subprocess as s
from gensim.corpora import Dictionary, bleicorpus
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import time
import pandas as pd
from itertools import product
from src.aiders import Tools
from typing import Tuple, List
from collections import defaultdict
import seaborn as sns
# from btm import indexDocs
from langdetect import detect
from re import sub
# from scipy.special import comb
sns.set()


class LssHdpModeller:
    """A class that handles the representation of documents in a reduced LSS"""

    # Constructor
    def __init__(self,
                 hdp_path: str,
                 ldac_filename: str,
                 hdp_output_dir: str,
                 hdp_iters: int,
                 hdp_sample_hyper: bool,
                 hdp_seed: int,
                 word_grams: int,
                 drop_uncommon: bool = False,
                 freq_threshold: int = 0,
                 hdp_eta: float = 0.5,
                 hdp_gamma_s: float = 1.0,
                 hdp_alpha_s: float = 1.0,
                 input_docs_path: str = None,
                 verbose: bool = False):
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
        self.hdp_output_directory = (f"{hdp_output_dir}_{hdp_eta:0.2f}"
                                     f"_{hdp_gamma_s:0.2f}_{hdp_alpha_s:0.2f}"
                                     f"_common_{drop_uncommon}")
        self.hdp_iterations = hdp_iters
        self.hdp_seed = hdp_seed
        self.hdp_hyper_sampling = hdp_sample_hyper
        self.hdp_eta = hdp_eta
        self.hdp_gamma_s = hdp_gamma_s
        self.hdp_alpha_s = hdp_alpha_s
        self.word_grams = word_grams
        self.drop_uncommon = drop_uncommon
        self.freq_th = freq_threshold
        self.doc_index = []  # the index of the files read for reference
        self.verbose = verbose

    def _convert_corpus_to_bow(self,
                               file_ext: str = "txt"):
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
        with Tools.scan_directory(self.input_docs_path) as docs:
            for doc in docs:
                if doc.is_dir() or Tools.split_path(doc.path
                                                    )[1] != f".{file_ext}":
                    continue
                try:
                    f = open(doc.path, mode="r", encoding="utf8")
                    plain_documents.append(f.read())
                    self.doc_index.append(Tools.get_filename(doc.path))
                except PermissionError:
                    # Raised when trying to open a directory
                    print("Skipped while loading files: {}"
                          .format(doc.name))
                    pass
        # Collocation Detection can be applied here via gensim.models.phrases
        # Tokenise corpus and remove too short documents
        tokenised_corpus = [
                [' '.join(tkn) for tkn in
                 ngrams(word_tokenize(d.lower()), self.word_grams)]
                for d in plain_documents if len(d) > 3]

        if self.drop_uncommon:
            freq = defaultdict(int)
            for doc in tokenised_corpus:
                for word in doc:
                    freq[word] += 1
            tokenised_corpus = [
                    [w for w in doc if freq[w] > self.freq_th]
                    for doc in tokenised_corpus]
        # Form the word ids dictionary for vectorisation
        dictionary = Dictionary(tokenised_corpus)
        corpus = [dictionary.doc2bow(t_d) for t_d in tokenised_corpus]

        return(corpus,
               dictionary,
               pd.DataFrame(data=plain_documents, index=self.doc_index,
                            columns=["content"]))

    def _generate_lda_c_corpus(self):
        """ Convert a group of files LDA_C corpus and store it on disk"""
        bow_corpus, id2word_map, plain_docs = self._convert_corpus_to_bow()
        # Sterialise into LDA_C and store on disk
        output_dir = (r"{}\lda_c_format_{:0.1f}_{:0.1f}"
                      r"_{:0.1f}_common_{}").format(self.input_docs_path,
                                                    self.hdp_eta,
                                                    self.hdp_gamma_s,
                                                    self.hdp_alpha_s,
                                                    self.drop_uncommon)
        Tools.initialise_directory(output_dir)
        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)

        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=bow_corpus,
                id2word=id2word_map)
        return plain_docs, bow_corpus

    def _invoke_gibbs_hdp(self):
        """Invoke Gibbs hdp posterior inference on the corpus"""
        path_executable = r"{}\hdp.exe".format(self.hdp_path)
        param_data = (r"{}\lda_c_format_{:0.1f}_{:0.1f}"
                      r"_{:0.1f}_common_{}\{}.dat").format(
                self.input_docs_path,
                self.hdp_eta,
                self.hdp_gamma_s,
                self.hdp_alpha_s,
                self.drop_uncommon,
                self.lda_c_fname)

        param_directory = r"{}\{}".format(self.input_docs_path,
                                          self.hdp_output_directory)
        # Prepare the output directory
        Tools.initialise_directory(param_directory)

        if self.hdp_seed is not None and self.hdp_seed > 0:
            ret = s.run([path_executable,
                         "--algorithm",     "train",
                         "--data",          param_data,
                         "--directory",     param_directory,
                         "--max_iter",      str(self.hdp_iterations),
                         "--sample_hyper",  "yes" if self.hdp_hyper_sampling
                         else "no",
                         "--save_lag",      "-1",
                         "--eta",           str(self.hdp_eta),
                         "--random_seed",   str(self.hdp_seed),
                         "--gamma_a",     str(self.hdp_gamma_s),
                         "--alpha_a",     str(self.hdp_alpha_s)],
                        check=True, capture_output=True, text=True)
        else:
            ret = s.run([path_executable,
                         "--algorithm",     "train",
                         "--data",          param_data,
                         "--directory",     param_directory,
                         "--max_iter",      str(self.hdp_iterations),
                         "--sample_hyper",  "yes" if self.hdp_hyper_sampling
                         else "no",
                         "--save_lag",      "-1",
                         "--eta",           str(self.hdp_eta),
                         "--gamma_a",     str(self.hdp_gamma_s),
                         "--alpha_a",     str(self.hdp_alpha_s)],
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
        msg = self._invoke_gibbs_hdp()
        nt = time.perf_counter()
        print("***************************************")
        print(("HDP executed in {x:0.2f} seconds"
               ).format(x=nt-t))
        if self.verbose:
            print(msg)
        print("***************************************")
        return plain_docs, bow_rep

    def _load_lss_representation_into_df(self) -> pd.DataFrame:
        """
        Load a BoT LSS representation from disk to a returned dataframe.

        Returns
        -------
        lss_df : pd.DataFrame
            A matrix of shape (n_samples, n_features)

        Raises
        ------
        FileNotFoundError
            When the LSS representation isn't found on disk.

        """

        path = r"{}\{}\mode-word-assignments.dat".format(
                self.input_docs_path,
                self.hdp_output_directory)
        # We don't need document tables, so we'll skip the relative column,
        # But we do need word counts under each topic, to produce some sort
        # of a bag-of-topics model (BoT)
        try:
            lss_df = pd.read_csv(filepath_or_buffer=path,
                                 delim_whitespace=True)
    #                             usecols=["d", "w", "z"]).drop_duplicates()
            # Produce topic weights as counts of topic words
            lss_df = lss_df.pivot_table(
                    values='w', columns='z', index='d',
                    aggfunc='count', fill_value=0)
            # Index with file names for later reference
            lss_df.index = self.doc_index

            return lss_df
        except FileNotFoundError:
            print(("\nNo LSS precomputed file was found on disk via:\n{}\n"
                  "> Please generate LDA-C corpus and run HDP first...\n"
                   ).format(path))
            raise

    def _convert_to_gensim_corpus(self,
                                  lss: pd.DataFrame
                                  ) -> pd.DataFrame:
        # A topic is a term, and the weight is the count
        pass

    def get_corpus_lss(self, infer_lss,
                       bim_thresold: int = 0,
                       bim: bool = False
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get high- and low-dimenstional representations of data.

        Parameters
        ----------
        infer_lss: bool, optional
            If `True`, hdp will be run again to generate the LSS representation
            on disk. `False` means the representation was already generated and
            can be loaded from disk.

        Returns
        -------
        plain : DataFrame
            The plain documents which were processed.
        bow_df : DataFrame
            BoW representation of documents.
        DataFrame
            The LSS representation of the documents.

        Raises
        ------
        FileNotFoundError
            Raised when infer_lss = False and there is no precomputed LSS
            representation on disk.

        """

        if infer_lss:
            plain, bow_df = self._infer_lss_representation()
        else:
            bow_df, _, plain = self._convert_corpus_to_bow()

        lss = self._load_lss_representation_into_df()
        # Convert the BoT to BIM:
        if bim:
            if bim_thresold is None:
                bim_thresold = pd.np.nanquantile(lss[lss > 0], .1)
            lss = (lss > bim_thresold).astype(int)

        return plain, bow_df, lss


class LssOptimiser:
    """
    A class for eta and concentration parameters optimisation across all
    training data
    """

    def __init__(self,
                 train_folders_path: str,
                 hdp_path: str,
                 ldac_filename: str,
                 hdp_seed: int,
                 eta_range: List[float],
                 gamma_range: List[float],
                 alpha_range: List[float],
                 out_dir: str,
                 hdp_iters: int = 1000
                 ):
        self.training_folder = train_folders_path
        self.hdp_path = hdp_path
        self.ldac = ldac_filename
        self.seed = hdp_seed
        self.etas = eta_range
        self.gammas = gamma_range
        self.alphas = alpha_range
        self.out_dir = out_dir
        self.iters = hdp_iters

    def _get_number_words(self, vocab_filepath: str):
        """Return the number of lines in the vocab file (→ unique words)

        Returns
        -------
        int
            The number of unique words in an lda-c corpus.
        """

        with open(vocab_filepath, encoding="utf8") as f:
            for i, l in enumerate(f):
                pass
        return i+1

    def assess_hyper_sampling(self, tail_prcnt: float,
                              verbose: bool = False):
        """
        A function to measure the average per word log-likelihood after
        hyper-sampling the concentration parameters of the Dirichlet
        distributions.
        Caution: the hdp must have been run on the data with hyper sampling and
        without it, in order to load the two representations and compare.

        Returns
        -------
        dct: dict
            A dictionary containing the per word log-likelihood of the train
            data with the two methods pertaining to sampling the concentration
            parameters: normal and hyper.

        """
        path_normal = r"/hdp_lss_HyperFalse/state.log"
        path_hyper = r"/hdp_lss_HyperTrue/state.log"
        path_ldac = (r"/lda_c_format_HyperTrue"
                     r"/dummy_ldac_corpus.dat.vocab")
        per_word_ll_normal = []
        per_word_ll_hyper = []

        if verbose:
            print("------Concentration Parameters Optimisation------")

        with Tools.scan_directory(self.training_folder) as dirs:
            for d in dirs:
                if d.name[0:7] != "problem":
                    continue

                if verbose:
                    print(f"\t► Processing {d.name}")

                normal = f"{d.path}\\{path_normal}"
                hyper = f"{d.path}\\{path_hyper}"
                vocab = f"{d.path}\\{path_ldac}"

                n_words = self._get_number_words(vocab)
                df_normal = pd.read_csv(filepath_or_buffer=normal,
                                        delim_whitespace=True,
                                        index_col="iter",
                                        usecols=["iter", "likelihood"],
                                        squeeze=True)
                ll_normal = df_normal.tail(round(len(df_normal) * tail_prcnt
                                                 )).mean()
                per_word_ll_normal.append(ll_normal / n_words)

                df_hyper = pd.read_csv(filepath_or_buffer=hyper,
                                       delim_whitespace=True,
                                       index_col="iter",
                                       usecols=["iter", "likelihood"],
                                       squeeze=True)
                ll_hyper = df_hyper.tail(round(len(df_hyper) * tail_prcnt
                                               )).mean()
                per_word_ll_hyper.append(ll_hyper / n_words)

        dct = {"Normal_Sampling":
               round(sum(per_word_ll_normal) / len(per_word_ll_normal), 4),
               "Hyper_Sampling":
               round(sum(per_word_ll_hyper) / len(per_word_ll_hyper), 4)}

        if verbose:
            print("-------------------------------------------------")

        pd.DataFrame(data=dct, index=[0]).to_csv(
                f"{self.out_dir}/hyper_optimisation.csv", index=False)
        return dct

    def _generate_hdps_outputs(self,
                               skip_factor: int = 1,
                               verbose: bool = False):
        st = time.perf_counter()
        ldac_path = r"lda_c_format_HyperFalse\\dummy_ldac_corpus.dat"
        words_nums = {}
        vocab_file = r"lda_c_format_HyperFalse\\dummy_ldac_corpus.dat.vocab"
#        size = ((60 // skip_factor)
#                * len(self.etas)
#                * len(self.gammas)**2
#                * len(self.alphas)**2)
        # Since we fixed the scales of Gammas
        size = ((60 // skip_factor)
                * len(self.etas)
                * len(self.gammas)
                * len(self.alphas))
        i = 0
        with Tools.scan_directory(self.training_folder) as ps_folders:
            for c, folder in enumerate(ps_folders):
                if not folder.name[0:7] == "problem":
                    if verbose:
                        print(f"→ Skipping {folder.name}")
                    continue
                # Implement the skipping factor
                if c % skip_factor != 0:
                    continue

                t = time.perf_counter()
                # Fix the scale parameters for the Gamma priors
                g_r = 1
                a_r = 1
                for eta in self.etas:
                    # for g_s, g_r in product(self.gammas, repeat=2):
                    # for a_s, a_r in product(self.alphas, repeat=2):
                    # Only switch the shape parameter of Gammas
                    for g_s in self.gammas:
                        for a_s in self.alphas:
                            # Cache the number of words for later
                            if folder.name not in words_nums:
                                vocab_path = f"{folder.path}\\{vocab_file}"
                                n_words = self._get_number_words(vocab_path)
                                words_nums.update({folder.name: n_words})

                            i = i + 1
                            percentage = f"{100 * i / size:06.02f}"
                            suff = (f"{g_s:0.2f}_{g_r:0.2f}_"
                                    f"{a_s:0.2f}_{a_r:0.2f}")
                            if verbose:
                                print(f"► Applying HDP with "
                                      f"eta={eta:0.1f} "
                                      f"gamma({g_s:0.2f}, {g_r:0.2f}) "
                                      f"alpha({a_s:0.2f}, {a_r:0.2f}) "
                                      f"on {folder.name} [{percentage}%]")

                            directory = (f"{self.out_dir}/optimisation"
                                         f"/{eta:0.1f}__{suff}"
                                         f"/{folder.name}")

                            if (Tools.path_exists(directory)):
                                if verbose:
                                    print("\tcached result found at "
                                          f"{directory}")
                                continue

                            path_executable = r"{}\hdp.exe".format(
                                    self.hdp_path)
                            data = f"{folder.path}/{ldac_path}"

                            # Prepare the output directory
                            Tools.initialise_directories(directory)

                            if self.seed is not None:
                                s.run([path_executable,
                                       "--algorithm",     "train",
                                       "--data",          data,
                                       "--directory",     directory,
                                       "--max_iter",      str(self.iters),
                                       "--sample_hyper",  "no",
                                       "--save_lag",      "-1",
                                       "--eta",           str(eta),
                                       "--gamma_a",     str(g_s),
                                       "--gamma_b",     str(g_r),
                                       "--alpha_a",     str(a_s),
                                       "--alpha_b",     str(a_r),
                                       "--random_seed",   str(self.seed)],
                                      stdout=s.DEVNULL,
                                      check=True,
                                      capture_output=False,
                                      text=True)
                            else:
                                s.run([path_executable,
                                       "--algorithm",     "train",
                                       "--data",          data,
                                       "--directory",     directory,
                                       "--max_iter",      str(self.iters),
                                       "--sample_hyper",  "no",
                                       "--save_lag",      "-1",
                                       "--eta",           str(eta),
                                       "--gamma_a",     str(g_s),
                                       "--gamma_b",     str(g_r),
                                       "--alpha_a",     str(a_s),
                                       "--alpha_b",     str(a_r)],
                                      stdout=s.DEVNULL,
                                      check=True,
                                      capture_output=False,
                                      text=True)

                if verbose:
                    print(f"--- {folder.name} done in "
                          f"{time.perf_counter() - t:0.1f} seconds ---")

        period = round(time.perf_counter() - st, 2)
        print(f"▬▬▬▬▬ Vectorisation done in {period} seconds ▬▬▬▬▬")
        return words_nums

    def smart_optimisation(self,
                           plot_cat: str = "likelihood",
                           tail_prcnt: float = 0.80,
                           skip_factor: int = 1,
                           verbose: bool = False):
        # First generate the outputs to compare:
        words_counts = self._generate_hdps_outputs(skip_factor=skip_factor,
                                                   verbose=verbose)

        ret = {}
        # Loop over the outputs of different etas
        master_folder = (f"{self.out_dir}\\optimisation")
        log_likelihoods = []
        avg_num_topics = []
        std_num_topics = []
        pw_ll = []
        errors = []
        with Tools.scan_directory(master_folder) as perms:
            for perm in perms:
                # generate plots
                if not Tools.is_path_dir(perm.path):
                    continue

                self.generate_gibbs_states_plots(states_path=perm.path,
                                                 cat=plot_cat)
                with Tools.scan_directory(perm.path) as problems:
                    for problem in problems:
                        try:
                            n_words = words_counts[problem.name]
                            path_state = f"{problem.path}\\state.log"
                            df_state = pd.read_csv(
                                    filepath_or_buffer=path_state,
                                    delim_whitespace=True,
                                    index_col="iter",
                                    usecols=["iter", "likelihood",
                                             "num.topics"]
                                    )
                            ll = df_state.likelihood.tail(round(
                                    len(df_state) * tail_prcnt
                                    )).mean()
                            avg_topics = df_state["num.topics"].tail(round(
                                    len(df_state) * tail_prcnt
                                    )).mean()
                            std_topics = df_state["num.topics"].tail(round(
                                    len(df_state) * tail_prcnt
                                    )).std()

                            log_likelihoods.append(ll)
                            pw_ll.append(ll / n_words)
                            avg_num_topics.append(avg_topics)
                            std_num_topics.append(std_topics)
                        except FileNotFoundError as e:
                            print(f"{e}")
                            errors.append(f"{e}")
                            continue
                        except KeyError:
                            # Plots folders are being queried for n_words
                            continue
                ret.update({f"{perm.name}":
                            [round(sum(log_likelihoods) / len(log_likelihoods),
                                   4),
                             round(sum(pw_ll) / len(pw_ll),
                                   4),
                             round(sum(avg_num_topics) / len(avg_num_topics),
                                   4),
                             round(sum(std_num_topics) / len(std_num_topics),
                                   4)
                             ]
                            })
        # Save any encountered errors to disk too
        Tools.save_list_to_text(mylist=errors,
                                filepath=(f"{self.out_dir}\\optimisation"
                                          f"\\opt_errors.txt"))

        pd.DataFrame(data=ret,
                     index=["Log-l", "PwLL", "T-Avg", "T-Std"]
                     ).T.to_csv(
                             f"{self.out_dir}\\optimisation\\optimisation.csv",
                             index=True)

        return ret

    def traverse_gamma_alpha(self,
                             ps: int,
                             tail_prcnt: float = 0.80,
                             verbose: bool = True):
        ldac_path = r"lda_c_format_HyperFalse\dummy_ldac_corpus.dat"
        dat_path = f"{self.training_folder}\\problem{ps:03d}\\{ldac_path}"
        directory = f"{self.out_dir}\\gamma_alpha"
        path_executable = r"{}\hdp.exe".format(self.hdp_path)

        res = defaultdict(list)
        total_work = len(self.gammas)**2 * len(self.alphas)**2
        c = 0
        print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
        for g_s, g_r in product(self.gammas, repeat=2):
            for a_s, a_r in product(self.alphas, repeat=2):
                for a_r in self.alphas:
                    c = c + 1
                    progress = 100.0 * c / total_work
                    suff = f"_{g_s:0.2f}_{g_r:0.2f}_{a_s:0.2f}_{a_r:0.2f}"
                    if verbose:
                        print(f"► Working on "
                              f"Gamma({g_s:0.2f},{g_r:0.2f}) "
                              f"and Alpha({a_s:0.2f},{a_r:0.2f}) "
                              f"[{progress:06.2f}%]")
                    s.run([path_executable,
                           "--algorithm",     "train",
                           "--data",          dat_path,
                           "--directory",   (f"{directory}\\{c:03d}"
                                             f"hdp_out{suff}"),
                           "--max_iter",      str(500),
                           "--sample_hyper",  "no",
                           "--save_lag",      "-1",
                           "--eta",           "0.5",
                           "--random_seed",   str(self.seed),
                           "--gamma_a",     str(g_s),
                           "--gamma_b",     str(g_r),
                           "--alpha_a",     str(a_s),
                           "--alpha_b",     str(a_r)],
                          check=True, capture_output=True, text=True)
                    # Read the likelihood
                    ll = pd.read_csv(
                            (f"{directory}\\{c:03d}hdp_out{suff}"
                             f"\\state.log"),
                            delim_whitespace=True
                            ).likelihood.tail(round(tail_prcnt * 500)
                                              ).mean()
                    res["gamma_shape"].append(g_s)
                    res["gamma_rate"].append(g_r)
                    res["alpha_shape"].append(a_s)
                    res["alpha_rate"].append(a_r)
                    res["gamma"].append(g_s * g_r)
                    res["alpha"].append(a_s * a_r)
                    res["likelihood"].append(ll)
        # Save the results to disk
        df_res = pd.DataFrame(res)
        df_res.to_csv(f"{directory}\\results.csv", index=False)
        if verbose:
            print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ Done ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
#        Tools.remove_directory(f"{directory}\\hdp_out")
        return df_res

    def generate_gibbs_states_plots(self,
                                    states_path: str,
                                    cat: str = "likelihood"):
        new_dir = f"{states_path}\\{cat}_plots"
        if Tools.path_exists(new_dir):
            print("Plots found, skipping..")
            return

        Tools.initialise_directory(new_dir)
        with Tools.scan_directory(states_path) as outputs:
            for i, output in enumerate(outputs):
                try:
                    state_file = f"{output.path}\\state.log"
                    df = pd.read_csv(filepath_or_buffer=state_file,
                                     delim_whitespace=True,
                                     index_col="iter")
                    ax = sns.lineplot(x=df.index, y=cat, data=df)
                    ax.margins(x=0)
                    name = output.name
                    fig = ax.get_figure()
                    fig.savefig(
                            f"{states_path}\\{cat}_plots\\{name}.png",
                            dpi=300,
                            bbox_incehs="tight",
                            format="png")
                    fig.clf()
                    print(f"{i}")
                except FileNotFoundError:
                    print(f"→ Skipping {output.name}")


class LssBTModeller:

    def __init__(self,
                 directory_path: str,
                 t: int,
                 alpha: float,
                 beta: float,
                 btm_exe_path: str = r"..\BTM-master\src\btm.exe",
                 n_iter: int = 10000,  # To guarantee convergence
                 model_dir_suffix: str = "",
                 doc_inference_type: str = "sum_b"
                 ):
        self.directory_path = directory_path
        self.t = t
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.doc_index = []  # the index of the files read for reference
        self.w = None
        self.btm_exe = btm_exe_path
        self.doc_inf_type = "sum_b"  # Due to later dependant computations

        self.output_dir = f"{directory_path}\\BTM_{model_dir_suffix}"
        self.plain_corpus_path = f"{self.output_dir}\\btmcorpus.txt"
        self.tokenised_btmcorpus_filepath = (f"{self.output_dir}\\vectorised\\"
                                             "tokenised_btmcorpus.txt")
        self.vocab_ids_path = f"{self.output_dir}\\vectorised\\voca_pt"

    def _concatenate_docs_into_btmcorpus(self,
                                         remove_bgw: bool = False,
                                         drop_uncommon: bool = False,
                                         drop_punctuation: bool = False):
        # Read in the plain text files
        plain_documents = []
        with Tools.scan_directory(self.directory_path) as docs:
            for doc in docs:
                if doc.is_dir():
                    continue
                try:
                    f = open(doc.path, mode="r", encoding="utf8")
                    plain_documents.append(f.read())
                    self.doc_index.append(Tools.get_filename(doc.path))
                except PermissionError:
                    # Raised when trying to open a directory
                    print("Skipped while loading files: {}"
                          .format(doc.name))
                    pass
                finally:
                    f.close()
        # lowercase and strip \n away
        plain_documents = [str.replace(d, "\n", "").lower()
                           for d in plain_documents]
        # it was observed that the topics are composed of a lot of stop words
        # Following the BTM paper and the observation, we remove these
        if remove_bgw:
            # Detect the language
            lang = detect(" ".join(plain_documents))
            if lang == "en":
                lang = "english"
            elif lang == "nl":
                lang = "dutch"
            else:
                lang = "greek"

            new_documents = []
            for d in plain_documents:
                terms = [w for w in word_tokenize(text=d, language=lang)
                         if w not in set(stopwords.words(lang))]
                new_documents.append(" ".join(terms))
            plain_documents = new_documents

        if drop_punctuation:
            plain_documents = [sub(pattern=r"[^\w\s]",
                                   repl="",
                                   string=d) for d in plain_documents]
        # save it to disk
        Tools.save_list_to_text(mylist=plain_documents,
                                filepath=self.plain_corpus_path)
        return plain_documents

    def _vectorise_btmcorpus(self):
        # we call routines from indexDocs.py
        indexDocs.indexFile(self.plain_corpus_path,
                            self.tokenised_btmcorpus_filepath)
        indexDocs.write_w2id(self.vocab_ids_path)
        # Assign the number of words to the BTM object
        f = open(self.vocab_ids_path, mode="r", encoding="utf8")
        temp = f.readlines()
        self.w = len(temp)
        f.close()

    def _estimate_btm(self):
        """Invoke Gibbs BTM posterior inference on the tokenised corpus"""

        ret = s.run([self.btm_exe,
                     "est",
                     str(self.t),
                     str(self.w),
                     str(self.alpha),
                     str(self.beta),
                     str(self.n_iter),
                     str(self.n_iter),  # Save Step
                     self.tokenised_btmcorpus_filepath,
                     f"{self.output_dir}\\"
                     ],
                    check=True, capture_output=True, text=True)
        return ret.stdout

    def _infer_btm_pz_d(self):
        """Invoke Gibbs BTM docs inference on the corpus"""

        ret = s.run([self.btm_exe,
                     "inf",
                     self.doc_inf_type,
                     str(self.t),
                     self.tokenised_btmcorpus_filepath,
                     f"{self.output_dir}\\"
                     ],
                    check=True, capture_output=True, text=True)
        return ret.stdout

    def infer_btm(self,
                  remove_bg_terms: bool,
                  drop_uncommon_terms: bool = False,
                  drop_puncs: bool = False,
                  use_biterm_freqs: bool = False):
        self._concatenate_docs_into_btmcorpus(
            remove_bgw=remove_bg_terms,
            drop_uncommon=drop_uncommon_terms,
            drop_punctuation=drop_puncs)
        self._vectorise_btmcorpus()
        self._estimate_btm()
        self._infer_btm_pz_d(use_frequencies=use_biterm_freqs)

    def _doc_gen_biterms(doc: List, window: int = 15) -> List:
        """
        Replicate the generation of terms by the original C++ implementation
        Link: https://github.com/xiaohuiyan/BTM/blob/master/src/doc.h#L35

        Parameters
        ----------
        doc : List
            The tokenised document to generate the biterms from.
        window : int, optional
            The window whereby biterms are elicited. The default is 15.

        Returns
        -------
        List
            The generated list of biterms.

        """
        biterms = []
        if len(doc) < 2:
            return None
        for i, term in enumerate(doc):
            for j in range(i+1, min(i+window, len(doc))):
                # !!! redundancy kept as per the C++ code
                biterms.append((doc[i], doc[j]))

        return biterms

    def load_pz_d_into_df(self,
                          use_frequencies: bool = False):
        """


        Parameters
        ----------
        use_frequencies : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        btm_lss : TYPE
            DESCRIPTION.

        """
        # ??? This function is not used, should be used in tester._vectorise_ps
        # Load the lss into df
        pzd_fpath = f"{self.directory_path}k{self.t}.pz_d"
        try:
            btm_lss = pd.read_csv(filepath_or_buffer=pzd_fpath,
                                  delim_whitespace=True)

            if not self.doc_index:
                # We will need to build the index
                with Tools.scan_directory(self.directory_path) as docs:
                    for doc in docs:
                        if doc.is_dir():
                            continue
                        self.doc_index.append(Tools.get_filename(doc.path))
            btm_lss.index = self.doc_index

            if use_frequencies:
                # The saved documents are in p(z|d) values
                # We want to proportion them to frequencies so that we have the
                # frequency of terms belonging to a topic
                # Since sum_b is used, we will use the count of biterms
                # Treating each p(zi|dj) as a proportion, we will count biterms
                with open(self.tokenised_btmcorpus_filepath) as c:
                    tcorpus = c.readlines()
                # How many biterms are there?
                # Analyzing the C++ code, a widnow of 15 is used
                # regenerate the biterms and count as statistics can detect
                # redundancies in unordered terms:
                freqs = [len(self._doc_gen_biterms(tdoc))
                         for tdoc in tcorpus]
                btm_lss = btm_lss.mul(freqs, axis="index")

            return btm_lss
        except FileNotFoundError:
            return None


def main():
    # Specify which topic model to use?
    use_btm = True

    if use_btm:
        #   Control Parameters ###
        train_phase = True
        t = 10  # number of btm topics
        ##########################

        print("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
        print("BTM modelling and authorial clustering")
        print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")

        if train_phase:
            r = range(1, 2)
            dpath = (r"D:\Projects\Authorial_Clustering_Short_Texts_nPTM"
                     r"\Datasets\pan17_train")
        else:
            r = range(1, 121)
            dpath = (r"D:\Projects\Authorial_Clustering_Short_Texts_nPTM"
                     r"\Datasets\pan17_test")

        for ps in r:
            # Loop over the problemsets
            ps_path = f"{dpath}\\problem{ps:03d}"
            print(f"\nProcessing #{ps:03d}:")
            #   Inferring BTM ###
            #####################
            # TODO: avoid creating r BTM objects by delegating ps_path
            btm = LssBTModeller(directory_path=ps_path,
                                t=t,
                                alpha=1.0,
                                beta=0.01,
                                model_dir_suffix="remove_stopwords_puncts")
            btm.infer_btm(remove_bg_terms=True,
                          drop_puncs=True,
                          use_biterm_freqs=False)
            print("\t→ btm inference done")
    else:

        print("Main thread started..\n")
        folders_path = (r"D:\College\DKEM\Thesis"
                        r"\AuthorshipClustering\Datasets\pan17_train")
        hdp = r"D:\College\DKEM\Thesis\AuthorshipClustering\Code\hdps\hdp"

        optimiser = LssOptimiser(train_folders_path=folders_path,
                                 hdp_path=hdp,
                                 ldac_filename="dummy_ldac_corpus.dat",
                                 hdp_seed=None,
                                 eta_range=[0.3, 0.5, 0.8, 1],
                                 gamma_range=[0.1, 0.3, 0.5],
                                 alpha_range=[0.1, 0.3, 0.5],
                                 out_dir=r".\\__outputs__",
                                 hdp_iters=1000)

    #    ret = optimiser.assess_hyper_sampling(verbose=True)
    #    print(ret)
    #
        ret_eta = optimiser.smart_optimisation(tail_prcnt=0.8,
                                               skip_factor=5,
                                               plot_cat="num.tables",
                                               verbose=True)
        print(ret_eta)

    #    ret_gamma_alpha = optimiser.traverse_gamma_alpha(ps=12)
    #    print(ret_gamma_alpha)

    #    print("Generating plots...")
    #    optimiser.generate_gibbs_states_plots(f"{optimiser.out_dir}\\"
    #                                          f"gamma_alpha")

        print("Done.")


if __name__ == "__main__":
    main()
    print("Execution finished.")
