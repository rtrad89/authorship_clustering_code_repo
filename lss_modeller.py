# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations  # To defer evaluation of type hints
import subprocess as s
from gensim.corpora import Dictionary, bleicorpus
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import time
import pandas as pd
from itertools import product
from aiders import Tools
from typing import Tuple, List
from collections import defaultdict
import seaborn as sns
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
                 hdp_eta: float = 0.5,
                 input_docs_path: str = None,
                 input_amazon_path: str = None,
                 input_amazon_fname: str = None,
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
        self.input_amazon_path = input_amazon_path
        self.input_amazon_filename = input_amazon_fname
        self.lda_c_fname = ldac_filename
        self.hdp_output_directory = f"{hdp_output_dir}_Hyper{hdp_sample_hyper}"
        self.hdp_iterations = hdp_iters
        self.hdp_seed = hdp_seed
        self.hdp_hyper_sampling = hdp_sample_hyper
        self.hdp_eta = hdp_eta
        self.word_grams = word_grams
        self.doc_index = []  # the index of the files read for reference
        self.verbose = verbose

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
        with Tools.scan_directory(self.input_docs_path) as docs:
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
        output_dir = r"{}\lda_c_format_Hyper{}".format(self.input_docs_path,
                                                       self.hdp_hyper_sampling)
        Tools.initialise_directory(output_dir)
        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)
        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=bow_corpus,
                id2word=id2word_map)
        return plain_docs, bow_corpus

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
        Tools.initialise_directory(output_dir)
        save_location = r"{}\{}.dat".format(
                output_dir, self.lda_c_fname)
        bleicorpus.BleiCorpus.serialize(
                fname=save_location, corpus=amazon_df.bow,
                id2word=id2word_map)
        return amazon_df

    def _invoke_gibbs_hdp(self):
        """Invoke Gibbs hdp posterior inference on the corpus"""
        path_executable = r"{}\hdp.exe".format(self.hdp_path)
        param_data = r"{}\lda_c_format_Hyper{}\{}.dat".format(
                self.input_docs_path,
                self.hdp_hyper_sampling,
                self.lda_c_fname)
        param_directory = r"{}\{}".format(self.input_docs_path,
                                          self.hdp_output_directory)
        # Prepare the output directory
        Tools.initialise_directory(param_directory)

        ret = s.run([path_executable,
                     "--algorithm",     "train",
                     "--data",          param_data,
                     "--directory",     param_directory,
                     "--max_iter",      str(self.hdp_iterations),
                     "--sample_hyper",  "yes" if self.hdp_hyper_sampling
                     else "no",
                     "--save_lag",      "-1",
                     "--eta",           str(self.hdp_eta),
                     "--random_seed",   str(self.hdp_seed)],
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

    def get_corpus_lss(self, infer_lss,
                       bim_thresold: int = 0, bim: bool = False
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
        size = ((60 // skip_factor)
                * len(self.etas)
                * len(self.gammas)**2
                * len(self.alphas)**2)
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
                for eta in self.etas:
                    for g_s, g_r in product(self.gammas, repeat=2):
                        for a_s, a_r in product(self.alphas, repeat=2):
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
                           tail_prcnt: float = 0.80,
                           skip_factor: int = 1,
                           verbose: bool = False):
        # First generate the outputs to compare:
        words_counts = self._generate_hdps_outputs(skip_factor=skip_factor,
                                                   verbose=verbose)

        ret = {}
        # Loop over the outputs of different etas
        master_folder = (f"{self.out_dir}\\optimisation")
        pw_ll = []
        errors = []
        with Tools.scan_directory(master_folder) as perms:
            for perm in perms:
                # generate plots
                self.generate_gibbs_states_plots(states_path=perm.path)
                with Tools.scan_directory(perm.path) as problems:
                    for problem in problems:
                        try:
                            n_words = words_counts[problem.name]
                            path_state = f"{problem.path}\\state.log"
                            df_state = pd.read_csv(
                                    filepath_or_buffer=path_state,
                                    delim_whitespace=True,
                                    index_col="iter",
                                    usecols=["iter", "likelihood"],
                                    squeeze=True)
                            ll = df_state.tail(round(
                                    len(df_state) * tail_prcnt
                                    )).mean()
                            pw_ll.append(ll / n_words)
                        except FileNotFoundError as e:
                            print(f"{e}")
                            errors.append(f"{e}")
                            continue
                        except KeyError:
                            # Plots folders are being queried for n_words
                            continue
                ret.update({f"{perm.name}":
                            round(sum(pw_ll) / len(pw_ll), 4)})
        # Save any encountered errors to disk too
        Tools.save_list_to_text(mylist=errors,
                                filepath=(f"{self.out_dir}\\optimisation"
                                          f"\\opt_errors.txt"))

        pd.DataFrame(data=ret, index=[0]).to_csv(
                f"{self.out_dir}\\optimisation\\optimisation.csv",
                index=False)

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
                                    states_path: str):
        Tools.initialise_directory(f"{states_path}\\plots")
        with Tools.scan_directory(states_path) as outputs:
            for i, output in enumerate(outputs):
                try:
                    state_file = f"{output.path}\\state.log"
                    df = pd.read_csv(filepath_or_buffer=state_file,
                                     delim_whitespace=True,
                                     index_col="iter")
                    ax = sns.lineplot(x=df.index, y=df.likelihood, data=df)
                    ax.margins(x=0)
                    name = output.name
                    fig = ax.get_figure()
                    fig.savefig(
                            f"{states_path}\\plots\\{name}.png",
                            dpi=300,
                            bbox_incehs="tight",
                            format="png")
                    fig.clf()
                    print(f"{i}")
                except FileNotFoundError:
                    print(f"→ Skipping {output.name}")


def main():
    print("Main thread started..\n")
    folders_path = (r"D:\College\DKEM\Thesis"
                    r"\AuthorshipClustering\Datasets\pan17_train")
    hdp = r"D:\College\DKEM\Thesis\AuthorshipClustering\Code\hdps\hdp"

    optimiser = LssOptimiser(train_folders_path=folders_path,
                             hdp_path=hdp,
                             ldac_filename="dummy_ldac_corpus.dat",
                             hdp_seed=None,
                             eta_range=[0.5, 2.5, 5],
                             gamma_range=[0.1, 0.5],
                             alpha_range=[0.05, 0.25],
                             out_dir=r".\\__outputs__",
                             hdp_iters=1000)

#    ret = optimiser.assess_hyper_sampling(verbose=True)
#    print(ret)
#
    ret_eta = optimiser.smart_optimisation(tail_prcnt=0.8,
                                           skip_factor=5,
                                           verbose=True)
    print(ret_eta)

#    ret_gamma_alpha = optimiser.traverse_gamma_alpha(ps=12)
#    print(ret_gamma_alpha)

#    print("Generating plots...")
#    optimiser.generate_gibbs_states_plots(f"{optimiser.out_dir}\\gamma_alpha")

    print("Done.")


if __name__ == "__main__":
    main()
