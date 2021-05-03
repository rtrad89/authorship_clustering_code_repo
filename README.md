# Authorial Clustering of Shorter Texts With Non-parametric Topic Models
This code repository implements the approach presented in the associated [paper](https://rdcu.be/cjErU) ([preprint here](https://arxiv.org/abs/2011.15038)) to cluster a corpus of documents by authorship and produce the evaluated clustering on several fronts.

# Requirements
The code is developed mainly with Python 3. You can refer to `requirements.txt` file for necessary Python packages in order to run the code. Please pay special attention to `scikit-learn` version, **which should NOT be newer than 0.22.0** due to compatibility problems with Spherical K-Means implementation (cf. [this relevant issue](https://github.com/jasonlaska/spherecluster/issues/26)). In addition, I have manually patched the code of Spherical KMeans to circumvent a similar compatibility problem, as the release `0.1.7` didn't do so. Future official releases shall fix this but till then, the local version runs smoothly and there is no need to install `spherecluster` library.

Moreover, the code depends on Hierarchical Dirichlet Process -- HDP as implemented in [blei-lab](https://github.com/blei-lab/hdp). The user should have the HDP code compiled properly in order to use it to produce the latent semantic representation of texts, aka. *LSSR*, for clustering.

Statistical testing was executed in `R` to produce the CD plots presented in the paper, but it is not critical to running the code.

As a final note, the code was developed and run on Windows 10. Slight differences in some results can occur from machine to machine depending on the hardware and software configurations, especially that we use a lot of floating-point calculations in our code.

# Usage
There is one main entry point to use the code: `cluster_docs.py`, besides an auxiliary entry point: `lssr_docs.py`. As the names suggest, `cluster_docs.py` clusters documents *represented in a LSSR* and `lssr_docs.py` *builds the relevant LSSR from a corpus of documents* using Blei's HDP.

There are two operation modes of the code files: _single_ and _multiple_ modes. The _single_ mode is designed to operate on one corpus, so the main input to the code in this case is a corpus of documents, **each in its own text file**, encased in a directory. With the _multiple_ mode, the user can operate on a directory of folders representing corpora, where each corpus is a folder with text files in it like in the single mode. If the user wants to use the multiple mode, `m` is entered as the first parameter when running the code files, otherwise any other value will be interpreted as _single_ operation mode. Since the first step in the workflow is to build the LSSR, let's explain how to use `lssr_docs.py` first then follow it with `cluster_docs.py`.

## `lssr_docs.py`
The code calls `HDP.exe`, provided that it is compiled beforehand as explained above to produce the compatible LSSR. The code is compatible with Windows operating system, and it is only a wrapper for HDP.

The option `-h` details all the parameters and their semantics, and example usages are:

```
python lssr_docs.py m D:\pan_train D:\Projects\Authorial_Clustering_Short_Texts_nPTM\Code\hdps\hdp -iter 10000

python lssr_docs.py -h
```

If desired, the code can be run similarly within Spyder's IPython console with `%run` command.

The result of the code is a folder named after the hyperparameters of HDP, containing four files. The most important file is `mode-word-assignments.dat`, which shall be utilised to derive LSSR. `state.log` is an auditing log that records the dynamics of execution.

## `cluster_docs.py`
The main inputs to this code are the LSSR and the ground truth. For the former, Blei's HDP generates four files, but we need only `mode-word-assignments.dat` as input to the clustering algorithm. The code automatically reshapes `mode-word-assignments.dat` and forms the topic counts of each document in the corpus. The ground truth is expected to be in a `JSON` file, which indicates the correct authorial clusters of the documents. It is used for evaluation purposes. In general, the data formats comply with [Author Clustering 2017](https://pan.webis.de/clef17/pan17-web/author-clustering.html) shared task in PAN @ CLEF 2017; authorship-link ranking is irrelevant though.

When operating in the _multiple_ mode, the same is applied to each corpus in the directory of corpora. This is a streamlined _single_ execution on a set of corpora. The help option `-h` details the parameters and their meanings. Examples:

```
python cluster_docs.py s D:\pancp D:\pancp\hdp_lss_0.30_0.10_0.10_common_False D:\pancp\truth\clustering.json D:\pancp -k 4

python cluster_docs.py m D:\pan_train hdp_lss_0.30_0.10_0.10_common_False D:\pan_train\truth -k 2 -suffix k2

python cluster_docs.py -h
```

Running the code results in two `CSV` files: `authorial_clustering_results` which exposes different extrinsic and intrinsic clustering evaluation scores given the ground truth, and `authorial_clustering_kvals` which stores the estimations of *k* -- the number of authorial clusters and essentially the authors -- selected by the different methods.

### Example
Running the code with the sample data provided under `example/data/corpora` with the associated metadata under `example/data/meta` should produce similar results to the ones under `example/results`. The sample data is taken from [PAN17-Clustering](https://pan.webis.de/data.html?q=pan17) training data set and all credits belong to Potthast et al. The call (on Windows) and output messages are:

```
python cluster_docs.py m ./example/data/corpora hdp_lss_0.30_0.10_0.10_common_True ./example/data/meta/truth ./example/results

[INFO]: > Clustering "problem001"
[INFO]: NumExpr defaulting to 8 threads.
[INFO]: LSSR loaded successfully
[INFO]: Spherical KMeans clustering done
[INFO]: Constrained KMeans clustering done.
[INFO]: > Clustering "problem002"
[INFO]: LSSR loaded successfully
[INFO]: Spherical KMeans clustering done
[INFO]: Constrained KMeans clustering done.
[INFO]: > Clustering "problem003"
[INFO]: LSSR loaded successfully
[INFO]: Spherical KMeans clustering done
[INFO]: Constrained KMeans clustering done.
[INFO]: > Clustering "problem004"
[INFO]: LSSR loaded successfully
[INFO]: Spherical KMeans clustering done
[INFO]: Constrained KMeans clustering done.
[INFO]: > Clustering "problem005"
[INFO]: LSSR loaded successfully
[INFO]: Spherical KMeans clustering done
[INFO]: Constrained KMeans clustering done.
[INFO]: > Clustering "truth"
[INFO]: Execution completed and results saved under ./example/results.
```

----

<small>*This project is licensed under the terms of the MIT license.*</small>
