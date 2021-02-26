# Authorial Clustering of Shorter Texts With Non-parametric Topic Models

Authorial clustering involves the grouping of documents written by the same author or team of authors without any prior positive examples of an author’s writing style or thematic preferences. For authorial clustering on shorter texts (paragraph-length texts that are typically shorter than conventional documents), the document representation is particularly important: very high-dimensional feature spaces lead to data sparsity and suffer from serious consequences like the curse of dimensionality, while feature selection may lead to information loss. I programmed a high-level framework which utilizes a compact data representation in a latent feature space derived with non-parametric topic modeling. Authorial clusters are identified thereafter in two scenarios: (a) fully unsupervised and (b) semi-supervised where a small number of shorter texts are known to belong to the same author (must-link constraints) or not (cannot-link constraints).

Experiments with 120 collections in three languages and two genres show that the topic-based latent feature space provides a promising level of performance while reducing the dimensionality by a factor of 1500x compared to state-of-the-arts! I also found that while prior knowledge on the precise number of authors (i.e. authorial clusters) does not contribute much to additional quality, little knowledge on constraints in authorial clusters memberships leads to clear performance improvements in front of this difficult task. 

In the end, thorough experimentation with standard metrics indicates that there still remains an ample room for improvement for authorial clustering, especially with shorter texts.

Full paper: [preprint](https://arxiv.org/abs/2011.15038), [peer-reviewed](https://ida2021.org/).