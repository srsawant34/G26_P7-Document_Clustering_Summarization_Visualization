# CSE 573 - G26 P7:Document Clustering Summarization Visualization

This repository is the implementation of the project "Document Clustering, Summarization and Visualization". For this project we perform document clustering, summarization and draw out insightful visualization for 20 News group dataset. 

## Data
We use [20 Newsgroups data](http://qwone.com/~jason/20Newsgroups/) for the implementaion. The same data could be found in [sklearn](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) as well, this data is saved as [20newsdata.csv](./data/20newsdata.csv).

## System Architecture and Algorithms

### Clustering

Following figure demonstrates the complete architecture and experiments for custering the documents.

![clustering](./figs/clustering_architecture.png)

### Summarization

Below figure demonstrates the architectureand experiments involved in abstractive and extractive summarization.

![summarization](./figs/summarization_architecture.png)

## Evaluation

### Clustering

#### RoBERTa

|                   | RAW       | UMAP     | PCA      | t-SNE     |
| ----------------- | --------- | -------- | -------- | --------- |
| Silhouette Score  | 0.05      | 0.468    | 0.406    | 0.32      |
| Davies Bouldin    | 3.68      | 0.74     | 0.77     | 0.85      |
| Calinski harabasz | 168.51    | 30561.54 | 13235.05 | 12130.29  |

Additional evauluatons can be found in [clustering notebook](./Clustering/bert%20lda%20clustering.ipynb).

### Summarization

|  ROUGE 1   | Precision  | Recall    | F-measure   |
| ---------- | ---------- | --------- | ----------- |
| PEGASUS    | 0.47       | 0.07      | 0.13        |
| GPT2       | 0.29       | 0.31      | 0.299       |

All visualizations can be found in respective notebooks: [clustering notebook](./Clustering/bert%20lda%20clustering.ipynb), [summarization](./Summarization/Abstractive_Summarization.ipynb)

## References
- [20 newsgroups. Home Page for 20 Newsgroups Data Set. (n.d.).](http://qwone.com/~jason/20Newsgroups/)
- [Karmakar, Saurav, "Syntactic and Semantic Analysis and Visualization of Unstructured English Texts." Dissertation, Georgia State University, 2011.](https://doi.org/10.57709/2292261)
- [Kim, SW., Gil, JM. Research paper classification systems based on TF-IDF and LDA schemes. Hum. Cent. Comput. Inf. Sci. 9, 30 (2019).](https://doi.org/10.1186/s13673-019-0192-7)
- S. Zaware, D. Patadiya, A. Gaikwad, S. Gulhane and A. Thakare, "Text Summarization using TF-IDF and Textrank algorithm," 2021 5th International Conference on Trends in Electronics and Informatics (ICOEI), Tirunelveli, India, 2021, pp. 1399-1407.
- [Kapadia, S. (2022, December 23). Topic modeling in Python: Latent dirichlet allocation (LDA). Medium](https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0)
- [Pegasus: Pre-training with extracted gap-sentences for abstractive ... (n.d.).](https://arxiv.org/pdf/1912.08777.pdf)
- [Devlin, J., Chang, M.-W., Lee, K., &amp; Toutanova, K. (2019, May 24). Bert: Pre-training of deep bidirectional Transformers for language understanding. arXiv.org.](https://arxiv.org/abs/1810.04805)
- [Ghantiwala, Alifia. “Using Word Clouds and N-grams to Visualize Text Data.” Medium, 16 Apr. 2022, Accessed 9 Oct. 2023.](gghantiwala.medium.com/using-wordclouds-and-n-grams-to-visualise-text-data-e71e96a3f152)
- [Eckerson, Wayne. “Using Treemaps to Visualize Data.” Data Plus Science, 5 Jan. 2016,](www.dataplusscience.com/UsingTreemaps.html)
- [Subakti, A., Murfi, H. & Hariadi, N. The performance of BERT as data representation of text clustering. J Big Data 9, 15 (2022).](https://doi.org/10.1186/s40537-022-00564-9)
- [Bert - Hugging face. BERT. (n.d.)](https://huggingface.co/docs/transformers/main/en/model_doc/bert)
- [Grootendorst, M. P. (n.d.). BERTopic.](https://maartengr.github.io/BERTopic/index.html)
- [Clustering with scikit-learn: A tutorial on unsupervised learning. KDnuggets. (n.d.).](https://www.kdnuggets.com/2023/05/clustering-scikitlearn-tutorial-unsupervised-learning.html)