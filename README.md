# embedland
Theoretically this is a universe of code for playing with embeddings. In reality it contains one file. More to come, I hope.

![](https://user-images.githubusercontent.com/279531/221034510-aa4084a9-86dd-4ddc-99de-8718acd211b4.png)

### bench.py
This file benchmarks various embeddings using the Enron email corpus. Once you install the various libraries it needs, you can run it with python bench.py. It will:
* Download your kindle highlights. This method uses the file generated by [Calibre](https://calibre-ebook.com/)
* Run embeddings on it
* Cluster the embeddings.
* Label the clusters by sampling the subject lines from the clusters and sending them to GPT-3.
* Show you a pretty chart, like the one you see above. 