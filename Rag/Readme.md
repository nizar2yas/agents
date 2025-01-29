# RAGs:
This folder contains tries to create a rag using differente componantes.
### RAG1 (notebook)
This is the first test, in which everything is in local.<br>
I start by partition the file `doc_latest.md` (generated from the synthetic ai) using `unstructured` library, then I embed the chunks and store them in a `pgvector` database deployed in a docker container, <br>
After that I played with differente parameters and test adding the rag

### RAG1 (py)
Based on the vecotr db populated in the previous step, I created a RAG<br>
again, here everything is in local.

### RAG2 
in this file I tried to move from local pgvector instance to remote one, deployed in cloud sql, thus as RAG1, I did the same thing except that the db is in gcp

### RAG3
I tried to move things on to the cloud, thus I put the file in `gcs` but I had some difficulties using `unstructured` with a file in `gcs`, that's why I tried the google `documentai` instead of `unstructured`

### RAG4
I finally figure out how to use `unstructured` with a file in `gcs`, so this file use 2 elements from the cloud : 
 * source file 
 * vector db