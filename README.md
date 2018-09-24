Description.

Script performs unsupervized binominal clustering of text data.

Script execution.

PATH - local data file for classification
file schema:
root
 |-- _c0: string (nullable = true)
 |-- _c1: string (nullable = true)
 |-- _c2: string (nullable = true)
 
Where:
_c0 - key id, _c1 - text_id, _c2 - text
python classification.py <PATH>

exploration.ipynb has part of teoretical explanation and general script logic. 

Future steps:
1. Implement language detection. https://github.com/facebookresearch/fastText
2. Additional work to Stopwords
3. Models grid search.
4. Finish implementation of LDA
5. Make groupby on ownerIndex before clasterization.


