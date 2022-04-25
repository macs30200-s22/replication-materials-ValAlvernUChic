[![DOI](https://zenodo.org/badge/481045361.svg)](https://zenodo.org/badge/latestdoi/481045361)

# Findings

The below image features a heat map that highlights the divergence between different years with respect to the use of the term migrantworker. The score ranges from 0-2, with higher values indicating greater divergences and vice versa. For example, between 2012 and 2013, the use of the term 'minister' had little change 

![alt text](https://github.com/macs30200-s22/replication-materials-ValAlvernUChic/blob/main/riot%20embeddings.png)
![alt text](https://github.com/macs30200-s22/replication-materials-ValAlvernUChic/blob/main/migrantworker%20embeddings.png)

# Saving Files
Save hansard_full 2 into your chosen directory - record this directory; you'll need it

# Pre-Processing
It is recommended you either work from the Data_preprocessing.ipynb file or run the following functions in a python kernel. 

### Step 0: Install requirements.txt 
```python 
pip install -r requirements.txt
import Data_preprocessing
```

### Step 1: Scan Folder and Initialize Helper Functions 
```python 
files = scan_folder(_hansard_full 2 directory_)
```

### Initialize: 
```python 
word_tokenize(word_list)
migrant_worker(y)
normalizeTokens(word_list)
clean(df)
```

### Step 2: Create Dataframes 
```python
savepath = "wherever you want to save your yearly dataframes *make sure it's a directory! and ends with a /"
create_df(files, savepath)
```

### Step 3: Create Final Dataframe 
```python 
path = "wherever you saved your yearly dataframes"
savepath = "wherever you wanna save your final df" 

create_final_df(path, savepath)
```

# Word Embeddings

### Step 1: Initialize Helper Functions (if working from .ipynb file)
```python 
calc_syn0norm(model)
smart_procrustes_align_gensim(base_embed, other_embed, words=None)
intersection_align_gensim(m1, m2, words=None)
```

### Step 2: Create embeddings (start here if running module from python kernel)
```python
import data_analyze_embeddings
df = "path to final_df created in preprocessing"
rawEmbeddings, comparedEmbeddings = compareModels(df, 'year', sort = True)
```

### Step 3: Create heat map showing the temporal shift in chosen word 
```python 
word = "word of interest" 
embeddingsDict = comparedEmbeddings
pltDF = getDivergenceDF(targetWord, comparedEmbeddings)
fig, ax = plt.subplots(figsize = (10, 7))
seaborn.heatmap(pltDF, ax = ax, annot = True) #set annot True for a lot more information
ax.set_xlabel("Starting year")
ax.set_ylabel("Final year")
ax.set_ylabel("Final year")
ax.set_title("Yearly linguistic change for: '{}'".format(targetWord))
plt.show()
```
