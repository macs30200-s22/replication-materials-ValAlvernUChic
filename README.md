### Findings

The below image features a heat map that highlights the **cosine distance** between the use of the term migrantworker and the rest of the 

### Saving Files
Save hansard_full 2 into your chosen directory - record this directory; you'll need it

### Pre-Processing
It is recommended you either run the .ipynb file or run the following functions in a python kernel. 
## Step 0: Install requirements.txt 
```python 
pip install -r requirements.txt
```

## Step 1: Scan Folder and Initialize Helper Functions 
```python 
files = scan_folder(_hansard_full 2 directory_)
```

# Initialize:
```python 
word_tokenize(word_list)
migrant_worker(y)
normalizeTokens(word_list)
clean(df)
```

## Step 2: Create Dataframes 
```python
savepath = "wherever you want to save your yearly dataframes"
create_df(files, savepath)
```

## Step 3: Create Final Dataframe 
```python 
path = "wherever you saved your yearly dataframes"
savepath = "wherever you wanna save your final df" 

create_final_df(path, savepath)

### Word Embeddings
