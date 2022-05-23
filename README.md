[![DOI](https://zenodo.org/badge/481045361.svg)](https://zenodo.org/badge/latestdoi/481045361)

# Findings

The below image features a heat map that highlights the divergence between different years with respect to the use of the term 'minister'. The score ranges from 0-2, with higher values indicating greater divergences and vice versa. For example, between 2012 -2014, 2015-2019, and 2020-2021, the use of the term 'minister' had little change in those groups. This makes sense as these years coincide with the individual parliament sessions (12th, 13th, and 14th). 

![alt text](https://github.com/macs30200-s22/replication-materials-ValAlvernUChic/blob/main/minister%20embeddings.png)

The below heatmap features how the word 'riot' evolved over the years. Interestingly, there was an increased difference in the use of the word between 2013 and 2014 from 2012 to 2013, with a score of 1.1. This suggests that the way riot was used changed from 2012 to 2014, potentially capturing a change from when the Little India riot happened in December, 2013. 

![alt text](https://github.com/macs30200-s22/replication-materials-ValAlvernUChic/blob/main/riot%20embeddings.png)

The heatmap below illustrates the change in how the term 'migrantworker' was used. Prior to the riot (2012-2013), it seems that the use of the term was stable with the divergence between the two years at 0.85. However, this changes between 2013-2014 and interestingly, looking at the first two columns of the graph shows that the two years were consistently different from the years following the riot. This could point toward a significant and long-lasting shift in the way the workers were talked about. This is additionally suggested by the cluster of values below 1 from 2014 onwards, indicating little differentiation from each other. 

![alt text](https://github.com/macs30200-s22/replication-materials-ValAlvernUChic/blob/main/migrantworker%20embeddings.png)

Relevance: These findings validate that the method used can capture high-level patterns that coincide with real-world temporal trends. The 'riot' and 'migrantworker' heatmaps will be especially useful starting points for more qualitative inspections of the data. The heatmaps can also be compared with/validated with KL divergence heatmaps to further illustrate the significance of the events. 

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

### Step 3a: Create heat map showing the temporal shift in chosen word 
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

### Step 3b: Create cultural dimensions projection 
```python 
# Projections 
morality_words = pd.read_csv('/home/valalvern/models/morality_pairs.csv')
word = 'migrantworker'
moral_words = [x for x in morality_words['moral_eng'] if pd.isnull(x) == False]
immoral_words = [x for x in morality_words['immoral_eng'] if pd.isnull(x) == False]

safe = ['secure', 'safe', 'sound', 'harmless', 'innocuous', 'benign', 'wholesome', 'mild', 'guard', 'shield', 'shielded', 'guarded', 'secured']
unsafe = ['insecure', 'unsafe', 'unsound', 'threat', 'harmful', 'hostile', 'vulnerable', 'reckless', 'dangerous', 'threatening', 'risky', 'prevarious', 'unpredictable']

def project_year(embeddingsDict, migrant):
    '''
    construct affluence, morality, and status dimensions, and
    project 'democracy' on these dimensions for each years
    '''
    m = []
    s = []
    cats = sorted(set(embeddingsDict.keys()))
    for catOuter in cats:
        print(catOuter)
        #embeddings_aligned[catOuter] = [embeddings_raw[catOuter]]
        #for embed in embeddingsDict[catOuter][1:]:
        embed = embeddingsDict[catOuter][0]
        security = dimension(embed, safe, unsafe)
        morality = dimension(embed, moral_words, immoral_words)
        try:
            m.append(cosine_similarity(embed.wv[migrant].reshape(1, -1), morality.reshape(1,-1))[0][0])
            s.append(cosine_similarity(embed.wv[migrant].reshape(1, -1), security.reshape(1, -1))[0][0])
        except:
            print(catOuter)
    projection_df = pd.DataFrame({'morality' : m, 'security': s}, index = [i for i in cats]
                                  )
    return projection_df
    
def plot_projection(projection, title):
    #plt.plot(projection_df_safety, label='safety')
    plt.plot(projection['morality'], label = 'morality')
    plt.plot(projection['security'], label = 'security')
    plt.title('{} Projection'.format(title))
    plt.xlabel('Year')
    plt.ylabel('Projection')
    plt.legend(loc=3)
    
```

If you use this repository for a scientific publication, we would appreciate it if you cited the Zenodo [DOI](https://zenodo.org/badge/latestdoi/481045361) (see the "Cite as" section on our Zenodo page for more details).
