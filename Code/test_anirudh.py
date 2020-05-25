import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

words353_filename = "./reza/wordsim353_sim_rel/wordsim353_agreed.txt"
cname = ['desc','word1','word2','rating']
df = pd.read_csv(words353_filename, skiprows=11, sep='\t', names = cname)
df = df[['word1','word2','rating']]
data = np.load('./anirudh/Emb_SG10.npy', allow_pickle=True).item()

for k,v in data.items():
    data.update({k.upper(): v})


sim = []
for index, row in df.iterrows():
    w1 = row['word1'].upper()
    w2 = row['word2'].upper()
    sim.append(cosine_similarity(data[w1].reshape(1,-1), data[w2].reshape(1,-1)))


sim = np.array(sim).squeeze()
df['emb'] = sim

df.corr(method ='pearson')

print(1)