import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def get_text():
    quotes = pd.read_csv("list.csv")
    quotes = quotes["quote"].to_list()
    print(type(quotes))
    return quotes

def create_df(embeddings):
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    print(labels)
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    quotes = get_text()
    labels_dict = {'0': 'Letting Go', '1': 'Courage and Vulnerability', '2': 'Market Behavior & Investing', '3': 'Psychological Effects of Competition', '4': 'Interpreting Reality', '5': 'Financial Perspectives', '6': 'Vulnerability and Feedback', '7': 'Biological Lessons of History', '8': 'Taking Action and Making Decisions', '9': 'Wisdom and Resilience'}

    df = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label":  labels.astype(int), "quote": quotes})
    df["label"] = df["label"].apply(lambda label: labels_dict[str(label)])
    return df


def plot_ploty(df, file_name):
    # print(labels)
    # print(labels_dict)
    # df = pd.DataFrame(
    #     {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label":  labels.astype(int)})
    # df["label"] = df["label"].apply(lambda label: labels_dict[str(label)])
    # print(df.head())
    fig = px.scatter(df, x="x", y="y", color="label")
    fig.show()
    # save the image
    fig.write_image(file_name, width=1920, height=1080)

file_name = '337c6-openai-cluster10-quote1178-embeddings'
CLUSTER_COUNT = 10
embeddings = np.load(file_name+'.npy')
file_name = ""
df = create_df(embeddings)
output_df = df.drop(columns=['x', 'y'])
output_df.to_csv("output.csv", index=False)
plot_ploty(df, file_name + '.png')