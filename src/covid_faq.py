import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import operator

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def get_text_data():
    df = pd.read_csv("covid.csv")
    df.rename( columns={'Unnamed: 0':'id'}, inplace=True )
    df['encoded_questions'] = df['question'].apply(lambda x: get_sentence_embeding([x]))
    df=df[['id', 'question','answer','encoded_questions', 'source','country', 'link']]
    return df

def get_video_data():
    df = pd.read_csv("video_data2.csv")
    df=df[['id', 'Youtube link','Question','Transcript']]
    df.rename( columns={'Youtube link':'youtube_link'}, inplace=True )
    df['id'] = df['id'].astype(str)
    return df


def get_context_values(data):
    df = get_text_data()
    predicted_context = []
    i = 1
    for key, value in data.items():
        if i < 6:
            context = df.iloc[key]['answer']
            link = df.iloc[key]['link']
            id =  df.iloc[key]['id']
            first_array = []
            first_array.append(context)
            first_array.append(link)
            first_array.append(id)
            predicted_context.append(first_array)
            i += 1

    return predicted_context

def get_sentence_embeding(sentences):
    preprocessed_text = sbert_model.encode(sentences)
    return preprocessed_text


def search_covid_text_dataset(question):
    df = get_text_data()
    similar_vector_values = []
    similar_vector = {}
    query = question
    # print(query)
    query_vectors = sbert_model.encode([query])
    for index, row in df.iterrows():
        ans = row['encoded_questions']
        similar_vector_values.append(np.sum(cosine_similarity(ans, query_vectors)))
        similar_vector[index] = np.sum(cosine_similarity(ans, query_vectors))

    # print(similar_vector_values)
    dx = np.argmax(np.array(similar_vector_values))
    sorted_d = dict(sorted(similar_vector.items(), key=operator.itemgetter(1), reverse=True))
    pred_ans = get_context_values(sorted_d)

    return pred_ans

def get_time_stamp(question, ans):
  video_df = get_video_data()
  # print("head cdata " , df.head())
  ans = np.array(ans)
  ids = ans[:, 2]
  video_ans = []
  videos = video_df[video_df['id'].isin(ids)]
  # print(videos)

  for index, row in videos.iterrows():
      id = row['id']
      predicted_transcript = row['Transcript']
      sent_dict = {}
      video_split = predicted_transcript.splitlines()
      youtube_link = row['youtube_link']
      similar_vector_values = []
      for i in range(0, len(video_split),4):
          time = video_split[i]
          sent = video_split[i+1]
          if i+3 < len(video_split):
             sent = sent + video_split[i+3]
          sent_dict[time] = sent
      transcript_df = pd.DataFrame(list(sent_dict.items()),columns = ['timestamp','sent'])
      transcript_df['encoded_transcript'] = transcript_df['sent'].apply(lambda x: get_sentence_embeding([x]))

      query = question
      query_vectors = sbert_model.encode([query])
      for index, row in transcript_df.iterrows():
          ans = row['encoded_transcript']
          similar_vector_values.append(np.sum(cosine_similarity(ans, query_vectors)))

      # print(similar_vector_values)
      dx = np.argmax(np.array(similar_vector_values))

      predicted_time = transcript_df.iloc[dx]['timestamp']
      video_ans.append([predicted_time, youtube_link, id ])

  return video_ans

if __name__ == '__main__':
    question = "What is a novel coronavirus?"
    text_ans = search_covid_text_dataset(question)
    print("text ans " , text_ans)
    video_time = get_time_stamp(question, text_ans)

    print("video timestamps ", video_time)
