#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
openai.api_key = "416cf50a7b1d4159b33f59b681cb515f"
openai.api_base =  "https://ai-playpen-delta.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2022-12-01'
from flask import Flask, request, jsonify

app = Flask(__name__)


embedding_df=pd.read_csv('./embeddings.csv', index_col=0)
embedding_df['embeddings'] = embedding_df['embeddings'].apply(eval).apply(np.array)

# df.head()


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    You are helping our relationship manager in the company to recommend product to our customers
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

@app.route('/', methods=['GET'])
def answer_question(
    df=embedding_df,
    model="text-davinci-003",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    question = request.args.get('question')
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
            engine = model
        )
        return jsonify ({"response": response["choices"][0]["text"].strip()})
    except Exception as e:
        print(e)
        return ""
    
if __name__ == "__main__":
    app.run()


# # In[32]:


# answer_question(df, question="Do you know any mortgage products available in HSBC?", debug=False)


# # In[33]:


# answer_question(df, question="What is the interest rate for HSBC Elite Mortgage?", debug=False)


# # In[34]:


# answer_question(df, question="How to apply for HSBC Elite Mortgaga?", debug=False)


# # In[42]:


# answer_question(df, question="What services do HSBC provide?", debug=False)

