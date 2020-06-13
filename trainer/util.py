
from nltk.stem import PorterStemmer , WordNetLemmatizer 
import numpy as np
from google.cloud import storage
from zipfile import ZipFile
from zipfile import is_zipfile
import io

ps = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 




def get_embedding_matrix(word_index,embedding_path,embedding_dim):
        

    num_words = len(word_index)+1

    embedding_dict={}
    with open(embedding_path,'r') as f:
        for line in f:
            values=line.split()
            word=values[0]
            vectors=np.asarray(values[1:],'float32')
            embedding_dict[word]=vectors
    f.close()

    embedding_matrix = np.zeros((num_words,embedding_dim))

    for word,i in word_index.items():

      if i > num_words:
          continue

      emb_vec=embedding_dict.get(word)

      if emb_vec is  None:
          emb_vec=embedding_dict.get(lemmatizer.lemmatize(word))
      elif emb_vec is None:
          emb_vec=embedding_dict.get(ps.stem(word))

      elif emb_vec is not None:
          embedding_matrix[i]=emb_vec
          
    return embedding_matrix




def zipextract(bucketname, zipfilename_with_path):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucketname)

    destination_blob_pathname = zipfilename_with_path

    blob = bucket.blob(destination_blob_pathname)
    zipbytes = io.BytesIO(blob.download_as_string())

    if is_zipfile(zipbytes):
        with ZipFile(zipbytes, 'r') as myzip:
            for contentfilename in myzip.namelist():
                contentfile = myzip.read(contentfilename)
                blob = bucket.blob(zipfilename_with_path + "/" + contentfilename)
                blob.upload_from_string(contentfile)

# zipextract("mybucket", "path/file.zip") # if the file is gs://mybucket/path/file.zip




