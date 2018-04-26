
# coding: utf-8

# In[1]:

print('started')

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error


# In[2]:


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# In[3]:


PATH_TO_DATA = ''


# In[4]:


def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


# In[5]:


def preprocess(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            output_list.append(content_no_html_tags)
    return output_list


# In[6]:


from scipy.sparse import csr_matrix, hstack


# In[8]:


train_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'train.json'),)


# In[9]:


test_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'test.json'),)


# In[10]:


cv = CountVectorizer(max_features=60000)


# In[11]:


X_train = cv.fit_transform(train_raw_content)


# In[12]:


X_test = cv.transform(test_raw_content)


# In[13]:


train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                           index_col='id')


# In[14]:


y_train = train_target['log_recommends'].values


# In[15]:


from sklearn.linear_model import Ridge


# In[16]:


ridge = Ridge(random_state=17)


# In[17]:


def get_date(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
            content = pd.to_datetime(json_data['published']['$date'])
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
          #  return output_list
    return output_list


# In[18]:


def get_author(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
            content = json_data['author']['url']
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
           # return output_list
    return output_list;


# In[19]:


def get_domain(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
            content = json_data['domain']
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
           # return output_list
    return output_list;


# In[20]:


authors_train = get_author(os.path.join(PATH_TO_DATA,'train.json'))


# In[21]:


authors_test = get_author(os.path.join(PATH_TO_DATA,'test.json'))


# In[22]:


df = pd.DataFrame(
    {'authors': authors_train,
     'y': y_train
    })

df_test = pd.DataFrame(
    {'authors': authors_test
    })

df_common = pd.DataFrame(
    {'authors': authors_train+authors_test
    })


# In[23]:


df_test["domain"] = get_domain((os.path.join(PATH_TO_DATA,'test.json')))


# In[24]:


df["domain"] = get_domain((os.path.join(PATH_TO_DATA,'train.json')))


# In[60]:


df_common["domain"] = df["domain"].append(df_test["domain"], ignore_index=True)


# In[25]:


X_train_authors = csr_matrix(hstack([X_train, 
                             (pd.get_dummies(df_common["authors"]))[:df.shape[0]]]))


# In[26]:


X_test_authors = csr_matrix(hstack([X_test, 
                             (pd.get_dummies(df_common["authors"]))[df.shape[0]:]]))


# In[27]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[28]:


tf_idf_transformer = TfidfTransformer(use_idf=True).fit(X_train)
X_train_tf_idf = tf_idf_transformer.transform(X_train)


# In[29]:


X_test_tf_idf = tf_idf_transformer.transform(X_test)


# In[30]:


X_train_authors_tf_idf = csr_matrix(hstack([X_train_tf_idf, 
                             (pd.get_dummies(df_common["authors"]))[:df.shape[0]]]))


# In[31]:


X_test_authors_tf_idf = csr_matrix(hstack([X_test_tf_idf, 
                             (pd.get_dummies(df_common["authors"]))[df.shape[0]:]]))


# In[32]:


def get_minute_read(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
            content = int((json_data['meta_tags']['twitter:data1']).split()[0])
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
          #  return output_list
    return output_list


# In[33]:


def get_title_read(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
            content = json_data['meta_tags']['title'].split('\u2013')[0].strip()
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
          #  return output_list
    return output_list


# In[34]:


df['read_minutes']=get_minute_read(os.path.join(PATH_TO_DATA,'train.json'))


# In[35]:


df_test['read_minutes']=get_minute_read(os.path.join(PATH_TO_DATA,'test.json'))


# In[36]:


df['read_minutes2'] = df['read_minutes'].apply(lambda t: t if t<= 17 else 18)


# In[37]:


df_test['read_minutes2'] = df_test['read_minutes'].apply(lambda t: t if t<= 17 else 18)


# In[38]:


X_train_authors_tf_idf_read_minutes2 = csr_matrix(hstack([X_train_authors_tf_idf, 
                             df['read_minutes2'].values.reshape(-1, 1)                                                     ]
                                                    ))


# In[39]:


X_test_authors_tf_idf_read_minutes2 = csr_matrix(hstack([X_test_authors_tf_idf, 
                             df_test['read_minutes2'].values.reshape(-1, 1)                                                     ]
                                                    ))


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[42]:


scaler.fit(df['read_minutes2'].values.reshape(-1, 1))
df['read_minutes2_scaled'] = scaler.transform(df['read_minutes2'].values.reshape(-1, 1))


# In[43]:


df_test['read_minutes2_scaled'] = scaler.transform(df_test['read_minutes2'].values.reshape(-1, 1))


# In[44]:


X_train_authors_tf_idf_read_minutes2_scaled = csr_matrix(hstack([X_train_authors_tf_idf, 
                             df['read_minutes2_scaled'].values.reshape(-1, 1)                                                     ]
                                                    ))


# In[45]:


X_test_authors_tf_idf_read_minutes2_scaled = csr_matrix(hstack([X_test_authors_tf_idf, 
                             df_test['read_minutes2_scaled'].values.reshape(-1, 1)                                                     ]
                                                    ))


# In[46]:


def get_image_size(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
           # print(json_data['image_url'])
            content = 0
            if json_data['image_url']:
                content = int((json_data['image_url']).split('/')[4])
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
          #  return output_list
    return output_list


# In[47]:


df['image_size']=get_image_size(os.path.join(PATH_TO_DATA,'train.json'))


# In[48]:


df_test['image_size']=get_image_size(os.path.join(PATH_TO_DATA,'test.json'))


# In[49]:


scaler.fit(df['image_size'].values.reshape(-1, 1))
df['image_size_scaled'] = scaler.transform(df['image_size'].values.reshape(-1, 1))


# In[50]:


df_test['image_size_scaled'] = scaler.transform(df_test['image_size'].values.reshape(-1, 1))


# In[51]:


def get_section(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm(inp_file):
       #     print(line)
            json_data = read_json_line(line)
            content = ((json_data['_id']).split('/')[3])
          #  print(content)
            #content_no_html_tags = strip_tags(content)
            output_list.append(content)
          #  return output_list
    return output_list


# In[52]:


df['section']=get_section(os.path.join(PATH_TO_DATA,'train.json'))


# In[53]:


df_test['section']=get_section(os.path.join(PATH_TO_DATA,'test.json'))


# In[54]:


df_common["section"] = df["section"].append(df_test["section"], ignore_index=True)


# In[55]:


X_train_authors_tf_idf_read_minutes2_scaled_section = csr_matrix(hstack([X_train_authors_tf_idf_read_minutes2_scaled, 
                             (pd.get_dummies(df_common["section"]))[:df.shape[0]]]))


# In[58]:


X_test_authors_tf_idf_read_minutes2_scaled_section = csr_matrix(hstack([X_test_authors_tf_idf_read_minutes2_scaled, 
                             (pd.get_dummies(df_common["section"]))[df.shape[0]:]]))


# In[61]:


X_train_domain = csr_matrix(hstack([X_train_authors_tf_idf_read_minutes2_scaled_section, 
                             (pd.get_dummies(df_common["domain"]))[:df.shape[0]]]))


# In[62]:


X_test_domain = csr_matrix(hstack([X_test_authors_tf_idf_read_minutes2_scaled_section, 
                             (pd.get_dummies(df_common["domain"]))[df.shape[0]:]]))


# In[63]:


X_train_domain_image = csr_matrix(hstack([X_train_domain, 
                             df['image_size_scaled'].values.reshape(-1, 1)                                                     ]
                                                    ))


# In[64]:


X_test_domain_image = csr_matrix(hstack([X_test_domain, 
                             df_test['image_size_scaled'].values.reshape(-1, 1)                                                     ]
                                                    ))


# In[65]:


def to_log_value(item):
    if item < np.log(2).round(5):
        return np.log(2).round(5);
    return np.log(np.round(np.exp(item))).round(5)


# In[66]:


ridge.fit(X_train_domain_image, y_train)


# In[67]:


X_test_domain_image_pred = ridge.predict(X_test_domain_image)


# In[68]:


X_test_domain_image_pred_log = list(map(to_log_value, X_test_domain_image_pred))


# In[69]:


X_test_domain_image_pred_log_hsum=X_test_domain_image_pred_log+(4.33328-np.mean(X_test_domain_image_pred_log))


# In[71]:


def write_submission_file(prediction, filename,
    path_to_sample=os.path.join(PATH_TO_DATA, 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# In[72]:


write_submission_file(prediction=X_test_domain_image_pred_log_hsum, 
                      filename='X_test_domain_image_pred_log_hsum_final.csv')

print('end. output file X_test_domain_image_pred_log_hsum_final.csv')


