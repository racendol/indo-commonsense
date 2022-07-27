import jsonlines
from copy import deepcopy
from itertools import combinations
import pandas as pd
import numpy as np

def create_df_coref(coref_filepath):
    corpus_sentences = []
    with jsonlines.open(coref_filepath) as reader:
        cnt = 0
        coref = list()
        for obj in reader:
            clusters = obj['clusters']
            sentence = [x for y in obj['sentences'] for x in y]
            corpus_sentences.append(obj['sentences'])

            dct = dict()
            cnt_s = 0
            for s in obj['sentences']:
                dct[len(s) + cnt_s]= []
                cnt_s += len(s)

            dct_init = deepcopy(dct)

            for cluster in clusters:
                words = []
                dct_cluster = deepcopy(dct_init)

                for span in cluster:
                    offset = 0
                    for key in dct_cluster:
                        if span[0] < key:
                            span[0] -= offset
                            span[1] -= offset
                            dct_cluster[key].append(span)
                            break
                        offset = key

                for key in dct_cluster:
                    dct[key].append(dct_cluster[key])

            coref.append(dct)


    data_dct = {'sentence': [], 'span1': [], 'span2': [], 'label': []}
    corpus_counter = 0
    for text_dct in coref:
        sentence_counter = 0
        for key in text_dct:
            clusters = text_dct[key]
            sentence = corpus_sentences[corpus_counter][sentence_counter]
            sentence_counter += 1

            # only use sentence that have minimal 2 mention / 1 span
            len_t = 0
            for cluster in clusters:
                len_t += len(cluster)    

            if len_t < 2:
                continue

            spans_lst = list()
            for idx, cluster in enumerate(clusters):
                if len(cluster) > 1:
                    for span in cluster:
                        spans_lst.append((idx, span))

            comb_spans = combinations(spans_lst, 2)

            for comb in comb_spans:
                span_cluster1, span1 = comb[0]
                span_cluster2, span2 = comb[1]

                if span_cluster1 != span_cluster2:
                    label = 0 # different cluster, so false / not the same
                else:
                    label = 1


                data_dct['sentence'].append(sentence)
                data_dct['span1'].append(span1)
                data_dct['span2'].append(span2)
                data_dct['label'].append(label)

        corpus_counter += 1
        
    return pd.DataFrame(data_dct)


from conllu import parse_tree_incr, parse_incr

def create_df_dependency(path):
    data_file = open(path, "r", encoding="utf-8")

    sentence_lst = list(parse_incr(data_file))
    data_dct = {'sentence': [], 'span1': [], 'span2': [], 'label': []}

    for sentence in sentence_lst:
        sentence_token = [token['form'] for token in sentence]
        root = sentence.to_tree()

        nodes = list()
        nodes.append(root)

        while len(nodes) > 0:
            curr_root = nodes.pop(0)
            for children in curr_root.children:
                nodes.append(children)

                data_dct['sentence'].append(sentence_token)
                data_dct['span1'].append([curr_root.token['id']-1, curr_root.token['id']-1])
                data_dct['span2'].append([children.token['id']-1, children.token['id']-1])
                data_dct['label'].append(children.token["deprel"])
    
    return pd.DataFrame(data_dct)


from nltk.tree import Tree

def create_df_const(path):
    data_dct = {'sentence': [], 'span': [], 'label': []}
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
        for line in lines:
            stripline = line.strip()
            root = Tree.fromstring(stripline)

            root_sentence = root.leaves()
            root_label = root.label()
            data_dct['sentence'].append(root_sentence)
            data_dct['label'].append(root_label)
            data_dct['span'].append([0, len(root_sentence)-1])

            nodes = []
            for children in root:
                nodes.append(children)

            while len(nodes) > 0:
                curr_node = nodes.pop(0)

                sentence = curr_node.leaves()
                label = curr_node.label()

                start_span = root_sentence.index(sentence[0])
                end_span = root_sentence.index(sentence[-1])

                data_dct['sentence'].append(root_sentence)
                data_dct['label'].append(label)
                data_dct['span'].append([start_span, end_span])

                for children in curr_node:
                    if type(children) == str:
                        break
                    nodes.append(children)
                
    return pd.DataFrame(data_dct)


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
def preprocess_comve_a(comve_df):
    # using train_test_split to split data 50/50 for each label
    df_a0, df_a1 = pd.DataFrame(),  pd.DataFrame()
    label0, label1 = train_test_split(comve_df, test_size=0.5, random_state=1)
    
    df_a0['s1'] = label0['nonsense_sentence']
    df_a0['s2'] = label0['logical_sentence']
    df_a0['label'] = 0
    
    df_a1['s1'] = label1['logical_sentence']
    df_a1['s2'] = label1['nonsense_sentence']
    df_a1['label'] = 1
    
    df_a = pd.concat((df_a0, df_a1))
    df_a = shuffle(df_a, random_state=1)
    
    return pd.DataFrame(df_a)

def preprocess_comve_b(comve_df):
    # only use correct_reasoning 1, wrong_reasoning 1, 2
    # using train_test_split to split data 33/33/33 for each label
    df_a0, df_a1, df_a2 = pd.DataFrame(),  pd.DataFrame(),  pd.DataFrame()
    label0, label1 = train_test_split(comve_df, train_size=1/3, random_state=1)
    label1, label2 = train_test_split(label1, train_size=0.50, random_state=1)
    
    df_a0['s1'] = label0['nonsense_sentence']
    df_a0['r1'] = label0['correct_reasoning_1']
    df_a0['r2'] = label0['wrong_reasoning_1']
    df_a0['r3'] = label0['wrong_reasoning_2']
    df_a0['label'] = 0
    
    df_a1['s1'] = label1['nonsense_sentence']
    df_a1['r1'] = label1['wrong_reasoning_1']
    df_a1['r2'] = label1['correct_reasoning_1']  
    df_a1['r3'] = label1['wrong_reasoning_2']
    df_a1['label'] = 1
    
    df_a2['s1'] = label2['nonsense_sentence']
    df_a2['r1'] = label2['wrong_reasoning_1']
    df_a2['r2'] = label2['wrong_reasoning_2']
    df_a2['r3'] = label2['correct_reasoning_1'] 
    df_a2['label'] = 2
    
    df_a = pd.concat((df_a0, df_a1, df_a2))
    df_a = shuffle(df_a, random_state=1)
    
    return pd.DataFrame(df_a)


from transformers import TFAutoModelForMultipleChoice
import tensorflow as tf
def build_classifier_model_wsc_hf(path=None, seq_len=128, freeze=False, from_pt=False, n=2):
    input_id1 = tf.keras.layers.Input(shape=(n, seq_len,), name='input_id1', dtype='int32')
    input_masks_id1 = tf.keras.layers.Input(shape=(n, seq_len,), name='mask_id1', dtype='int32')
    input_token_id1 = tf.keras.layers.Input(shape=(n, seq_len,), name='token_type_id1', dtype='int32')

    input1 = {'input_ids': input_id1, 'attention_mask':input_masks_id1, "token_type_ids":input_token_id1}
    
    encoder = TFAutoModelForMultipleChoice.from_pretrained(path, from_pt=from_pt)
    
    output = encoder(input1)

    return tf.keras.Model(inputs=[input_id1, input_masks_id1, input_token_id1], outputs=output.logits)


from transformers import TFAutoModel
import tensorflow as tf
def build_classifier_model_nli(path, seq_len=128, freeze=False, n=3, from_pt=False):
    input_id1 = tf.keras.layers.Input(shape=(seq_len,), name='input_id1', dtype='int32')
    input_masks_id1 = tf.keras.layers.Input(shape=(seq_len,), name='mask_id1', dtype='int32')
    input_token_id1 = tf.keras.layers.Input(shape=(seq_len,), name='token_type_id1', dtype='int32')

    input1 = {'input_ids': input_id1, 'attention_mask':input_masks_id1, "token_type_ids":input_token_id1}

    encoder = TFAutoModel.from_pretrained(path, from_pt=from_pt)
    encoder._name = 'bert_model'

    trainable = not freeze
    encoder.trainable = trainable

    output1 = encoder(input1)

    cls1 = output1[0][:, 0, :]
    cls1 = tf.keras.layers.Dropout(0.1)(cls1)
    classif = tf.keras.layers.Dense(n, name='classifier')(cls1)

    return tf.keras.Model(inputs=[input_id1, input_masks_id1, input_token_id1], outputs=classif)


from transformers import TFAutoModel
import tensorflow as tf
def build_classifier_model_coref(path, seq_len=128, freeze=False, from_pt=False, output_label=2):
    input_id1 = tf.keras.layers.Input(shape=(seq_len,), name='input_id1', dtype='int32')
    input_masks_id1 = tf.keras.layers.Input(shape=(seq_len,), name='mask_id1', dtype='int32')
    input_token_id1 = tf.keras.layers.Input(shape=(seq_len,), name='token_type_id1', dtype='int32')
    
    span_mask1 = tf.keras.layers.Input(shape=(seq_len,), name='span_mask1', dtype=bool)
    span_mask2 = tf.keras.layers.Input(shape=(seq_len,), name='span_mask2', dtype=bool)

    input1 = {'input_ids': input_id1, 'attention_mask':input_masks_id1, "token_type_ids":input_token_id1}

    encoder = TFAutoModel.from_pretrained(path, from_pt=from_pt)
    encoder._name = 'bert_model'

    trainable = not freeze
    encoder.trainable = trainable

    output1 = encoder(input1)

    cls1 = output1[0][:, 0, :]
    
    avg_span1 = tf.math.reduce_mean(tf.ragged.boolean_mask(output1[0], span_mask1), 1)
    avg_span2 = tf.math.reduce_mean(tf.ragged.boolean_mask(output1[0], span_mask2), 1)
    
    concate = tf.keras.layers.Concatenate(axis=-1)([cls1, avg_span1, avg_span2])
    dropout = tf.keras.layers.Dropout(0.1)(concate)
    
    #classif
    classif = tf.keras.layers.Dense(output_label, name='classifier')(dropout)

    return tf.keras.Model(inputs=[input_id1, input_masks_id1, input_token_id1, span_mask1, span_mask2], outputs=classif)


from transformers import TFAutoModel
import tensorflow as tf
def build_classifier_model_const(path, seq_len=128, freeze=False, from_pt=False, output_label=2):
    input_id1 = tf.keras.layers.Input(shape=(seq_len,), name='input_id1', dtype='int32')
    input_masks_id1 = tf.keras.layers.Input(shape=(seq_len,), name='mask_id1', dtype='int32')
    input_token_id1 = tf.keras.layers.Input(shape=(seq_len,), name='token_type_id1', dtype='int32')
    
    span_mask1 = tf.keras.layers.Input(shape=(seq_len,), name='span_mask1', dtype=bool)

    input1 = {'input_ids': input_id1, 'attention_mask':input_masks_id1, "token_type_ids":input_token_id1}

    encoder = TFAutoModel.from_pretrained(path, from_pt=from_pt)
    encoder._name = 'bert_model'

    trainable = not freeze
    encoder.trainable = trainable

    output1 = encoder(input1)

    cls1 = output1[0][:, 0, :]
    
    avg_span1 = tf.math.reduce_mean(tf.ragged.boolean_mask(output1[0], span_mask1), 1)
    
    concate = tf.keras.layers.Concatenate(axis=-1)([cls1, avg_span1])
    dropout = tf.keras.layers.Dropout(0.1)(concate)
    
    #classif
    classif = tf.keras.layers.Dense(output_label, name='classifier')(dropout)

    return tf.keras.Model(inputs=[input_id1, input_masks_id1, input_token_id1, span_mask1], outputs=classif)


from transformers import TFAutoModelForTokenClassification
def build_classifier_model_token(path, from_pt=False, num_labels=1):
    return TFAutoModelForTokenClassification.from_pretrained(path, from_pt=from_pt, num_labels=num_labels)


def tokenize_copa(df, tokenizer, max_len=256):
    premises = df['premise'].tolist()
    questions = df['question'].tolist()
    choice1 = df['choice1'].tolist()
    choice2 = df['choice2'].tolist()
    label = df['label'].tolist()
    
    input_id, input_mask, input_token = [],[],[]
    
    for i in range(len(premises)):
        if questions[i] == 'cause':
            inputs = tokenizer([choice1[i], choice2[i]], [premises[i], premises[i]], add_special_tokens=True, pad_to_max_length=True, truncation=True,
                                return_attention_mask=True, return_token_type_ids=True,
                                max_length=max_len, return_tensors='tf')
        else:
            inputs = tokenizer([premises[i], premises[i]], [choice1[i], choice2[i]], add_special_tokens=True, pad_to_max_length=True, truncation=True,
                        return_attention_mask=True, return_token_type_ids=True,
                        max_length=max_len, return_tensors='tf')
                                
        input_id.append(inputs['input_ids'])
        input_mask.append(inputs['attention_mask'])
        input_token.append(inputs['token_type_ids'])
    
    return np.array(input_id), np.array(input_mask), np.array(input_token), np.array(tf.convert_to_tensor(label))


def tokenize_nli(df, tokenizer, max_len=128):
    premises = df['premise'].tolist()
    hyps = df['hypothesis'].tolist()
    input_id, input_mask, input_token = [],[],[]

    inputs = tokenizer(premises, hyps, add_special_tokens=True, pad_to_max_length=True, truncation=True,
                                        return_attention_mask=True, return_token_type_ids=True,
                                        return_tensors='tf', max_length=max_len)
    
    input_id = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_token = inputs['token_type_ids']

    return np.array(input_id), np.array(input_mask), np.array(input_token), np.array(df['label'])


def tokenize_g(df, tokenizer, data_type=None, max_len=128, input_type=None, model=None):
    texts = df['sentence'].tolist()
    options1 = df['option1'].tolist()
    options2 = df['option2'].tolist()
    
    input_id, input_mask, input_token = [],[],[]
    left_texts1, left_texts2, right_texts, texts1, texts2 = [], [], [], [], []
    options1_input, options2_input = [], []
    
    if data_type == None or data_type == 'winogrande':
        label = df['answer'].apply(int) - 1
    elif data_type == 'indograd':
        label = df['label'].apply(int) - 1
        
    for i in range(len(texts)):
        text = texts[i]
        option1 = options1[i]
        option2 = options2[i]

        sep_pos = text.find('_')
        left_text = text[:sep_pos+1]
        right_text = text[sep_pos+1:]

        left_text1 = left_text.replace('_', option1)
        left_text2 = left_text.replace('_', option2)

        text1 = left_text1 + right_text
        text2 = left_text2 + right_text

        left_texts1.append(left_text1)
        left_texts2.append(left_text2)
        right_texts.append(right_text)
        texts1.append(text1)
        texts2.append(text2)
        
    for i in range(len(left_texts1)):
        inputs = tokenizer([texts1[i], texts2[i]], add_special_tokens=True, pad_to_max_length=True, truncation=True,
                                    return_attention_mask=True, return_token_type_ids=True,
                                    max_length=max_len, return_tensors='tf')
        input_id.append(inputs['input_ids'])
        input_mask.append(inputs['attention_mask'])
        input_token.append(inputs['token_type_ids'])


    return np.array(input_id), np.array(input_mask), np.array(input_token), np.array(tf.convert_to_tensor(label))


def tokenize_coref(df, tokenizer, max_len=256):
    sentences = df['sentence'].apply(list).to_list()
    spans1 = df['span1'].to_list()
    spans2 = df['span2'].to_list()
    labels = df['label'].to_list()
    
    input_id, input_mask, input_token, span_mask1, span_mask2 = [],[],[],[],[]
    true_labels = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        span1 = spans1[i]
        span2 = spans2[i]
        
        inputs = tokenizer(sentence, add_special_tokens=True, pad_to_max_length=True, truncation=True,
                                            return_attention_mask=True, return_token_type_ids=True,
                                            return_tensors='tf', max_length=max_len, is_split_into_words=True
                          )        
        
        binary_mask1 = [False] * max_len
        binary_mask2 = [False] * max_len
            
        word_ids = inputs.word_ids()  # Map tokens to their respective word.
        idx1 = []
        idx2 = []
        for j in range(len(word_ids)):
            if word_ids[j] == span1[0] or word_ids[j] == span1[1]+1:
                idx1.append(j)
                
            if word_ids[j] == span2[0] or word_ids[j] == span2[1]+1:
                idx2.append(j)
        
        # input is beyond max_len
        if len(idx2) == 0 or len(idx1) == 0:
            continue
            
        for k in range(idx1[0], idx1[-1]+1):
            binary_mask1[k] = True

        for k in range(idx2[0], idx2[-1]+1):
            binary_mask2[k] = True

        span_mask1.append(binary_mask1)
        span_mask2.append(binary_mask2)
        
        input_id.append(inputs['input_ids'][0])
        input_mask.append(inputs['attention_mask'][0])
        input_token.append(inputs['token_type_ids'][0])
        true_labels.append(labels[i])

    return np.array(input_id), np.array(input_mask), np.array(input_token), np.array(span_mask1), np.array(span_mask2), np.array(true_labels)


def tokenize_const(df, tokenizer, max_len=256):
    sentences = df['sentence'].apply(list).to_list()
    spans = df['span'].to_list()
    labels = df['label'].to_list()
    
    input_id, input_mask, input_token, span_mask = [],[],[],[]
    true_labels = []
    
    for i in range(len(sentences)):
        sentence = sentences[i]
        span1 = spans[i]
        
        inputs = tokenizer(sentence, add_special_tokens=True, pad_to_max_length=True, truncation=True,
                                            return_attention_mask=True, return_token_type_ids=True,
                                            return_tensors='tf', max_length=max_len, is_split_into_words=True
                          )        
        
        binary_mask1 = [False] * max_len
            
        word_ids = inputs.word_ids()  # Map tokens to their respective word.
        idx1 = []
        for j in range(len(word_ids)):
            if word_ids[j] == span1[0] or word_ids[j] == span1[1]+1:
                idx1.append(j)
        
        # if beyond max sequence length
        if len(idx1) == 0:
            continue
            
        for k in range(idx1[0], idx1[-1]+1):
            binary_mask1[k] = True

        span_mask.append(binary_mask1)
        
        input_id.append(inputs['input_ids'][0])
        input_mask.append(inputs['attention_mask'][0])
        input_token.append(inputs['token_type_ids'][0])
        true_labels.append(labels[i])

    return np.array(input_id), np.array(input_mask), np.array(input_token), np.array(span_mask), np.array(true_labels)


# https://huggingface.co/docs/transformers/tasks/token_classification
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    
    if 'pos_tags' in examples:
        tag = f"pos_tags"
    elif 'ner_tags' in examples:
        tag = f"ner_tags"
    
    for i, label in enumerate(examples[tag]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

