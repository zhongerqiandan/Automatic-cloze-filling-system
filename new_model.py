import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json
import linecache
import re
from copy import deepcopy



vocabulary_size = 50000
embedding_size = 512
lstm_size = 512
max_length = 35
learning_rate = 0.000002
beta = 0.5
batch_size = 20
epochs = 6
checkpoint_dir = 'train/'
EPS = 1e-12

lstm_layers = 3

with open('word2vect/word2id.json') as f:
    for line in f:
        word2id_dict = json.loads(line)

with open('word2vect/id2word.json') as f:
    for line in f:
        id2word_dict = json.loads(line)



filename = 'training_data_output.txt'
p = 1
def batch_generator(batch_size):
    global p
    #print('training lines %d -- %d'%(p,p + batch_size))
    q = p
    batch_sentences = []
    for i in range(p,p + batch_size):
        line = linecache.getline(filename,i)
        if line:
            batch_sentences.append(line)
            q += 1
        else:
            q = 1
            break
    p = q
    if len(batch_sentences) != batch_size:
        for i in range(p,p + batch_size - len(batch_sentences)):
            line = linecache.getline(filename,i)
            batch_sentences.append(line)
            q += 1
        p = q
        
    #print(batch_sentences)
    return batch_sentences

def development():
    i = 0
    data = {}
    choice = {}
    sentence = []
    question_num = 801
    with open('development_set.txt') as f:
        for line in f:
            if line != '\n':
                if i % 6 == 0:
                    for word in re.sub(r'[^\w\s]','',line.strip()).split()[1:]:
                        if word != '_____':
                            if word.lower() in word2id_dict:
                                sentence.append(word2id_dict[word.lower()])
                            else:
                                sentence.append(word2id_dict['UNK'])
                                print(question_num,word)
                        else:
                            sentence.append(-1)

                if i % 6 == 1:
                    a = deepcopy(sentence)
                    for j in range(len(a)):
                        if a[j] == -1:
                            tmp = j
                    if re.sub(r'[^\w\s]','',line.strip()).split()[1].lower() in word2id_dict:
                        a[tmp] = word2id_dict[re.sub(r'[^\w\s]','',line.strip()).split()[1].lower()]
                    else:
                        a[tmp] = word2id_dict['UNK']
                        print(question_num,re.sub(r'[^\w\s]','',line.strip()).split()[1],line)
                    choice['a'] = a                    

                if i % 6 == 2:
                    b = deepcopy(sentence)
                    for j in range(len(b)):
                        if b[j] == -1:
                            tmp = j
                    if re.sub(r'[^\w\s]','',line.strip()).split()[1].lower() in word2id_dict:
                        b[tmp] = word2id_dict[re.sub(r'[^\w\s]','',line.strip()).split()[1].lower()]
                    else:
                        b[tmp] = word2id_dict['UNK']
                        print(question_num,re.sub(r'[^\w\s]','',line.strip()).split()[1],line)
                    choice['b'] = b

                if i % 6 == 3:
                    c = deepcopy(sentence)
                    for j in range(len(c)):
                        if c[j] == -1:
                            tmp = j
                    if re.sub(r'[^\w\s]','',line.strip()).split()[1].lower() in word2id_dict:
                        c[tmp] = word2id_dict[re.sub(r'[^\w\s]','',line.strip()).split()[1].lower()]
                    else:
                        c[tmp] = word2id_dict['UNK']
                        print(question_num,re.sub(r'[^\w\s]','',line.strip()).split()[1],line)
                    choice['c'] = c

                if i % 6 == 4:
                    d = deepcopy(sentence)
                    for j in range(len(d)):
                        if d[j] == -1:
                            tmp = j
                    if re.sub(r'[^\w\s]','',line.strip()).split()[1].lower() in word2id_dict:
                        d[tmp] = word2id_dict[re.sub(r'[^\w\s]','',line.strip()).split()[1].lower()]
                    else:
                        d[tmp] = word2id_dict['UNK']
                        print(question_num,re.sub(r'[^\w\s]','',line.strip()).split()[1],line)
                    choice['d'] = d

                if i % 6 == 5:
                    e = deepcopy(sentence)
                    for j in range(len(e)):
                        if e[j] == -1:
                            tmp = j
                    if re.sub(r'[^\w\s]','',line.strip()).split()[1].lower() in word2id_dict:
                        e[tmp] = word2id_dict[re.sub(r'[^\w\s]','',line.strip()).split()[1].lower()]
                    else:
                        e[tmp] = word2id_dict['UNK']
                        print(question_num,re.sub(r'[^\w\s]','',line.strip()).split()[1],line)
                    choice['e'] = e

                    data[question_num] = deepcopy(choice)
                    question_num += 1
                    choice = {}
                    sentence =[]
                i += 1

    print(len(data))   
    return data

def development_answer():
    answer = []
    with open('development_set_answers.txt') as f:
        for line in f:
#             print(re.sub(r'[^\w\s]','',line.strip()).split())
            answer.append(re.sub(r'[^\w\s]','',line.strip()).split()[1])
    return np.array(answer)


def concat(conditions,choices,location):
    lst = []
    for i in range(len(location)):
        
        x = conditions[i:i+1,:location[i],:]
        y = conditions[i:i+1,location[i]:,:]
        z = choices[i:i+1,:,:]
        lst.append(np.concatenate((x,z,y),1))
    return np.concatenate(lst,0)
    




development_data = development()
development_answer = development_answer()
'''
g_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
g_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
def creat_generator(conditions,loc):
    with tf.variable_scope('generator') as scope:
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(g_lstm_cell_fw,g_lstm_cell_bw,conditions,dtype=tf.float32)
        outputs_concated = tf.concat(outputs,2)
        lstm_outputs = tf.concat([tf.expand_dims(outputs_concated[i,loc[i],:],0) for i in range(batch_size)],0)
        lstm_outputs_bn = tf.layers.batch_normalization(lstm_outputs,training=True)
        #lstm_outputs [batch_size,2*lstm_size]
        #w_fc_1 = tf.get_variable(name='g_w_fc_1',initializer=tf.truncated_normal(shape=[2 * lstm_size,lstm_size],dtype=tf.float32))
        #b_fc_1 = tf.get_variable(name='g_b_fc_1',initializer=tf.zeros([lstm_size]))
        #fc_outputs_1 = tf.nn.relu(tf.matmul(lstm_outputs,w_fc_1) + b_fc_1)
        #w_fc_2 = tf.get_variable(name='g_w_fc_2',initializer=tf.truncated_normal(shape=[lstm_size,embedding_size],dtype=tf.float32))
        #b_fc_2 = tf.get_variable(name='g_b_fc_2',initializer=tf.zeros([embedding_size]))
        w_fc = tf.get_variable(name='g_w_fc_',initializer=tf.truncated_normal(shape=[2 * lstm_size,embedding_size],dtype=tf.float32))
        b_fc = tf.get_variable(name='g_b_fc_',initializer=tf.zeros([embedding_size]))
        logits = tf.matmul(lstm_outputs_bn,w_fc) + b_fc
        norm = tf.sqrt(tf.reduce_sum(tf.square(logits),axis=1,keep_dims=True))
    return logits
'''
def single_cell_fn(num_units,forget_bias,dropout,residual_connection=False):
  	#Create an instance of a single RNN cell.
    
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units,forget_bias=forget_bias,layer_norm=True)
    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell,input_keep_prob=(1.0 - dropout))

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
   

    return single_cell



def cell_list(num_units,num_layers,forget_bias,dropout,num_residual_layers=0):
    #Create a list of RNN cells.
    cell_list = []
    for i in range(num_layers):
    
        single_cell = single_cell_fn(num_units=num_units,forget_bias=forget_bias,dropout=dropout,residual_connection=(i >= num_layers - num_residual_layers))
        cell_list.append(single_cell)

    return cell_list

def create_rnn_cell(num_units,num_layers,forget_bias,dropout,num_residual_layers=0):
    cell_list_ = cell_list(num_units=num_units,num_layers=num_layers,num_residual_layers=num_residual_layers,forget_bias=forget_bias,dropout=dropout)
    if len(cell_list_) == 1:  # Single layer.
        return cell_list_[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list_)

def creat_generator(conditions,loc):
    with tf.variable_scope('generator') as scope:
        cell_fw = create_rnn_cell(num_units=lstm_size,num_layers=lstm_layers,forget_bias=1.0,dropout=0.3)
        cell_bw = create_rnn_cell(num_units=lstm_size,num_layers=lstm_layers,forget_bias=1.0,dropout=0.3)
        outputs,states = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,conditions,dtype=tf.float32)
        outputs_concated = tf.concat(outputs,2)
        lstm_outputs = tf.concat([tf.expand_dims(outputs_concated[i,loc[i],:],0) for i in range(batch_size)],0)
        lstm_outputs_bn = tf.layers.batch_normalization(lstm_outputs,training=True)
        #lstm_outputs [batch_size,2*lstm_size]
        w_fc = tf.get_variable(name='g_w_fc_',initializer=tf.truncated_normal(shape=[2 * lstm_size,embedding_size],dtype=tf.float32))
        b_fc = tf.get_variable(name='g_b_fc_',initializer=tf.zeros([embedding_size]))
        logits = tf.matmul(lstm_outputs_bn,w_fc) + b_fc
        norm = tf.sqrt(tf.reduce_sum(tf.square(logits),axis=1,keep_dims=True))
    return logits / norm




'''
g_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
def creat_generator(conditions):
    
    with tf.variable_scope('generator') as scope:
#         h0 = np.random.randn(batch_size,lstm_size).astype(np.float32)
        h0 = g_lstm_cell.zero_state(batch_size,tf.float32)
        outputs,final_states = tf.nn.dynamic_rnn(g_lstm_cell,inputs=conditions,initial_state=h0,scope=scope)
        state = tf.reshape(tf.slice(outputs,[0,max_length-2,0],[batch_size,1,lstm_size]),[batch_size,lstm_size])
        #w_fc_1 = tf.get_variable(name='g_w_fc_1',initializer=tf.truncated_normal(shape=[lstm_size,128],dtype=tf.float32))
        #fc_output_1 = tf.matmul(state,w_fc_1)
        #w_fc_2 = tf.get_variable(name='g_w_fc_2',initializer=tf.truncated_normal(shape=[128,embedding_size],dtype=tf.float32))
        w_fc = tf.get_variable(name='g_w_fc',initializer=tf.truncated_normal(shape=[lstm_size,embedding_size],dtype=tf.float32))
        b_fc = tf.get_variable(name='g_b_fc',initializer=tf.zeros([lstm_size]))
        logits = tf.matmul(state,w_fc) + b_fc
        norm = tf.sqrt(tf.reduce_sum(tf.square(logits),axis=1,keep_dims=True))
        
#         fake_ids = tf.argmax(logits,1)
        
    return logits / norm
'''
'''
d_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)

def creat_discreminator(inputs):
    with tf.variable_scope(name_or_scope='discriminator') as scope:
        
        h0 = d_lstm_cell.zero_state(batch_size,tf.float32)
        outputs,final_states = tf.nn.dynamic_rnn(d_lstm_cell,inputs=inputs,initial_state=h0,scope=scope)
        state = tf.reshape(tf.slice(outputs,[0,max_length-1,0],[batch_size,1,lstm_size]),[batch_size,1,lstm_size,1])
        #w_fc_1 = tf.get_variable(name='d_w_fc1',initializer=tf.truncated_normal(shape=[lstm_size,128],dtype=tf.float32))
        #fc_output_1 = tf.matmul(state,w_fc_1)
        #w_fc_2 = tf.get_variable(name='d_w_fc2',initializer=tf.truncated_normal(shape=[128,1],dtype=tf.float32))
        kernel = tf.get_variable(name='d_w_conv',initializer=tf.truncated_normal(shape=[1,5,1,6],dtype=tf.float32))
        bias = tf.get_variable(name='d_b_conv',initializer=tf.constant(0,tf.float32,[6]))
        conv2d = tf.nn.conv2d(state,kernel,[1,1,1,1],padding='SAME') + bias
        h_conv2d = tf.nn.relu(conv2d)
        h_conv2d_flat = tf.reshape(h_conv2d,[-1,lstm_size * 6])
        w_fc = tf.get_variable(name='d_w_fc',initializer=tf.truncated_normal(shape=[lstm_size * 6,1],dtype=tf.float32))
        b_fc = tf.get_variable(name='d_b_fc',initializer=tf.zeros([1]))
        
    return tf.sigmoid(tf.matmul(h_conv2d_flat,w_fc) + b_fc)
''' 
def creat_discreminator(inputs,is_training):
    with tf.variable_scope(name_or_scope='discriminator') as scope:
        inputs_reshaped = tf.reshape(inputs,[-1,max_length,embedding_size,1])
        kernel_1 = tf.get_variable(name='d_w_conv_1',initializer=tf.truncated_normal(shape=[5,embedding_size,1,60],dtype=tf.float32))
        bias_1 = tf.get_variable(name='d_b_conv_1',initializer=tf.constant(0,tf.float32,[60]))
        conv2d_1 = tf.nn.conv2d(inputs_reshaped,kernel_1,[1,1,embedding_size,1],padding='SAME') + bias_1
        h_conv2d_1 = tf.nn.relu(conv2d_1)
        h_conv2d_1_bn = tf.layers.batch_normalization(h_conv2d_1,training=is_training)
        h_conv2d_1_reshaped = tf.reshape(h_conv2d_1_bn,[-1,max_length * 60])
        '''
        kernel_2 = tf.get_variable(name='d_w_conv_2',initializer=tf.truncated_normal(shape=[3,16,1,36],dtype=tf.float32))
        bias_2 = tf.get_variable(name='d_b_conv_2',initializer=tf.constant(0,tf.float32,[36]))
        conv2d_2 = tf.nn.conv2d(h_conv2d_1_reshaped,kernel_2,[1,1,16,1],padding='SAME') + bias_2
        h_conv2d_2 = tf.nn.relu(conv2d_2)
        h_conv2d_2_flat = tf.reshape(h_conv2d_2,[-1,max_length * 36])
        '''
        
        w_fc_1 = tf.get_variable(name='d_w_fc_1',initializer=tf.truncated_normal(shape=[max_length * 60,500],dtype=tf.float32))
        b_fc_1 = tf.get_variable(name='d_b_fc_1',initializer=tf.zeros([500]))
        outputs_fc_1 = tf.nn.relu(tf.matmul(h_conv2d_1_reshaped,w_fc_1) + b_fc_1)
        outputs_fc_1_bn = tf.layers.batch_normalization(outputs_fc_1,training=is_training)
   
        w_fc_2 = tf.get_variable(name='d_w_fc_2',initializer=tf.truncated_normal(shape=[500,100],dtype=tf.float32))
        b_fc_2 = tf.get_variable(name='d_b_fc_2',initializer=tf.zeros([100]))
        outputs_fc_2 = tf.nn.relu(tf.matmul(outputs_fc_1_bn,w_fc_2) + b_fc_2)
        outputs_fc_2_bn = tf.layers.batch_normalization(outputs_fc_2,training=is_training)

        w_fc_3 = tf.get_variable(name='d_w_fc_3',initializer=tf.truncated_normal(shape=[100,1],dtype=tf.float32))
        b_fc_3 = tf.get_variable(name='d_b_fc_3',initializer=tf.zeros([1]))
    return tf.sigmoid(tf.matmul(outputs_fc_2_bn,w_fc_3) + b_fc_3)

    


embeddings = tf.get_variable(name='normalized_embeddings',shape=[vocabulary_size,embedding_size])

words_id_list = tf.placeholder(dtype=tf.int32,shape=[1,None])
choice_id_list = tf.placeholder(dtype=tf.int32,shape=[1,None])
words_vector_conditions = tf.nn.embedding_lookup(embeddings,words_id_list)
words_vector_conditions_paded = tf.pad(tf.squeeze(words_vector_conditions),[[0,max_length - 1 - tf.shape(words_id_list)[1]],[0,0]])
        #words_vector_conditions_paded_reshaped = tf.reshape(words_vector_conditions_paded,shape=[1,max_length - 1,embedding_size])
choice_vector = tf.squeeze(tf.nn.embedding_lookup(embeddings,choice_id_list))

developmen_words_vector_paded = tf.pad(tf.squeeze(words_vector_conditions),[[0,max_length - tf.shape(words_id_list)[1]],[0,0]])

conditions = tf.placeholder(tf.float32,shape=[None,max_length - 1,embedding_size])
loc = tf.placeholder(tf.int32,[batch_size])
logits = creat_generator(conditions,loc)
fake_words_vector = tf.reshape(logits,[batch_size,1,embedding_size])


d_inputs_real = tf.placeholder(tf.float32,shape=[None,max_length,embedding_size])
dev_inputs = tf.placeholder(tf.float32,shape=[5,max_length,embedding_size])
#loc = tf.placeholder(tf.int32,[batch_size])
# sentence_num = tf.placeholder(tf.int32,[batch_size])
# sentence_num = np.array([i for i in range(batch_size)])
# tmp_tensor_lst = [tf.Variable(trainable=False,initial_value=tf.truncated_normal([1,35,512]))] * batch_size

# for i in range(batch_size):
#     tf.assign(tmp_tensor_lst[i],tf.concat([tf.slice(conditions,[sentence_num[i],0,0],[1,loc[i],embedding_size]),tf.expand_dims(fake_words_vector[i],0),tf.slice(conditions,[sentence_num[i],loc[i],0],[1,max_length - 1 - loc[i],embedding_size])],1))

d_inputs_fake = tf.concat([tf.concat([tf.slice(conditions,[i,0,0],[1,loc[i],embedding_size]),tf.slice(fake_words_vector,[i,0,0],[1,1,embedding_size]),tf.slice(conditions,[i,loc[i],0],[1,max_length - 1 - loc[i],embedding_size])],1) for i in range(batch_size)],0)


optim = tf.train.AdamOptimizer(learning_rate)
gen_tvars0 = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
op = optim.minimize(fake_words_vector,var_list=[gen_tvars0])


# d_inputs_fake = tf.concat([conditions,fake_words_vector],1)
with tf.variable_scope(name_or_scope='discriminator',reuse=None):
    real_predict = creat_discreminator(d_inputs_real,True)
    #answer_id = tf.argmax(real_predict,dimension=0)
with tf.variable_scope(name_or_scope='discriminator',reuse=True):
    fake_predict = creat_discreminator(d_inputs_fake,True)
    dev_predict = creat_discreminator(dev_inputs,False)
    answer_id = tf.argmax(dev_predict,dimension=0)
# answer_id = tf.argmax(real_predict,dimension=0)



discrim_loss = tf.reduce_mean(-(tf.log(real_predict + EPS) + tf.log(1 - fake_predict + EPS)))
#gen_loss = tf.reduce_mean(-fake_predict)
gen_loss = tf.reduce_mean(-tf.log(fake_predict + EPS))
# lst = [var for var in tf.trainable_variables()]
discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
discrim_optim = tf.train.AdamOptimizer(learning_rate)
#discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
#discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
discrim_train = discrim_optim.minimize(discrim_loss,var_list=discrim_tvars)

gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#print(gen_tvars)
# gen_tvars = [var for var in tf.trainable_variables()]
gen_optim = tf.train.AdamOptimizer(learning_rate)
#gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
#gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
gen_train = gen_optim.minimize(gen_loss,var_list=gen_tvars)


# global_step = tf.contrib.framework.get_or_create_global_step()
# incr_global_step = tf.assign(global_step, global_step+1)


init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=[embeddings])
saver0 = tf.train.Saver()       
        



with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'word2vect/train/norm_embedding-5000000')
    #saver0.restore(sess,checkpoint_dir + 'model.ckpt')
    for x in development_data:
        for y in development_data[x]:
            if len(development_data[x][y]) > 35:
                development_data[x][y] = development_data[x][y][:35]
            development_data[x][y] = sess.run(developmen_words_vector_paded,feed_dict={words_id_list:np.array(development_data[x][y]).reshape(1,-1)})
            development_data[x][y] = development_data[x][y].reshape(1,max_length,embedding_size)
    development_data_concat = {}
    for x in development_data:
        development_data_concat[x] = np.concatenate((development_data[x]['a'],development_data[x]['b'],development_data[x]['c'],development_data[x]['d'],development_data[x]['e']))
        #print(development_data_concat[x].shape)
    print('Development data prepared.')
    for epoch in range(epochs):
        for step in range(26000000 // batch_size):
            #batch_data,batch_choices,batch_location = batch_generator(6)
            #print(batch_data.shape,batch_choices.shape)
            #print(batch_location)
            try:
                batch_sentences = batch_generator(batch_size)
                batch_data = []
                batch_choices = []
                batch_location = []
                for sentence in batch_sentences:
                    words_id_list_ = []
                    choice_id_list_ = []
                    for word in sentence.split()[:len(sentence.split()) - 2]:
                        if word in word2id_dict:
    #                         print('a')
                            words_id_list_.append(word2id_dict[word])
                        else:
                            #print('error')
                            #print(word)
                            words_id_list_.append(word2id_dict['UNK'])

        #             if len(words_id_list_) > 34:
        #                 words_id_list_ = words_id_list_[:34]
                    if sentence.split()[-2] in word2id_dict:
    #                     print('a')
                        choice_id_list_.append(word2id_dict[sentence.split()[-2]])
                    else:
                        #print('error')
                        #print(sentence.split()[-2])
                        choice_id_list_.append(word2id_dict['UNK'])
                    batch_location.append(int(sentence.split()[-1]))
                    batch_data.append(sess.run(words_vector_conditions_paded,feed_dict={words_id_list:np.array(words_id_list_).reshape(1,-1)}))
                    batch_choices.append(sess.run(choice_vector,feed_dict = {choice_id_list:np.array(choice_id_list_).reshape(1,-1)}))

                batch_data = np.array(batch_data)
                batch_choices = np.array(batch_choices).reshape(batch_size,1,embedding_size)
                batch_location = np.array(batch_location)
        #         fake_words_vector_ = sess.run(fake_words_vector,feed_dict={conditions:batch_data})
        #         d_input_fake_ = np.concatenate((batch_data,fake_words_vector_),1)
        #         print(batch_data.shape)
        #         print(batch_choices.shape)
        #         print(batch_location)
                d_input_real_ = concat(batch_data,batch_choices,batch_location)
                for _ in range(6):
                    _,d_loss = sess.run((discrim_train,discrim_loss),feed_dict={conditions:batch_data,d_inputs_real:d_input_real_,loc:batch_location})
                    print(d_loss)
                _,g_loss = sess.run((gen_train,gen_loss),feed_dict={conditions:batch_data,loc:batch_location})
                print('after step %d,d_loss is %f,g_loss is %f'%(step,d_loss,g_loss))
        #         print(sess.run(gen_tvars))
                if step % 1000 == 0:
                    saver0.save(sess, checkpoint_dir + 'model.ckpt')
                    answers = []
                    for x in development_data_concat:
                        answer_id_ = sess.run(answer_id,feed_dict={dev_inputs:development_data_concat[x]})
                        if answer_id_[0] == 0:
                            answers.append('a')
                        if answer_id_[0] == 1:
                            answers.append('b')
                        if answer_id_[0] == 2:
                            answers.append('c')
                        if answer_id_[0] == 3:
                            answers.append('d')
                        if answer_id_[0] == 4:
                            answers.append('e')
                    answers = np.array(answers)
                    accurate = np.sum(answers == development_answer) / 240.0
                    print('After %d step,accurate is %f'%(step,accurate))
            except:
                pass
            
                

