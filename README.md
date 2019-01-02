# 自动完形填空系统构建
## 问题描述：
   在语义连贯的句子中去掉一个词语,形成空格,要求在给出的对应备选答案中,系统自动选出一个最佳的答案,使语句恢复完整。
## 相关语料
* Training data:未标注训练语料,供同学选择使用。同学也可根据需要自行选择其他语料,但需要在在实验报告中注明所使用训练语料的规模、来源及性质。
* Development set:提供一份含有 240 句话的语料及答案,供同学自行测试结果,根据结果调整优化自己的算法。
* Test set:提供一份含有 800 句话的测试语料,每句话有一个空格和 5 个备选答案。该语料不提供答案,同学提交测试结果,由助教统一评测。
## 评测方法
准确率=正确填空句子的个数/全部句子的个数
## 题目要求
要求同学根据自己设计训练得到的系统,对测试语料进行预测,对每句话提供一个系
统认为正确的选项。
本作业无统一标准方法,同学可自行设计模型,鼓励同学积极创新。
提示:模型的构建可以简单也可以复杂。例如,可以基于 n 元模型建立一个朴素的系
统;也可以引入词性、句法树等;也可以使用神经网络等其他方法;可以使用自行搜集到的
词典或者规则作为辅助。当然不限于这些方法,鼓励创新。
## 作业要求
* 可分组进行，但每个小组的规模不能超过 2 人(即≤ 2)
* 实现相关程序，可用 c/c++、Python 以及 java 语言完成。可参考网上源代码，但
必须重新实现，要求程序代码完整，有必要的说明文档和 Makefile 等文件；
*提供测试语料的预测结果，输出文件以“题目号+选项+英文单词”形式输出，中
间用空格或制表符间隔，每个答案占一行。例如：
* 1 choice1 answer1
* …… ……
* …… ……
* 800 choice 800 answer800
* 撰写实验报告以及 PPT。实验报告以小论文的形式，要有必要的参考文献等信息，
将使用的方法讲解清楚；PPT 用于在课堂上报告实验成果；
* 将预测答案、实验报告、PPT 及源程序提交到助教用以评分。
* 作业提交截止时间：2018 年 6 月 6 日。

# conditional_GAN_model
## 1.总体思路
借鉴pix2pix model的思想，假设一个句子为S，挖去S中的一个词W，S去掉W的部分记为C（condition）。现在有一个生成器G，把C输入G，输出一个fake word，记为f_W，把C和f_W组成一个新的句子记为f_S。然后把真实的句子S，和fake的句子f_S分别输入一个分类器D中，D输出一个概率p(x)，表示输入x为真实的句子的概率。首先，我们有第一代的生成器G1和第一代的分类器D1，G1根据条件C生成fake句子f_S，训练D1让其能够分辨f_S和S。然后训练第二代的生成器G2，令其生成的句子f_S可以骗过第一代的分类器D1。然后再训练第二代的分类器D2，令其能够分辨G2生成的f_S和真实的S。这样循环往复，我们便有了n代的生成器Gn，n代的分类器Dn。Dn应该有强大的分辨能力，能分辨出真实分布里的句子和捏造的句子。用Dn去做test，将每个选项带入空格句子形成5个不同的句子，分别输入Dn中，输出概率最大的为真实选项。
## 2.算法主体
初始化G和D的参数Ɵg和Ɵd。在每个训练步骤中：
* 1.从训练集中抽样m个条件C，C1...Cm，和与之对应的m个真实句子S1...Sm。
* 2.C1...Cm输入G中，得到生的m个句子f_S1...f_S2。
* 3.固定Ɵg，利用梯度下降算法更新Ɵd，使下式最大化：
1/m Σlog(D(Si))+1/mΣlog(1-D(f_Si))
* 4.更新Ɵg，使下式最小化：1/mΣlog(1-D(f_Si))
## 3.实现
首先要预训练出embedding张量，供整个模型直接使用。
训练集我是这样做的：先从文本中清理出有效的句子。然后对每个句子扣掉一个词W，然后句子剩余部分C，和W，以及W在句子中的位置信息N放在一行（C,W,N）。  
至于生成器和分类器的具体结构，可以有不同的方案，这里应该也是实验成功或失败的关键。

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

我构造的生成器要用到rnn cell，上面的代码是产生不同类型或单层或多层的rnn实例代码片段。

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
          
接着是生成器的构造，它接受条件C（词向量张量）和单词W的位置信息，生成fake的单词。这里生成的是词向量，而不是生成词的离散的id。因为生成词的id是一个抽样过程，无法使用反向传播算法更新参数。因此这里选择直接生成词向量。
接着要把生成的词向量，按照位置信息和C拼接起来组成f_S,输入到分类器D中。
下面是分类器部分的代码：

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

它接受S或f_S，输出一个实数，表示输入是真实的概率。然后是loss和train_op部分的代码如下：

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

其中，fake_predict和real_predict分别是输入为f_S何S时，D的输出。Loss函数借鉴了pix2pix模型里的loss。EPS是一个极小的常数。
最后是训练部分代码：

      for _ in range(6):
           _,d_loss = sess.run((discrim_train,discrim_loss),feed_dict={conditions:batch_data,d_inputs_real:d_input_real_,loc:batch_location})
           print(d_loss)
      _,g_loss = sess.run((gen_train,gen_loss),feed_dict={conditions:batch_data,loc:batch_location})
      
 我选择的是更新k次D的参数，然后更新一步G的参数。
## 4.问题
首先最大的问题是D和G的具体结构的选择。G的部分尝试过单层或多层的单向或多向的lstm网络，D的部分尝试过lstm或卷积网络。但是效果都不好，训练的过程中经常遇到real_predict和fake_predict同时趋于1或0，loss总是一个固定的常数不变化。我认为其中的问题可能是G和D不匹配，如果G太强，或太弱，都可能会造成D无法分辨真实和伪造。
但是也有可能是对与语言这种离散的数据来说，GAN的思路很难有效果。因为根据我的实际经验，在用GAN做图片生成问题的时候，loss也总是趋于一个固定的数，但是生成器的效果确实是逐渐变好的。
或许其中的问题是这样：当用G生成图片的时候，因为图片的像素值是连续的，所有它生成的总是一张图片。但是当用G生成词向量时，它生成的词向量可能在数据集中根本就不存在。
总之这个问题还有待我参考更多的文献。
然后遇到的问题基本上就是硬件问题了。首先，训练文档中每句话扣除一些词组成不同的训练数据，直接导致训练文本达到几个G，然后我的内存就爆了。后面用实验的服务器，因为可供我使用的时间有限，因此没有尝试更多的实验。
## 参考文献
* [1]IanJ.Goodfellow,Jean Pouget-Abadiey,Mehdi Mirza,Bing Xu,David Warde-Farley,Sherjil Ozairz,Aar.Generative Adversarial Nets.NIPS2014.
* [2]Richard Nocky,Zac Cranko,Aditya Krishna Menon,Lizhen Qu,Robert C. Williamson.f-GANs in an Information Geometric Nutshell.arXiv:1707.04385
* [3]Phillip Isola,Jun-Yan Zhu,Tinghui Zhou,Alexei A.Efros.Image-to-Image Translation with Conditional Adversarial Networks.arXiv:1611.07004
* [4]Mehdi Mirza,Simon Osindero.Conditional Generative Adversarial Nets.arXiv:1411.1784
