#*-* coding:utf-8 *-*
#

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import re
import json
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from sklearn import metrics
from TextCNN import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
dic_path='data'
dat_path='data/preprocess'
tf.flags.DEFINE_string("train_data", "{}/train.seg".format(dat_path), "Data source for the train data.")
tf.flags.DEFINE_string("test_data", "{}/dev.seg".format(dat_path), "Data source for the test data.")
tf.flags.DEFINE_string("label_dict", "{}/label_dict".format(dic_path), "Data source for the label_dict data.")
tf.flags.DEFINE_string("word2vec", "{}/embedding/voc.txt".format(dic_path), "Data source for the word2vec data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_pos_dim", 50, "Dimensionality of position embedding (default: 50)")
tf.flags.DEFINE_integer("embedding_entity_dim", 50, "Dimensionality of entity embedding (default: 50)")
tf.flags.DEFINE_integer("embedding_sf_dim", 10, "Dimensionality of entity embedding (default: 10)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 150, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_string("title_filter_sizes", "1,2", "Comma-separated filter sizes (default: '1,2')")
tf.flags.DEFINE_integer("title_num_filters", 50, "Number of filters per filter size (default: 50)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.005, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("sent_length", 21, "the fixed window size (default: 13),奇数")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 150, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("train_mode", "train_from_scratch", "Train mode (default: train_from_scratch)")
tf.flags.DEFINE_string("checkpoint_dir", "../data/runs/1517922440/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")
#x_train, title_train, y_train, train_beg, train_end, train_entity, train_sf = \
y_train, x_train, train_entity = \
        data_helpers.load_data_and_labels(
                FLAGS.train_data, class_num=10)
#x_dev, title_dev, y_dev, dev_beg, dev_end, dev_entity, dev_sf = \
y_dev, x_dev, dev_entity = \
        data_helpers.load_data_and_labels(
                FLAGS.test_data, class_num=10)

# # 确定窗口大小n（n=13）
# # ==================================================
# num_0 = 0
# num_1 = 0
# for i in xrange(len(train_beg)):
#     if train_beg[i] >= 6:
#         num_0 += 1
#     if len(x_train[i])-1-train_end[i] >= 6:
#         num_1 += 1
# print num_0
# print num_1
# print len(train_beg)

x_train, train_pos, train_entity = \
        data_helpers.data_trim(x_train, train_entity, FLAGS.sent_length)
x_dev, dev_pos, dev_entity = \
        data_helpers.trigger2mid(x_dev, dev_entity, FLAGS.sent_length)

# 装载词向量
def loadWord2Vec(filename):
    '''
    '''
    print "loaded word2vec..."
    vocab = []
    embd = []
    fr = open(filename, 'r')
    line = fr.readline().decode('utf-8').strip()
    word_dim = int(line.split(' ')[1])
    vocab.append("<UNK>")
    embd.append([0] * word_dim)
    for line in fr:
        row = line.strip().split(' ')
        vocab.append(row[0].decode())
        embd.append(row[1:])
    fr.close()
    print "word2vec load over..."
    return vocab, embd

vocab, embd = loadWord2Vec(FLAGS.word2vec)
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)

# Build vocabulary
TOKENIZER_RE = re.compile(r"[^\s]+", re.UNICODE)
def tokenizer(iterator):
  """Tokenizer generator.

  Args:
    iterator: Input iterator with strings.

  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
      yield TOKENIZER_RE.findall(value)

# 把文档id化
max_x_length = max([len(x) for x in x_train])
# 文档最大长度max_x_length， 采用tokenizer分词
vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_x_length, tokenizer_fn=tokenizer)
pretrain = vocab_processor.fit(vocab)   # 创建词表
# 训练集id化
x_train = np.array(list(vocab_processor.transform([" ".join(x) for x in x_train])))
# 开发集id化
x_dev = np.array(list(vocab_processor.transform([" ".join(x) for x in x_dev])))

# 统计词表覆盖率
# print len(x_vocab)
# num = 0
# for i in x_vocab:
#     if vocab_processor.vocabulary_.get(i) == 0:
#         num += 1
#         print i
# print num
# print x_train
# print len(vocab_processor.vocabulary_)

# print json.dumps(title_train, ensure_ascii=False)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_x_length,
            title_length=max_title_length,
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dim,
            embedding_size_pos=FLAGS.embedding_pos_dim,
            embedding_size_entity=FLAGS.embedding_entity_dim,
            embedding_size_sf=FLAGS.embedding_sf_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            title_filter_sizes=list(map(int, FLAGS.title_filter_sizes.split(","))),
            title_num_filters=FLAGS.title_num_filters,
            sent_length=FLAGS.sent_length,
            entity_length = 5,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        timestamp = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        title_vocab_processor.save(os.path.join(out_dir, "title_vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        if FLAGS.train_mode == "train":
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)  # 通过检查点文件锁定最新的模型
            # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
            print("Continue train from the model {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        def train_step(x_batch, y_batch, title_batch, train_pos, train_entity, train_sf):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_title: title_batch,
              cnn.input_pos: train_pos,
              cnn.input_entity: train_entity,
              cnn.input_sf: train_sf,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.embedding_placeholder: embedding
            }
            _, step, summaries, _, loss, accuracy, pred= sess.run( \
                [train_op, global_step \
                    , train_summary_op, cnn.embedding_init \
                    , cnn.loss, cnn.accuracy \
                    , cnn.predictions \
                ], \
                feed_dict)
            y_true = np.argmax(y_batch, 1)
            acc = metrics.precision_score(y_true, pred, average="micro")
            recall = metrics.recall_score(y_true, pred, average="micro")
            f1_score = metrics.f1_score(y_true, pred, average="micro")
            auc = metrics.roc_auc_score(y_true, pred, average="micro")
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, auc {:g}, acc {:g}, recal {:g}, f1 {:g}".format( \
                    time_str, step, loss, auc, acc, recall, f1_score))
            sys.stdout.flush()
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, title_batch, dev_pos, dev_entity, dev_sf, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.input_title: title_batch,
              cnn.input_pos: dev_pos,
              cnn.input_entity: dev_entity,
              cnn.input_sf: dev_sf,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, pred = sess.run( \
                [global_step, dev_summary_op \
                        , cnn.loss, cnn.accuracy\
                        , cnn.predictions \
                        ], \
                feed_dict)
            y_true = np.argmax(y_batch, 1)
            acc = metrics.precision_score(y_true, pred, average="micro")
            recall = metrics.recall_score(y_true, pred, average="micro")
            f1_score = metrics.f1_score(y_true, pred, average="micro")
            auc = metrics.roc_auc_score(y_true, pred, average="micro")
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, auc {:g}, acc {:g}, recal {:g}, f1 {:g}".format( \
                    time_str, step, loss, auc, acc, recall, f1_score))
            sys.stdout.flush()
            if writer:
                writer.add_summary(summaries, step)
            return f1_score

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train, title_train, train_pos, train_entity, train_sf)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        dev_accuracy = 0.0
        for batch in batches:
            x_batch, y_batch, title_batch, train_pos, train_entity, train_sf = zip(*batch)
            train_step(x_batch, y_batch, title_batch, train_pos, train_entity, train_sf)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_acc = dev_step(x_dev, y_dev, title_dev, dev_pos, dev_entity, dev_sf, writer=dev_summary_writer)
                print("")
                if dev_acc > dev_accuracy:
                    dev_accuracy = dev_acc
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            # if current_step % FLAGS.checkpoint_every == 0:
            #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #     print("Saved model checkpoint to {}\n".format(path))
