#*-* coding:utf-8 *-*
#

import sys,os
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import re
import tensorflow as tf
import data_helpers
import csv
from tensorflow.contrib import learn
from sklearn import metrics

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_data", "../data/test_data", "Data source for the test data.")
tf.flags.DEFINE_string("label_dict", "../data/label_dict", "Data source for the label_dict data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("sent_length", 21, "the fixed window size (default: 13),奇数")
tf.flags.DEFINE_string("checkpoint_dir", "../data/runs/1517922440/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print >> sys.stderr, "\nParameters:"
for attr, value in sorted(FLAGS.__flags.items()):
    print >> sys.stderr, "{}={}".format(attr.upper(), value)
print >> sys.stderr, ""


# x_train, y_train = data_helpers.load_data_and_labels("../data/train_filter_data")
x_dev, title_dev, y_dev, dev_beg, dev_end, dev_entity, dev_sf, triggers = data_helpers.load_data_and_labels(FLAGS.test_data, FLAGS.label_dict, is_train=False, is_trigger = True)
x_dev, dev_pos, dev_entity = data_helpers.trigger2mid(x_dev, dev_beg, dev_end, dev_entity, FLAGS.sent_length)

label_dict = data_helpers.load_label_dict(FLAGS.label_dict)
# print sorted(label_dict.items(), lambda x, y: cmp(x[1], y[1]))
target_names = [k[0] for k in sorted(label_dict.items(), lambda x, y: cmp(x[1], y[1]))]

# label转为类别
def label_reverse(y):
    for k, v in label_dict.items():
        if v == y:
            return k
# Map data into vocabulary
# 词向量存放路径并取出
def read_label(dict_dir='../data/'):
    return data_helpers.load_label_dict(dict_dir+'label_dict_default_70.0')
default_dict = read_label()
#print default_dict
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

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# 测试语料,写入一个array,依次串行写入
x_dev = np.array(list(vocab_processor.transform([" ".join(x) for x in x_dev])))

title_vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "title_vocab")
title_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(title_vocab_path)
title_dev = np.array(list(title_vocab_processor.transform([" ".join(x) for x in title_dev])))

# Launch the graph
# 加载图
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)  # 通过检查点文件锁定最新的模型
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')  # 载入图结构，保存在.meta文件中
    saver.restore(sess, ckpt.model_checkpoint_path)

    graph = tf.get_default_graph()

    # accuracy = tf.get_collection('accuracy')[0]
    # print accuracy

    input_x = graph.get_operation_by_name('input_x').outputs[0]
    input_y = graph.get_operation_by_name('input_y').outputs[0]
    input_title = graph.get_operation_by_name('input_title').outputs[0]
    input_pos = graph.get_operation_by_name('input_pos').outputs[0]
    input_entity = graph.get_operation_by_name('input_entity').outputs[0]
    input_sf = graph.get_operation_by_name('input_sf').outputs[0]
    dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

    predictions = graph.get_operation_by_name("output/predictions").outputs[0]
    all_predictions = sess.run(predictions, {input_x: x_dev, input_title: title_dev, input_pos: dev_pos,
              input_entity: dev_entity, input_sf: dev_sf, dropout_keep_prob: 1.0})

y_dev = np.argmax(y_dev, axis=1)

# 输出预测结果(所有类别)
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction")
# f = open(out_path, "w")
num = 0
for x in vocab_processor.reverse(x_dev):
    # f.write("%s\t%s\t%s\n" % (label_reverse(all_predictions[num]), label_reverse(y_dev[num]), x))
    this_label = label_reverse(all_predictions[num])
    item = triggers[num]
    if this_label=='notanyclass':
        if default_dict.has_key(item):
            this_label = default_dict[item]
            print item
#    if item in default_dict.keys():
    print("%s\t%s\t%s" % (this_label, label_reverse(y_dev[num]), x))
    num += 1
# f.close()
