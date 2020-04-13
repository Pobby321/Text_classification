
import os
import tensorflow as tf
import numpy as np
import pickle
from gensim import corpora
from loader import load_data, prepare_dataset,BatchManager
print(tf.__version__)

root_path = os.getcwd()+os.sep  #F:\文本分类项目特训\文本分类\
flags = tf.app.flags

flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       True,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       False,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "cnews.vocab.txt",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
# flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
# flags.DEFINE_string("emb_file",     os.path.join(root_path+"data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join(root_path+"data", "cnews.train.txt"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join(root_path+"data", "cnews.dev.txt"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join(root_path+"data", "cnews.test.txt"),   "Path for test data")

FLAGS = tf.app.flags.FLAGS



def train():
    train_data = load_data(FLAGS.train_file)
    dev_data = load_data(FLAGS.dev_file)
    test_data = load_data(FLAGS.test_file)
    if not os.path.isfile(FLAGS.map_file):
        token_list = [list(sentence) for sentence in train_data[0]]
        sen_dictionary = corpora.Dictionary(token_list)
        char_to_id = sen_dictionary.token2id
        tag_dictionary = corpora.Dictionary(train_data[1])
        tag_to_id = tag_dictionary.id2token
        with open(FLAGS.map_file,'r','wb') as f:
            pickle.dump([char_to_id,tag_to_id],f)
    else:
        with open(FLAGS.map_file,'r') as f:
            char_to_id, tag_to_id = pickle.load(f)

    train_data = prepare_dataset(train_data,char_to_id,tag_to_id)
    dev_data = prepare_dataset(dev_data,char_to_id,tag_to_id)
    test_data = prepare_dataset(test_data,char_to_id,tag_to_id)

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

def clean():
    pass

def evaluate_line():
    pass

def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()

if __name__=="__main__":
    tf.app.run(main)