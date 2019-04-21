import tensorflow as tf
import pickle
from train import getModel
from utils import build_dict, build_dataset, batch_iter, get_text_list1
import argparse

def add_arguments(parser):

    parser.add_argument("--num_hidden", type=int, default=256, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=3, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=16, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=256, help="Word embedding size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=128, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.9, help="Dropout keep prob.")
    parser.add_argument("--restoreInTrain", type=bool, default=True, help="restore in train")
    parser.add_argument("--toy", action="store_true", help="Use only 50K samples of data")
    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")
    parser.add_argument("--checkoutPath", type=str, default='saved_model/checkpoint', help='save path')

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict()
print("Loading training dataset...")
valid_x, valid_y = get_text_list1(flag="dev")

valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

with tf.Session() as sess:
    print("Loading saved model...")
    model = getModel(sess, reversed_dict, article_max_len, summary_max_len, args, forward=True)
    # model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
    # saver = tf.train.Saver(tf.global_variables())
    # ckpt = tf.train.get_checkpoint_state("./saved_model/")
    # if ckpt:
    #     saver.restore(sess, tf.train.latest_checkpoint(ckpt.model_checkpoint_path))

    #batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)
    batches = batch_iter(valid_x, valid_y, args.batch_size, 1)
    print("Writing summaries to 'result.txt'...")
    for batch_x, _ in batches:
        batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

        valid_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }

        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]

        with open("result.txt", "a") as f:
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                print(" ".join(summary), file=f)

    print('Summaries are saved to "result.txt"...')
