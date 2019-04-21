import time

start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from utils import build_dict, build_dataset, batch_iter, get_text_list1


# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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



def getModel(session, reversed_dict, article_max_len, summary_max_len, args, forward=False):
    model = Model(reversed_dict, article_max_len, summary_max_len, args, forward)
    if args.restoreInTrain:
        # 判断FLAGS.train_dir是否有已经训练好的模型
        ckpt = tf.train.get_checkpoint_state(args.checkoutPath)
        if ckpt:
            model_checkpoint_path = ckpt.model_checkpoint_path
            print("Reading model parameters from %s" % model_checkpoint_path)

            saver = tf.train.Saver()
            saver.restore(session, tf.train.latest_checkpoint(args.checkoutPath))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
    return model
if __name__=='__main__':
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args()

        with open("args.pickle", "wb") as f:
            pickle.dump(args, f)

        if not os.path.exists("saved_model"):
            os.mkdir("saved_model")
        else:
            if args.with_model:
                old_model_checkpoint_path = open(args.checkoutPath, 'r')
                old_model_checkpoint_path = "".join(
                    ["saved_model/", old_model_checkpoint_path.read().splitlines()[0].split('"')[1]])


        print("Building dictionary...")
        word_dict, reversed_dict, article_max_len, summary_max_len = build_dict()
        print("Loading training dataset...")
        train_x, train_y = get_text_list1("train")

        with tf.Session() as sess:
            # model = Model(reversed_dict, article_max_len, summary_max_len, args)
            # sess.run(tf.global_variables_initializer())
            model = getModel(sess, reversed_dict, article_max_len, summary_max_len, args)
            saver = tf.train.Saver(tf.global_variables())

            batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
            num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

            print("\nIteration starts.")
            print("Number of batches per epoch :", num_batches_per_epoch)
            for batch_x, batch_y in batches:
                batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))

                batch_decoder_input = list(map(lambda x: [word_dict["GO"]] + list(x), batch_y))
                batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
                batch_decoder_output = list(map(lambda x: list(x) + [word_dict["EOS"]], batch_y))

                batch_decoder_input = list(
                    map(lambda d: d + (summary_max_len - len(d)) * [word_dict["PAD"]], batch_decoder_input))
                batch_decoder_output = list(
                    map(lambda d: d + (summary_max_len - len(d)) * [word_dict["PAD"]], batch_decoder_output))

                train_feed_dict = {
                    model.batch_size: len(batch_x),
                    model.X: batch_x,
                    model.X_len: batch_x_len,
                    model.decoder_input: batch_decoder_input,
                    model.decoder_len: batch_decoder_len,
                    model.decoder_target: batch_decoder_output
                }
                try:
                    _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)
                except:
                    st = '*'*50
                    st += '！出现一个错误！'
                    st += '*'*50
                    print(st)
                if step % 1000 == 0:
                    print("step {0}: loss = {1}".format(step, loss))

                if step % num_batches_per_epoch == 0:
                    hours, rem = divmod(time.perf_counter() - start, 3600)
                    minutes, seconds = divmod(rem, 60)
                    saver.save(sess, "./saved_model/model.ckpt", global_step=step)
                    print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
                          "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")
