from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='semantic matching model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--clip_gradient', type=float, default=0.3, help='gradient clipping')

    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--word_normalize', action='store_true')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')  # fine-tune the word embeddings

    parser.add_argument('--out_dim', type=int, default=16,
                        help='number of feature maps.')
    parser.add_argument('--embedding_normalized', type=bool, default=True,
                        help='whether normalize char embeddings, default to True')
    parser.add_argument('--kernel_size', type=tuple, default=(3, 3))
    parser.add_argument('--padding', type=tuple, default=(1, 1))

    parser.add_argument('--max_que_length', type=int, default=148)
    parser.add_argument('--max_seq_length', type=int, default=96)

    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')  # use -1 for CPU
    parser.add_argument('--seed', type=int, default=42, help='the answer to life, the universe and everything.')
    parser.add_argument('--embed_file', type=str, default='./glove/glove.840B.300d.char.txt',
                        help='pre-trained word2vec file for init embedding layer, if None, use one-hot embedding.')
    # parser.add_argument('', type=int, default=1, help='.')
    # parser.add_argument('', type=int, default=1, help='.')
    # parser.add_argument('', type=int, default=1, help='.')
    # parser.add_argument('', type=int, default=1, help='.')

    # parser.add_argument('--neg_size', type=int, default=50,
    #                     help='negative sampling number')
    # parser.add_argument('--loss_margin', type=float, default=1.0)
    # parser.add_argument('--test', action='store_true', dest='test', help='get the testing set result')
    # parser.add_argument('--dev', action='store_true', dest='dev', help='get the development set result')
    # parser.add_argument('--log_every', type=int, default=100)
    # parser.add_argument('--dev_every', type=int, default=300)
    # parser.add_argument('--save_every', type=int, default=4500)
    # parser.add_argument('--patience', type=int, default=5, help="number of epochs to wait before early stopping")

    # parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    # parser.add_argument('--vocab_file', type=str, default='../vocab/vocab.word&rel.pt')
    # parser.add_argument('--rel_vocab_file', type=str, default='../vocab/vocab.rel.sep.pt')
    # parser.add_argument('--word_vectors', type=str, default='../vocab/glove.42B.300d.txt')
    # parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '../input_vectors.pt'))
    # parser.add_argument('--train_file', type=str, default='data/train.relation_ranking.pt')
    # parser.add_argument('--valid_file', type=str, default='data/valid.relation_ranking.pt')
    # parser.add_argument('--test_file', type=str, default='data/test.relation_ranking.pt')
    #
    # # added for testing
    # parser.add_argument('--trained_model', type=str, default='')
    # parser.add_argument('--results_path', type=str, default='results')
    # parser.add_argument('--write_res', action='store_true', help='write predict results to file or not')
    # parser.add_argument('--write_score', action='store_true')
    # parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    return args
