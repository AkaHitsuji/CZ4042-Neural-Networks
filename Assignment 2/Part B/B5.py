import B1, B2, B3, B4, B5
import tensorflow as tf

def main():
    print('running with dropouts..')
    print('\n===== Running B1 with dropouts =====')
    B1.main(with_dropout=True)
    print('B1 completed. Beginning B2...')

    print('\n===== Running B2 with dropouts =====')
    B2.main(with_dropout=True)
    print('B2 completed. Beginning B3...')

    print('\n===== Running B3 with dropouts =====')
    tf.reset_default_graph()
    B3.main(with_dropout=True)
    print('B3 completed. Beginning B4...')

    print('\n===== Running B4 with dropouts =====')
    tf.reset_default_graph()
    B4.main(with_dropout=True)
    print('B4 completed.')
    print('All models run and graphs generated.')
if __name__ == "__main__":
    main()
