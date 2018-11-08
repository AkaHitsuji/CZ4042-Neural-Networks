import B3, B4
import tensorflow as tf

def main():
    print('running with different cell types...')
    # B3 vanilla RNN
    print('\n===== Running B3 with vanilla rnn layer =====')
    B3.main(cell_type='rnn')
    print('B3 with vanilla rnn layer completed')

    # B3 LSTM
    print('\n===== Running B3 with LSTM layer =====')
    B3.main(cell_type='lstm')
    print('B3 with LSTM layer completed')

    # B4 vanilla RNN
    print('\n===== Running B4 with vanilla rnn layer =====')
    B4.main(cell_type='rnn')

    print('B4 with vanilla rnn layer completed')

    # B4 LSTM
    print('\n===== Running B4 with LSTM layer =====')
    B4.main(cell_type='lstm')
    print('B4 with LSTM layer completed')

    print('\n\nrunning with different number of layers...')
    # B3 2 layers
    print('\n===== Running B3 with 2 rnn layers =====')
    B3.main(num_layers=2)
    print('B3 with 2 layers completed')

    # B4 2 layers
    print('\n===== Running B4 with 2 rnn layers =====')
    B4.main(num_layers=2)
    print('B4 with 2 layers completed')

    print('\n\nrunning with gradient clipping...')
    # B3 gradient clipping
    print('\n===== Running B3 with gradient clipping =====')
    tf.reset_default_graph()
    B3.main(gradient_clipping=True)
    print('B3 with gradient clipping completed')

    # B4 gradient clipping
    print('\n===== Running B4 with gradient clipping =====')
    tf.reset_default_graph()
    B4.main(gradient_clipping=True)
    print('B4 with gradient clipping completed')

    print('all parts completed and graphs generated')



if __name__ == "__main__":
    main()
