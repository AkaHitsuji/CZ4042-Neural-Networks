import B1, B2, B3, B4, B5

def main():
    print('running with dropouts..')
    B1.main(with_dropout=True)
    print('B1 completed. Beginning B2...')

    B2.main(with_dropout=True)
    print('B2 completed. Beginning B3...')

    B3.main(with_dropout=True)
    print('B3 completed. Beginning B4...')

    B4.main(with_dropout=True)
    print('B4 completed.')
    print('All models run and graphs generated.')
if __name__ == "__main__":
    main()
