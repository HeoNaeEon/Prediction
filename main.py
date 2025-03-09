from ann_runner import ANNRunner
import itertools

nums = []
news = []

def main():

    seq = [str(x) for x in input(
        'Input a series of numbers separated by spaces (Press enter when done): ').split()]
        
    nums.append(seq)
    
    reverse_mapping = {'B': 0, 'P': 1, 'T': 2}
    
    original_seq = [reverse_mapping[char] for char in seq]

    news.append(original_seq)

    flattened_news = list(itertools.chain(*news))
    flattened_nums = list(itertools.chain(*nums))
    
    # Initialize the ANNRunner with the sequence data and training parameters
    runner = ANNRunner(sequence=flattened_news, epochs=10000, split_pct=80)

    # Train the ANN
    runner.train()

    # Make predictions using the trained ANN
    runner.predict()

    # Save the weights of the trained ANN
    runner.save_weights()

    print(flattened_nums)
    
if __name__ == "__main__":
 while True:
  main()
