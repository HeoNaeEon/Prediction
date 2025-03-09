from ann_runner import ANNRunner
import pandas as pd

res = []

def main():

    data = pd.read_csv('example.csv')

    df = pd.DataFrame(data) 

    for i in range(len(df)):
     if df["outcome"][i] == "tie":
      res.append(2)
     elif df["outcome"][i] == "player win":
      res.append(1)
     elif df["outcome"][i] == "banker win":
      res.append(0)
      
    # Initialize the ANNRunner with the sequence data and training parameters
    runner = ANNRunner(sequence=res, epochs=10000, split_pct=80)

    # Train the ANN
    runner.train()

    # Make predictions using the trained ANN
    runner.predict()

    # Save the weights of the trained ANN
    runner.save_weights()
    
if __name__ == "__main__":
 main()
