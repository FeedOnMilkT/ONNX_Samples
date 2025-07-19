# This component is unavailable for now
import torch

def main():
    print("Testing Model...")  
    print(torch.backends.mps.is_available())

if __name__ == "__main__":
    main()