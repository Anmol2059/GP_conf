import torch
import argparse
from myconformernew import MyConformer  # Import your MyConformer class

class Args:
    def __init__(self):
        self.emb_size = 256
        self.num_encoders = 4
        self.heads = 4
        self.kernel_size = 16

def test_myconformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = Args()

    model = MyConformer(
        emb_size=args.emb_size,
        n_encoders=args.num_encoders,
        heads=args.heads,
        kernel_size=args.kernel_size
    ).to(device)
    print("MyConformer initialized successfully.")

    random_input = torch.rand(8, 128, args.emb_size).to(device)
    print("Random input shape:", random_input.shape)

    try:
        output, embedding = model(random_input, device)
        print("MyConformer Output Shape:", output.shape)
        print("MyConformer Embedding Shape:", embedding.shape)
        print("MyConformer Output:", output)
    except Exception as e:
        print("Error during MyConformer testing:", str(e))

if __name__ == "__main__":
    test_myconformer()
