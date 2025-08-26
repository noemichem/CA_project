import sys
import random
import os

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_size>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: input_size must be an integer")
        sys.exit(1)

    # directory dove si trova questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)
    filename = f"numbers_{n}.txt"
    filepath = os.path.join(data_dir, filename)

    try:
        with open(filepath, "w") as outfile:
            random.seed(42)  # fixed seed for reproducibility
            for _ in range(n):
                real = random.random()
                imag = random.random()
                outfile.write(f"{real} {imag}\n")
    except OSError:
        print(f"Failed to open output file: {filepath}")
        sys.exit(1)

    print(f"Generated {n} complex numbers in {filepath}")

if __name__ == "__main__":
    main()
