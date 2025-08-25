import sys
import random

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_size> <output_file>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: input_size must be an integer")
        sys.exit(1)

    filename = sys.argv[2]

    try:
        with open(filename, "w") as outfile:
            random.seed(42)  # fixed seed for reproducibility
            for _ in range(n):
                real = random.random()
                imag = random.random()
                outfile.write(f"{real} {imag}\n")
    except OSError:
        print(f"Failed to open output file: {filename}")
        sys.exit(1)

    print(f"Generated {n} complex numbers in {filename}")

if __name__ == "__main__":
    main()
