import sys
import random
import os

def generate_numbers(n, data_dir):
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

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <power1> [<power2> ...]")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)

    for arg in sys.argv[1:]:
        try:
            k = int(arg)
            if k < 1:
                raise ValueError
        except ValueError:
            print(f"Error: each power must be a positive integer (invalid: {arg})")
            continue

        n = 2 ** k
        generate_numbers(n, data_dir)

if __name__ == "__main__":
    main()
