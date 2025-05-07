#!/usr/bin/env python3

import numpy as np
import os

def generate_large_matrix_input(size=128, output_path="programs/test_inputs/large_matrix_input.txt", 
                               expected_output_path="programs/test_outputs/large_matrix_output.txt"):
    print(f"Generating {size}x{size} matrix input...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(expected_output_path), exist_ok=True)
    
    np.random.seed(42)
    
    matrix_a = np.random.randint(1, 6, (size, size))
    
    matrix_b = np.eye(size, dtype=np.int32)
    
    result_matrix = matrix_a.copy()
    
    with open(output_path, 'w') as f:
        f.write(f"{size}\n")
        
        for i in range(size):
            row = " ".join([str(int(x)) for x in matrix_a[i]])
            f.write(f"{row}\n")
        
        for i in range(size):
            row = " ".join([str(int(x)) for x in matrix_b[i]])
            f.write(f"{row}\n")
    
    with open(expected_output_path, 'w') as f:
        for i in range(size):
            row = " ".join([str(int(x)) for x in result_matrix[i]])
            f.write(f"{row}\n")
    
    print(f"Generated matrix input file at: {output_path}")
    print(f"Generated expected output file at: {expected_output_path}")
    print(f"Matrix size: {size}x{size} ({size*size*2} elements)")
    print(f"This should generate {(size//32)**2} work chunks when using 32x32 block size")
    print(f"Expected file size: {(size*size*2*2 + size*2)/1024/1024:.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large matrix input for heterogeneous execution testing")
    parser.add_argument("--size", type=int, default=128, help="Size of square matrices (default: 128)")
    parser.add_argument("--output", type=str, default="programs/test_inputs/large_matrix_input.txt", 
                      help="Output file path (default: programs/test_inputs/large_matrix_input.txt)")
    parser.add_argument("--expected-output", type=str, default="programs/test_outputs/large_matrix_output.txt",
                      help="Expected output file path (default: programs/test_outputs/large_matrix_output.txt)")
    
    args = parser.parse_args()
    generate_large_matrix_input(args.size, args.output, args.expected_output)
    
    print("\nTo run the test with this matrix input:")
    print("  scripts/run_example.sh large_matrix_input")