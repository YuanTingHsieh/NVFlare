import argparse
import time
import numpy as np
from ipcl_python import PaillierKeypair


def generate_random_numbers(count, min_value, max_value, as_int: bool = True, as_pair: bool = True):
    """Generate random integers in the specified range.

    Parameters:
    - count (int): Number of random integers to generate.
    - min_value (int): Minimum value of the range (inclusive).
    - max_value (int): Maximum value of the range (inclusive).

    Returns:
    - np.ndarray: NumPy array containing the generated random integers.
    """
    # Generate random numbers between 0 and 1
    random_numbers = np.random.rand(count)

    # Scale and shift to the specified range
    numbers = (random_numbers * (max_value - min_value + 1)) + min_value


    if as_int:
        # Round to the nearest integer and convert to integers
        numbers = np.round(numbers).astype(int)
    

    if as_pair:
        interval = 2
        pairs = []
        for i in range(count//interval):
            pair = []
            for j in range(interval):
                if as_int:
                    pair.append(int(numbers[i + j]))
                else:
                    pair.append(numbers[i + j])
            pairs.append(np.array(pair))
        result = pairs
    elif as_int:
        result = [int(n) for n in numbers]
    else:
        result = numbers
    
    print(f"Created for {count=} {as_int=} {as_pair=}")
    return result


def encrypt(input_arr, pubkey):
    print("  start encrypting data")
    start_time = time.time()
    output_arr = [None] * len(input_arr)
    for i, d in enumerate(input_arr):
        ct_d = pubkey.encrypt(d)
        output_arr[i] = ct_d
    print(f"  encrypt data time is {time.time() - start_time}")
    return output_arr


def _encrypt(input_arr, pubkey):
    print("  start encrypting data")
    start_time = time.time()
    output_arr = pubkey.encrypt(input_arr)
    print(f"  encrypt data time is {time.time() - start_time}")
    return output_arr


def decrypt(input_arr, prikey):
    print("  start decrypting data")
    start_time = time.time()
    output_arr = [None] * len(input_arr)
    for i, ct_d in enumerate(input_arr):
        d = prikey.decrypt(ct_d)
        output_arr[i] = d
    print(f"  decrypt data time is {time.time() - start_time}")
    return output_arr


def check_two_arr(a_arr, b_arr):
    for i, j in zip(a_arr, b_arr):
        if not np.allclose(i, j):
            print("Not Equal")
            return
    print("All Equal")


def add_two_arr(a_arr, b_arr):
    print("  start adding data")
    start_time = time.time()
    c_arr = [None] * len(a_arr)
    for k, (i, j) in enumerate(zip(a_arr, b_arr)):
        c_arr[k] = i + j
    print(f"  adding data time is {time.time() - start_time}")
    return c_arr


def multiply_two_arr(a_arr, b_arr):
    print("  start mutiply two arrs")
    start_time = time.time()
    c_arr = [None] * len(a_arr)
    for k, (i, j) in enumerate(zip(a_arr, b_arr)):
        c_arr[k] = i * j
    print(f"  multiply time is {time.time() - start_time}")
    return c_arr


def print_arr(arr):
    for i, a in enumerate(arr):
        print(f"index {i=} has item {a=}")
    pass


def main():

    start_time = time.time()
    print("start creating key pairs")
    # create pub / pri key pairs
    pubkey, prikey = PaillierKeypair.generate_keypair(1024)
    print(f"create key pair time is {time.time() - start_time}")

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", default=200000, type=int)
    parser.add_argument("--as_int", action="store_true")
    parser.add_argument("--as_pair", action="store_true")
    args = parser.parse_args()
    N = args.N

    a = generate_random_numbers(N, -99999, 99999, as_int=args.as_int, as_pair=args.as_pair)
    print_arr(a)
    # test enc/dec
    ct_a = encrypt(a, pubkey)
    de_a = decrypt(ct_a, prikey)
    check_two_arr(a, de_a)

    # ciphertext addition
    b = generate_random_numbers(N, -99999, 99999, as_int=args.as_int, as_pair=args.as_pair)
    print_arr(b)
    ct_b = encrypt(b, pubkey)

    print("start adding ciphertext + ciphertext data")
    ct_c = add_two_arr(ct_a, ct_b)
    de_c = decrypt(ct_c, prikey)
    c = add_two_arr(a, b)
    check_two_arr(c, de_c)

    # ciphertext + plaintext addition
    print("start adding ciphertext + plaintext data")
    ct_c = add_two_arr(ct_a, b)
    de_c = decrypt(ct_c, prikey)
    check_two_arr(c, de_c)
    print_arr(c)
    print_arr(ct_c)
    print_arr(de_c)

    # TODO: this is disable because by default it uses gmpy2
    #       whose license is not Apache2
    ## ciphertext * plaintext multiplication
    #print("start multiply ciphertext * plaintext data")
    #ct_d = multiply_two_arr(ct_a, b)
    #de_d = decrypt(ct_d, prikey)
    #d = multiply_two_arr(a, b)
    #check_two_arr(d, de_d)


if __name__ == "__main__":
    main()
