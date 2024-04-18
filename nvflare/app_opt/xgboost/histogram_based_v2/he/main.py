from encryptor import Encryptor
from decrypter import Decrypter
from adder import Adder
from util import *
import numpy as np
import time

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

    #numbers = list(range(count))

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


num_workers = 10
N = 100000
BIN = 256
FEATURE = 30
AGGR = True

public_key, private_key = generate_keys()
encryptor = Encryptor(public_key, num_workers)
decrypter = Decrypter(private_key, num_workers)
adder = Adder(num_workers)

clear_gs = generate_random_numbers(N, -99999, 99999, as_int=True, as_pair=False)
clear_hs = generate_random_numbers(N, -99999, 99999, as_int=True, as_pair=False)

start_time = time.time()
combine_ghs = []
#print("Combine GHs:")
for g, h in zip(clear_gs, clear_hs):
    d = combine(g, h)
    #print(d)
    combine_ghs.append(d)
#print("Combine GHs end")
print(f"combine time is {time.time() - start_time}")

start_time = time.time()
encrypted_values = encryptor.encrypt(combine_ghs)
encoded = encode_encrypted_data(public_key, encrypted_values)
print(f"Encoding time is {time.time() - start_time}")

if AGGR:
    one_mask = np.random.randint(0, BIN, size=N)
    features = [(i, one_mask, BIN) for i in range(FEATURE)]
    start_time = time.time()
    result = adder.add(encrypted_values, features, encode_sum=False)
    print(f"Adding time is {time.time() - start_time}")
    start_time = time.time()
    for r in result:
        fid, gid, agg_ghs = r
        print(f"{fid=}")
        print(f"{gid=}")
        #print(f"agg_ghs {len(agg_ghs)}")
        #print(f"agg_ghs {agg_ghs}")
        #sub_start_time = time.time()
        combine_ghs_new = decrypter.decrypt(agg_ghs)
        #print(f"decoding time is {time.time() - sub_start_time}")
    print(f"Total decoding time is {time.time() - start_time}")
    start_time = time.time()
    #print("Decoded Combine GHs:")
    for i, d in enumerate(combine_ghs_new):
        #print(d)
        g, h = split(d)
        #print(g)
        #print(h)
    print(f"Split time is {time.time() - start_time}")
else:
    start_time = time.time()
    new_public_key, encrypted_ghs = decode_encrypted_data(encoded)
    assert new_public_key == public_key
    assert len(encrypted_ghs) == len(encrypted_values)
    print(f"{len(encrypted_ghs)}")
    print(f"{encrypted_ghs[0]}")
    combine_ghs_new = decrypter.decrypt(encrypted_ghs)
    print(f"decoding time is {time.time() - start_time}")

    start_time = time.time()
    print("Decoded Combine GHs:")
    for i, d in enumerate(combine_ghs_new):
        print(d)
        g, h = split(d)
        assert g == clear_gs[i]
        assert h == clear_hs[i]
    print(f"split time is {time.time() - start_time}")
