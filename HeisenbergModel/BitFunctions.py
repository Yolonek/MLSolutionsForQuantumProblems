from bitarray import bitarray
from typing import Union, List


def integer_to_bitstring(number: int, length: int) -> str:
    binary_number = bin(number)[2:]
    binary_length = len(binary_number)

    if binary_length < length:
        return binary_number.zfill(length)
    else:
        return binary_number


def integer_to_bitarray(number: int, length: int) -> bitarray:
    return bitarray(integer_to_bitstring(number, length))


def bitarray_to_integer(bit_array: bitarray) -> int:
    return int(bit_array.to01(), 2)


def flip(bit_array: bitarray, indexes: Union[int, List[int]]) -> bitarray:
    flipped = bit_array.copy()
    if isinstance(indexes, int):
        indexes = [indexes]
    for index in indexes:
        flipped[index] ^= True
    return flipped


def count_number_of_ones(number: int, length: int) -> int:
    return integer_to_bitstring(number, length).count('1')


def count_ones_and_zeros_difference(number: int, length: int) -> int:
    """Adds 1 when '1' and -1 when '0' and returns the sum"""
    return (ones := count_number_of_ones(number, length)) - (length - ones)


if __name__ == '__main__':
    print(count_number_of_ones(16, 4))
    print(count_ones_and_zeros_difference(0, 4))

