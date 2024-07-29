from bitarray import bitarray
from typing import Union, List


def integer_to_bitstring(integer: int, length: int) -> str:
    binary_number = bin(integer)[2:]
    binary_length = len(binary_number)

    if binary_length < length:
        return binary_number.zfill(length)
    else:
        return binary_number


def bitstring_to_int(bitstring: str) -> int:
    return int(bitstring, 2)


def integer_to_bitarray(integer: int, length: int) -> bitarray:
    return bitarray(integer_to_bitstring(integer, length))


def bitarray_to_integer(bit_array: bitarray) -> int:
    return int(bit_array.to01(), 2)


def flip(bit_array: bitarray, indexes: Union[int, List[int]]) -> bitarray:
    flipped = bit_array.copy()
    if isinstance(indexes, int):
        indexes = [indexes]
    for index in indexes:
        flipped[index] ^= True
    return flipped


def count_number_of_ones(integer: int, length: int) -> int:
    return integer_to_bitstring(integer, length).count('1')


def count_ones_and_zeros_difference(integer: int, length: int) -> int:
    """Adds 1 when '1' and -1 when '0' and returns the sum"""
    return (ones := count_number_of_ones(integer, length)) - (length - ones)


def shift_left(bit_array: bitarray, shift: int = 1) -> bitarray:
    return bit_array[(shift := shift % len(bit_array)):] + bit_array[:shift]


def shift_right(bit_array: bitarray, shift: int = 1) -> bitarray:
    return bit_array[-(shift := shift % len(bit_array)):] + bit_array[:-shift]


def shift_bits(bit_array: bitarray, shift: int = 1) -> bitarray:
    return shift_right(bit_array, shift) if shift >= 0 else shift_left(bit_array, abs(shift))


def shift_bits_of_integer(integer: int, length: int, shift: int = 1) -> int:
    return bitarray_to_integer(shift_bits(integer_to_bitarray(integer, length), shift))


if __name__ == "__main__":
    print(str(integer_to_bitstring(15, 8)))
    print(bitstring_to_int('00001111'))



