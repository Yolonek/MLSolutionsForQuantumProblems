from OperatorFunctions import s_total


def integer_to_binary(number: int, length: int):
    binary_number = bin(number)[2:]
    binary_length = len(binary_number)

    if binary_length < length:
        return binary_number.zfill(length)
    else:
        return binary_number


def generate_basis_states(start: int, stop: int, length: int,
                          block: bool = False, total_spin: int = 0):
    count = start
    while count < stop:
        binary_string = integer_to_binary(count, length)
        if block:
            if s_total(binary_string) == total_spin:
                yield binary_string
        else:
            yield binary_string
        count += 1
