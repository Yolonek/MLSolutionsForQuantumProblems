def generate_basis_states(start: int, stop: int, length: int,
                          block: bool = False, total_spin: int = 0):
    count = start
    while count < stop:
        binary_string = integer_to_bitstring(count, length)
        if block:
            if s_total(binary_string) == total_spin:
                yield binary_string
        else:
            yield binary_string
        count += 1