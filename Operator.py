

def s_z(spin_number: int, spin_configuration: str):
    spin_value = 0
    if spin_number < len(spin_configuration) and spin_configuration != '':
        if spin_configuration[spin_number] == '1':
            spin_value = 1 / 2
        elif spin_configuration[spin_number] == '0':
            spin_value = -1 / 2
    return spin_value


def s_total(spin_configuration: str):
    spins_up = spin_configuration.count('1')
    spins_down = spin_configuration.count('0')
    return spins_up * 1/2 + spins_down * (-1/2)


def s_plus_minus(spin_number: int, spin_configuration: str):
    new_state = spin_configuration
    if spin_number < len(new_state) - 1 and new_state != '':
        if new_state[spin_number: spin_number + 2] == '01':
            temp_state = list(new_state)
            temp_state[spin_number] = '1'
            temp_state[spin_number + 1] = '0'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def s_plus_minus_boundary(spin_configuration: str):
    new_state = spin_configuration
    if new_state != '':
        if new_state[-1] == '0' and new_state[0] == '1':
            temp_state = list(new_state)
            temp_state[-1] = '1'
            temp_state[0] = '0'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def s_minus_plus(spin_number: int, spin_configuration: str):
    new_state = spin_configuration
    if spin_number < len(new_state) - 1 and new_state != '':
        if new_state[spin_number:spin_number + 2] == '10':
            temp_state = list(new_state)
            temp_state[spin_number] = '0'
            temp_state[spin_number + 1] = '1'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def s_minus_plus_boundary(spin_configuration: str):
    new_state = spin_configuration
    if new_state != '':
        if new_state[-1] == '1' and new_state[0] == '0':
            temp_state = list(new_state)
            temp_state[-1] = '0'
            temp_state[0] = '1'
            new_state = ''.join(temp_state)
        else:
            new_state = ''
    return new_state


def calculate_interaction(spin_configuration: str, is_pbc: bool):
    list_of_spins = []
    for i in range(len(spin_configuration) - 1):
        list_of_spins.append(s_plus_minus(i, spin_configuration))
        list_of_spins.append(s_minus_plus(i, spin_configuration))
    if is_pbc:
        list_of_spins.append(s_plus_minus_boundary(spin_configuration))
        list_of_spins.append(s_minus_plus_boundary(spin_configuration))
    return list_of_spins
