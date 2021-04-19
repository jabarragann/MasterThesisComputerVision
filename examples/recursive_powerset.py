x = ['a', 'b', 'c']


def calculate_power_sets_v1(set_arg):
    if len(set_arg) == 0:
        return []
    elif len(set_arg) == 1 or len(set_arg) == 2:
        return [set_arg]
    else:
        final_set = [set_arg]

        for e in set_arg:
            new_arg = set_arg.copy()
            new_arg.remove(e)
            final_set += calculate_power_sets_v1(new_arg)

        return final_set


def calculate_power_sets_v2(set_arg):
    prefix = []
    power_sets = []
    for e in set_arg:
        if len(prefix)==0:
            power_sets = [[],[e]]
        else:
            power_sets = power_sets + [[e]] + calculate_subsets_prefix_new_element(prefix,[e],[])
        prefix.append(e)
    return power_sets


def calculate_subsets_prefix_new_element(prefix, new_element, previous_subsets):
    """
    Calculate subsets of a prefix and a new element

    :param prefix:
    :param new_element:
    :param previous_subsets:
    :return:
    """
    if len(prefix) == 0:
        return previous_subsets
    else:
        new_subset = prefix + new_element

        if new_subset in previous_subsets:
            return previous_subsets
        else:
            previous_subsets += [new_subset]
            for e in prefix:
                new_prefix = prefix.copy()
                new_prefix.remove(e)
                previous_subsets = calculate_subsets_prefix_new_element(new_prefix, new_element, previous_subsets)
            return previous_subsets


# p_sets = calculate_power_sets(x) + [[]]
# print(p_sets)
# print(len(p_sets))


p_sets = calculate_subsets_prefix_new_element(['a','b'],['c'],[])
print(p_sets)

x = ['a', 'b', 'c','d','e','f']
p_sets = calculate_power_sets_v2(x)
print(p_sets)
print(len(p_sets))