
def int_coef(data):
    weights = {
        3: 1,
        2: 0.5,
        4: 0.5,
        0: 0,
        1: 0,
        5: 0
    }

    # integral_coefficient = sum(weights.get(x, 0) for x in data) / len(data)
    weighted_sum = sum(weights.get(x, 0) for x in data)
    total_weight = sum(1 for x in data if x in weights)
    integral_coefficient = weighted_sum / total_weight

    return integral_coefficient

gen_int_coef = int_coef([3, 2, 4, 2, 3, 4, 3, 4, 2, 4])
print(f"Probability on generated data: {gen_int_coef:.2f}")

gen_failed_coef = int_coef([3, 3, 4, 2, 4, 3, 3, 4, 2, 3])
print(f"Probability on failed data: {gen_failed_coef:.2f}")

