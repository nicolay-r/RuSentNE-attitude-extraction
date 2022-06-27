def read_train_test(filepath):
    with open(filepath, "r") as f:
        parts = []
        for line in f.readlines():
            parts.append(line.strip().split(','))
    return parts[0], parts[1]
