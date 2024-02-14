
def load_data(filepath):
    values = []
    keys = []
    with open(filepath, "r", encoding="utf8") as file:
        for line in file:
            data = line.strip().split(",")
            try:
                keys.append(data[0])
                values.append(data[1])
            except:
                print("Value error at line: ", line)
    return keys, values


data_classes: list[str]
data_classes, data_content = load_data("C:/Users/Kamil/Desktop/spam_ham_ai/data_preparation/spam_or_ham.csv")
