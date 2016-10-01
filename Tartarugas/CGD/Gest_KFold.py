import pickle


base = 'C:\\Users\\Amand\\Documents\\Tartarugas\\CGD\\'
def saveKfold(kfold):
    output = open(base + 'kFold.pkl', 'wb')
    pickle.dump(kfold, output)
    output.close()

def upload_Kfold():
    filename = base + 'kFold.pkl'
    with open(filename, "rb") as f:
        while True:
            try:
                kfold = pickle.load(f)
            except EOFError:
                break
    return kfold