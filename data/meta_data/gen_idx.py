import pickle
residue2idx_dict = {'A':1, 'B': 20, 'C':2, 'D':3, 'E':4, 'U':3, 'F':5, 'G':6,
               'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'O':11,
               'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'X':20, 'Y':21, 'Z':20, '0':0}
pickle.dump(residue2idx_dict, open("residue2idx.pkl", 'wb'))