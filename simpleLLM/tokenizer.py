class Tokenizer:
    def __init__(self):
        self.word_to_id = {"I":0, "You":1, "like": 2, "do": 3, "not": 4, "coffee":5,"tea":6,".": 7}
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
    def encode(self, x):
        return [self.word_to_id[c] for c in x.split()]
    def decode(self, x):
        return ' '.join([self.id_to_word[c.item()] for c in x])


class TokenizerWithSystem:
    def __init__(self):
        self.word_to_id = {"I":0, "You":1, "like": 2, "do": 3, "not": 4, "coffee":5,"tea":6,".": 7, "human": 8, "system": 9}
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
    def encode(self, x):
        return [self.word_to_id[c] for c in x.split()]
    def decode(self, x):
        return ' '.join([self.id_to_word[c.item()] for c in x])


class TokenizerWithThinking:
    def __init__(self):
        self.word_to_id = {"I":0, "You":1, "like": 2, "do": 3, "not": 4, "coffee":5,"tea":6,".": 7, "human": 8, "system": 9, "<thinking>": 10, "</thinking>": 11}
        self.id_to_word = {v:k for k,v in self.word_to_id.items()}
    def encode(self, x):
        return [self.word_to_id[c] for c in x.split()]
    def decode(self, x):
        return ' '.join([self.id_to_word[c.item()] for c in x])
