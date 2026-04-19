import json

class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def fit(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join([self.itos.get(i, "") for i in ids])

    def __len__(self):
        return self.vocab_size
        
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.stoi = data['stoi']
            self.itos = {int(k): v for k, v in data['itos'].items()}
            self.vocab_size = len(self.stoi)

if __name__ == "__main__":
    tok = CharTokenizer()
    tok.fit("hola mundo")
    encoded = tok.encode("hola")
    print(f"Encode 'hola': {encoded}")
    print(f"Decode: {tok.decode(encoded)}")
    print(f"Vocab size: {len(tok)}")
