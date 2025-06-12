with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("len : ", len(text))

# all the unique characters that occur in this text
chars=sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
