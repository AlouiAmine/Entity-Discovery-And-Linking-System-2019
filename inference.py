from bert import Ner

model = Ner("out/")

output,_ = model.predict("Barack Obama the former president of US went to France",ok=True)

print(output)
