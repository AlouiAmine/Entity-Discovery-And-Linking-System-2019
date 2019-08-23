from bert import Ner

model = Ner("out/")
def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = ""
    label=[]
    tag=[]
    for line in f:
        if len(line)==1 or line[0]==" ":
            if len(sentence) > 0:
                data.append(sentence)
                tag.append(label)
                #print(tag)
                sentence = ""
                label=[]
            continue
        splits = line.split('\t')
        #print(splits[-1])
        sentence+=' '+splits[0].rstrip()
        label.append(splits[-1].rstrip())
        #print('ok')
        #print(label)

    if len(sentence) >0:
        data.append((sentence))
        sentence = ""
        tag.append(label)
        label=[]

    return data, tag

listtext,tags=readfile('test.txt')
model.get_bert_embedding(listtext,tags,file_name='embedding.json')

