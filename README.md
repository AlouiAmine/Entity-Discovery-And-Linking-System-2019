# Entity Discovery And Linking System TAC2019
The goal of TAC-KBP Entity Discovery and Linking (EDL) is to extract mentions of pre-defined entity types, and link (disambiguate and ground) them to the entities in an English knowledge base (KB). In the past several years, they have only focused on five major coarse-grained entity types: person (PER), geo-political entity (GPE), location (LOC), organization (ORG) and facility (FAC). Many real world applications in scenarios such as disaster relief and technical support require us to significantly extend the EDL capabilities to a wider variety of entity types (e.g., technical terms, lawsuits, disease, crisis, vehicles, food, biomedical entities). In TAC-KBP2019 they will extend the number of types from five to thousands defined in YAGO. 

There are two stages in EDL– Entity Discovery and Entity Linking.

**Entity Discovery:** annotators find and annotate mentions for certain kinds of entities that appear in a document.

**Entity linking:** annotators search through a knowledge base (KB) to determine whether it includes an entry for each entity annotated during Entity Discovery and, if so, link the entity cluster to the KB entry.
The mention types are organizes in a hierarchy (e.g., actor as a subtype of artist, which in turn is a sub-type of person). 


# Requirements

-  `python3`
- `pip3 install -r requirements.txt`

# MODEL:

![model diagram image](/img/model_diagram.jpg)

# DATA:

download the dataset:

`wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip`

# Data preprocess:

 Data preprocessing may take some time, so it's better to run it in background using the following command:
 
 `nohup python3 -u preprocess_data.py > log_file.log & tail -f log_file.log`
 
 As a result, we get 2 files: train.txt, test.txt
 
 train.txt: used to fine tune bert-base-cased pretrained model
 
 test.txt: used to test bert-base-cased pretrained model and will be splitted then into (train and test) to train and test the classifier model.
 
# Fine-Tune BERT-NER:
 
`nohup python3 -u run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out/ --max_seq_length=128 --num_train_epochs 5  --do_eval --do_train --warmup_proportion=0.4 > bert_log.log & tail -f bert_log.log`

the implementation is based on https://github.com/kamalkraj/BERT-NER

# Generating the Classifier training data :

`nohup python3 -u get_bert_embedding.py > embedding_log.log & tail -f embedding_log.log`

# Training the classifier:


# Inference:


```from bert import Ner

model = Ner("out/")

output,_ = model.predict("Barack Obama the former president of US went to France",ok=True)

print(output)

# bert output :

# ['B-person', 'I-person', 'O', 'O', 'B-person', 'O', 'B-location', 'O', 'O', 'B-location']

# [
# ('Barack', {'tag': 'B-person/political_figure'}), 
# ('Obama', {'tag': 'I-person/political_figure'}), 
# ('the', {'tag': 'O'}), 
# ('former', {'tag': 'O'}), 
# ('president', {'tag': 'B-person/political_figure'}), 
# ('of', {'tag': 'O'}), 
# ('US', {'tag': 'B-location/country'}), 
# ('went', {'tag': 'O'}), 
# ('to', {'tag': 'O'}), 
# ('France', {'tag': 'B-location/city'})
# ]



```





