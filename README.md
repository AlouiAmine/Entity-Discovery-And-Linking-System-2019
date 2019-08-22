# Entity-Discovery-And-Linking-System-2019
The goal of TAC-KBP Entity Discovery and Linking (EDL) is to extract mentions of pre-defined entity types, and link (disambiguate and ground) them to the entities in an English knowledge base (KB). In the past several years, they have only focused on five major coarse-grained entity types: person (PER), geo-political entity (GPE), location (LOC), organization (ORG) and facility (FAC). Many real world applications in scenarios such as disaster relief and technical support require us to significantly extend the EDL capabilities to a wider variety of entity types (e.g., technical terms, lawsuits, disease, crisis, vehicles, food, biomedical entities). In TAC-KBP2019 they will extend the number of types from five to thousands defined in YAGO. 

There are two stages in EDL– Entity Discovery and Entity Linking.
**Entity Discovery:** annotators find and annotate mentions for certain kinds of entities that appear in a document.

**Entity linking:** annotators search through a knowledge base (KB) to determine whether it includes an entry for each entity annotated during Entity Discovery and, if so, link the entity cluster to the KB entry.\\
The mention types are organizes in a hierarchy (e.g., actor as a subtype of artist, which in turn is a sub-type of person). 


# Requirements

-  `python3`
- `pip3 install -r requirements.txt`
