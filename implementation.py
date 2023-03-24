import re

import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel
import spacy
from spacy.cli import download
import copy
import string

from model import Model

def build_model_123(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2 and 3 of the Coreference resolution pipeline.
            1: Ambiguous pronoun identification.
            2: Entity identification
            3: Coreference resolution
    """
    return StudentModel(device, True, True)


def build_model_23(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2 and 3 of the Coreference resolution pipeline.
            2: Entity identification
            3: Coreference resolution
    """
    return StudentModel(device, False, True)


def build_model_3(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements step 3 of the Coreference resolution pipeline.
            3: Coreference resolution
    """
    return StudentModel(device, False, False)


class RandomBaseline(Model):
    def __init__(self, predict_pronoun: bool, predict_entities: bool):
        self.pronouns_weights = {
            "his": 904,
            "her": 773,
            "he": 610,
            "she": 555,
            "him": 157,
        }
        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        predictions = []
        for sent in sentences:
            text = sent["text"]
            toks = re.sub("[.,'`()]", " ", text).split(" ")
            if self.predict_pronoun:
                prons = [
                    tok.lower() for tok in toks if tok.lower() in self.pronouns_weights
                ]
                if prons:
                    pron = np.random.choice(prons, 1, self.pronouns_weights)[0]
                    pron_offset = text.lower().index(pron)
                    if self.pred_entities:
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks
                        )
                    else:
                        entities = [sent["entity_A"], sent["entity_B"]]
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks, entities
                        )
                    predictions.append(((pron, pron_offset), entity))
                else:
                    predictions.append(((), ()))
            else:
                pron = sent["pron"]
                pron_offset = sent["p_offset"]
                if self.pred_entities:
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks
                    )
                else:
                    entities = [
                        (sent["entity_A"], sent["offset_A"]),
                        (sent["entity_B"], sent["offset_B"]),
                    ]
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks, entities
                    )
                predictions.append(((pron, pron_offset), entity))
        return predictions

    def predict_entity(self, predictions, pron, pron_offset, text, toks, entities=None):
        entities = (
            entities if entities is not None else self.predict_entities(entities, toks)
        )
        entity_idx = np.random.choice([0, len(entities) - 1], 1)[0]
        return entities[entity_idx]

    def predict_entities(self, entities, toks):
        offset = 0
        entities = []
        for tok in toks:
            if tok != "" and tok[0].isupper():
                entities.append((tok, offset))
            offset += len(tok) + 1
        return entities


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, device: str, predict_pronoun: bool, predict_entities: bool):
        self.device = device
        self.predict_pronouns = predict_pronoun  #flag to enable pronoun identification
        self.predict_entities = predict_entities #flag to enable entity identification
        self.tokenizer = None                    #Tokenizer from the transformer used pre-trained model
        self.spacy_tokenizer = None              #Tokenizer from a pre-trained model from spacy library
        self.modelPronIdent = None               #model for pronoun identification
        self.modelEntIdent = None                #model for entities identification
        self.modelEntRes = None                  #model for entities resolution
        self.tokes_ids = None                    #dictionary with {index of token: (token, offset of token)} for each token in a text
        self.tokens_off = None                   #dictionary with {offset of token: (token, index of token)} for each token in a text

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        #Load the spacy tokenizer from "en_core_web_sm" model
        if self.spacy_tokenizer is None:
            download("en_core_web_sm")
            self.spacy_tokenizer = spacy.load("en_core_web_sm")
        
        #Load the transformer tokanizer from "bert-base-cased"
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        #Load pre-trained model for pronoun identification and set in evaluation mode
        if self.predict_pronouns and self.modelPronIdent is None:
            self.modelPronIdent = torch.load("model/modelPronIdent.pt", map_location=self.device)
            self.modelPronIdent.eval()

        #Load pre-trained model for entities identification and set in evaluation mode
        if self.predict_entities and self.modelEntIdent is None:
            self.modelEntIdent= torch.load("model/modelEntIdent.pt", map_location=self.device)
            self.modelEntIdent.eval()

        #Load pre-trained model for entities resolution and set in evaluation mode
        if self.modelEntRes is None:
            self.modelEntRes = torch.load("model/modelEntRes.pt", map_location=self.device)
            self.modelEntRes.eval()        

        #lists to store data and results
        texts = []
        pronouns = []
        entities = []
        result = []

        #read input data
        for t in tokens:

            #main input: all texts
            texts.append(t["text"])

            #if not performing pronou identification, read given pronouns
            if not self.predict_pronouns:
                pronouns.append((t["pron"],t["p_offset"]))

            #if not performing entity identification, read given entity
            if not self.predict_entities:
                entities.append(((t["entity_A"],t["offset_A"]),(t["entity_B"],t["offset_B"])))

        #tokenize each text, saving index and offset of each token
        self.tokens_ids, self.tokens_off = self.find_index_offsets(texts)

        #pronoun predictions
        if len(pronouns)==0:
            inp, pron = self.organize_data_for_pron_ident(texts)  #organize data for model input
            output = []
            for i,p in zip(inp,pron):
                with torch.no_grad():
                    output.append(torch.round(self.modelPronIdent(i,p))) #predictions
            pronouns = self.find_ambiguous_pronoun(output)               #decode predictions

        #entities predictions
        if len(entities)==0:
            inp, index_pron = self.organize_data_for_ent_ident(pronouns)  #organize data for model input
            output = []
            for i, ip in zip(inp,index_pron):
                with torch.no_grad():
                    output.append(self.modelEntIdent(i,ip))  #predictions
            entities = self.find_entities(output)            #decode predictions

        #entity resolution
        inp, indicies, tokens_entity = self.organize_data_for_ent_res(pronouns,entities)  #organize data for model input
        output = []
        for i, ti, te in zip(inp, indicies, tokens_entity):
            with torch.no_grad():
                output.append(torch.round(self.modelEntRes(i,ti,te))) #predictions

        #re-organize output: previous model return a list of entities, one for each element,
        #instead I need a list of couples of entities
        #[ent1,ent2,ent3,ent4,...] -> [(ent1,ent2),(ent3,ent4),...]
        new_output = [(output[i],output[i+1]) for i in range(0, len(output), 2)]

        #build the result: list[((pronoun,offset),(entity,offset)),...]
        for o,p,e in zip(new_output, pronouns, entities):

            #initialize result with pronoun and an empty entity
            res = [p,[]]

            #Add first found entity if classified as true
            if o[0]==1:
                #Assign a default values if no entity identified in the previous step
                if e[0] is None:
                    e[0] = ("None", 0)
                res[1] = e[0]

            #Add second found entity if classified as true
            if o[1]==1:
                #Assign a default values if no entity identified in the previous step
                if e[1] is None:
                    e[1] = ("None", 0)
                res[1] = e[1]

            #append the result for each element
            result.append(res)

        #return final list of results
        return result

    #Organize data for entity resolution
    def organize_data_for_ent_res(self, pronouns, entities):

        #find the indicies of the tokens 
        indicies_of_interest = self.find_indicies_of_interest(self.tokens_ids, pronouns, entities)
      
        new_input = []     #list to store the dictionaries {"input_ids":input_ids, "attention_mask":mask}
        tokens_entity = [] #list to store the number of the tokens of each entity

        #Pattern of data: "text [SEP] pronoun [SEP] entity"
        #It is considered all first token of tokenized text (the root of the word)
        for t_id,p,e in zip(self.tokens_ids,pronouns,entities):

            input_ids = []
            input_idsA = []
            input_idsB = []
            maskA = []
            maskB = []

            #tokenize all text
            tokens = [t[0] for t in t_id.values()]
            input_ids = [torch.tensor(self.tokenizer(ti)[0].ids[1]) for ti in tokens]
            input_ids.append(torch.tensor(self.tokenizer("[SEP]")[0].ids[1]))
            input_ids.append(torch.tensor(self.tokenizer(p[0])[0].ids[1]))
            input_ids.append(torch.tensor(self.tokenizer("[SEP]")[0].ids[1]))

            #add first entity
            input_idsA = copy.deepcopy(input_ids)
            ent = ("None",0) if e[0] is None else e[0] #set entity to default value if None
            entity = self.tokenizer.tokenize(ent[0])   #tokenize entity
            token_entity = 0
            for i in entity:
                input_idsA.append(torch.tensor(self.tokenizer(i)[0].ids[1])) #add all tokens entity
                token_entity+=1
            tokens_entity.append(token_entity) 
            input_idsA = torch.stack(input_idsA).unsqueeze(0)
            maskA = torch.ones(input_idsA.shape) #associate a mask made by all '1'

            #add second entity
            input_idsB = copy.deepcopy(input_ids)
            ent = ("None",0) if e[1] is None else e[1] #set entity to default value if None
            entity = self.tokenizer.tokenize(ent[0])   #tokenize entity
            token_entity = 0
            for i in entity:
                input_idsB.append(torch.tensor(self.tokenizer(i)[0].ids[1])) #add all tokens entity
                token_entity+=1
            tokens_entity.append(token_entity)
            input_idsB = torch.stack(input_idsB).unsqueeze(0)
            maskB = torch.ones(input_idsB.shape) #associate a mask made by all '1'

            #Add the processed data to a list for the final input shape
            new_input.append({"input_ids":input_idsA,"attention_mask":maskA})
            new_input.append({"input_ids":input_idsB,"attention_mask":maskB})

        #check if leghts of lists match
        assert len(new_input) == len(tokens_entity) == len(indicies_of_interest)

        #return results
        return new_input, indicies_of_interest, tokens_entity
     
    #Organize data for entity identification   
    def organize_data_for_ent_ident(self, pronouns):

        #input data
        new_input = []         #list to store the dictionaries {"input_ids":input_ids, "attention_mask":mask}
        indicies_pronouns = [] #list to store the token index of each pronoun
        
        #scan all data
        for t_id,p in zip(self.tokens_ids,pronouns):
            tokens = [t[0] for t in t_id.values()]

            #Add special token to represent the input as "text [SEP] pronoun [SEP]"
            tokens.append("[SEP]")
            tokens.append(p[0][0])
            tokens.append("[SEP]")

            #Build the dictionary with input_ids and mask for each text
            input_ids = []
            mask = []
            for ti in tokens:
                #covert a token into the correspondet id
                input_ids.append(torch.tensor(self.tokenizer(ti)[0].ids[1]))
            input_ids = torch.stack(input_ids).unsqueeze(0) 
            mask = torch.ones(input_ids.shape) #associate a mask made by all '1'
            new_input.append({"input_ids":input_ids,
                              "attention_mask":mask})

            #store the index for current pronoun
            index_pronouns = []
            for k,v in t_id.items():
                if p==v:
                    index_pronouns.append(k)

            #store all indicies of pronouns
            indicies_pronouns.append(index_pronouns)

        #check if leghts of lists match
        assert len(new_input) == len(indicies_pronouns)

        #return results
        return new_input, indicies_pronouns

    #organize data for pronoun identification
    def organize_data_for_pron_ident(self, texts):

        tokens, ident_pron = self.find_pronouns(texts)

        #tokens of text
        new_input = []
        masks = []
        for t in tokens:  
            input_ids = []
            mask = []
            for ti in t:
                #convert tokns in input ids for the transformer
                input_ids.append(torch.tensor(self.tokenizer(ti)[0].ids[1]))
            input_ids = torch.stack(input_ids).unsqueeze(0)
            mask = torch.ones(input_ids.shape) #associate a mask made by all '1'
            new_input.append({"input_ids":input_ids,
                              "attention_mask":mask})

        #check if leghts of lists match
        assert len(new_input) == len(ident_pron)

        #return results
        return new_input, ident_pron

    #function to find all the pronouns in a text
    def find_pronouns(self, texts):

        pronouns = [] #list to store pronouns
        tokens = []   #list to store text tokens

        #load a pre-trained model to do POS TAGGING
        nlp = self.spacy_tokenizer

        #scan all data
        for t in texts:
            doc = nlp(t)
            pron = {}
            token_text = []
            #for each token found by the pre-trained model tokanization
            #take it if is a pronoun: number of token, token, offset of token
            for i, token in enumerate(doc):
                token_text.append(str(token))
                if token.pos_ == "PRON":
                    pron.update({i:(token.text, str(token.idx))}) #{index of pronoun: (pronoun, offset)}
            tokens.append(token_text)
            pronouns.append(pron)

        #check if leghts of lists match
        assert len(texts) == len(tokens) == len(pronouns)

        #return results
        return tokens, pronouns

    #function to build dictionaries which store tokens of the texts with related index and offset
    def find_index_offsets(self, texts):

        tokens_off = [] #list to store dictionaries {offset of token: (token, index of token)} for each token in a text
        tokens_id = []  #list to store dictionaries {index of token: (token, offset of token)} for each token in a text

        #load a pre-trained model to do tokanization
        nlp = self.spacy_tokenizer

        #scan all data
        for t in texts:
            doc = nlp(t)
            token_id = {}
            token_off = {}
            #for each token found by the pre-trained model tokanization
            #take index of token, token, offset of token
            for i, token in enumerate(doc):
                token_id.update({i:(token.text, token.idx)})
                token_off.update({token.idx: (token.text, i)})
            tokens_id.append(token_id)
            tokens_off.append(token_off)

        #check if leghts of lists match
        assert len(texts) == len(tokens_off) == len(tokens_id)

        #return results
        return tokens_id, tokens_off

    #function to find the indicies of identified (or given) pronouns and entities
    def find_indicies_of_interest(self, tokens_ids, pronouns, entities):

        #Save the token index of each entity in the text 
        index_entities = []
        found = False
        for t,ent in zip(tokens_ids,entities):
            for e in ent:
                et = "None "if e is None else e[0] #Assign a default value if None
                eo = 0 if e is None else e[1]      #Assign a default value if None
                entity = (et.split(" ")[0],eo)     #Build the couple (entity,offset)
                #check for the entity between token to find index
                for k,v in t.items():
                    if entity==v:
                        index_entities.append(k)
                        found = True
                if not found:
                    index_entities.append(None) #if entity is not foud assign a None index
                else:
                    found = False

        #Save the token index of each pronoun in the text
        index_pronouns = []
        found = False
        #check for the pronou between token to find index
        for t,p in zip(tokens_ids,pronouns):
            for k,v in t.items():
                if p==v:
                    #insert the same pronoun twice beacuse the two entities (with same pronoun)
                    #are added to a list in a sequential way
                    index_pronouns.append(k)
                    index_pronouns.append(k)


        #build a list with elemnts (index pronoun, (index entityA, index entityB))
        index_of_interest = []
        for i1,i2 in zip(index_entities,index_pronouns):
            index_of_interest.append((i1,i2))  

        #return result
        return index_of_interest

    #find entities from predictions
    def find_entities(self, out):

        total_entities = [] #list to store predicted entities

        #scan all data to find identified entities
        for elem_pred, ti in zip(out,self.tokens_ids):

            #initialize the coupleof entity
            found_entities = [None,None]

            #for each predicted elemnt
            for i in range(len(elem_pred)):

                #take each element
                e = elem_pred[i]

                #if the predicted element is 1 (B - Beginning part of an entity)
                if e==1:

                    #look for first entity if not found yet
                    if found_entities[0] is None:
                        
                        try:
                            ent1,off1 = ti[i]
                        except KeyError: #the index is not present in the tokens
                            pass
                        try:
                            c = 1
                            #if the predicted element is 2 (I - intermediate token of a word)
                            while elem_pred[i+c]==2:
                                try:
                                    ent1 += " "+ti[i+c][0]
                                except KeyError: #the index is not present in the tokens
                                    pass
                                c+=1
                        except IndexError:
                            pass
                        found_entities[0] = (ent1,off1) #entity found

                    #look for second entity
                    elif found_entities[1] is None:
                        try:
                            ent2,off2 = ti[i]
                        except KeyError:
                            pass
                        try:
                            c = 1
                            #if the predicted element is 2 (I) ans obviusly follow (B)
                            while elem_pred[i+c]==2:
                                try:
                                    ent2 += " "+ti[i+c][0]
                                except KeyError:
                                    pass
                                c+=1
                        except IndexError:
                            pass
                        found_entities[1] = (ent2,off2)

                #ignore the element identified as '0'
                else:
                    continue

            #store the entities
            total_entities.append(found_entities)

        #return results
        return total_entities

    #function to find the ambiguous pronoun
    def find_ambiguous_pronoun(self, output):

        pron = [] #list to store the ambiguous pronoun of each text

        #scan the predictions to take the identified pronoun for each text
        for out, ti in zip(output,self.tokens_ids):
            el = torch.argmax(out).item()
            pron.append(ti[el])

        #return results
        return pron

#Model for Ambiguous Pronoun Identification
class AmbiguousPronounIdentification(torch.nn.Module):

    def __init__(self, n_hidden, classes, device, language_model_name="bert-base-cased"):
        super(AmbiguousPronounIdentification, self).__init__()

        #Transformer model
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)

        #Freeze Transformer
        for param in self.transformer_model.parameters():
          param.requires_grad = False

        #Number of features in output from the transformer
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Additive feature to represent where is a predicate
        self.add_features = 1

        #BiLSTM layer
        self.lstm = nn.LSTM(self.embedding_dim+self.add_features, n_hidden, bidirectional=True, num_layers = 2, dropout=0.3, batch_first = True)

        #Linear classifier
        self.hidden1 = torch.nn.Linear(2*n_hidden, classes)

    def forward(self, input_ids: torch.Tensor = None, pron: list = None):

        #outbut from transformer
        out = self.transformer_model(**input_ids)
        out = torch.stack(out.hidden_states[-2:], dim=0).mean(dim=0)
        
        #associate '1' in correspondence with the identified pronouns and '0' otherwise
        new_out = []
        for i in range(out[0].shape[0]):
            o = []
            #concatenate '1' if the token is a predicate, '0' otherwise
            if i in pron.keys():
                o.append(torch.cat((out[0,i,:], torch.ones(1))))
            else:
                o.append(torch.cat((out[0,i,:], torch.zeros(1))))
            new_out.append(torch.stack(o))
        new_out = torch.stack(new_out)

        out, _ = self.lstm(new_out) #BiLSTM
        out = self.hidden1(out)     #Linear classifier
        out = torch.sigmoid(out)

        return out.squeeze(-1)

#Model for Entities Identification
class EntitiesIdentification(torch.nn.Module):

    def __init__(self, n_hidden, classes, device, language_model_name="bert-base-cased"):
        super(EntitiesIdentification, self).__init__()

        #Transformer model
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)

        #Freeze Transformer
        #for param in self.transformer_model.parameters():
          #param.requires_grad = False

        #Number of features in output from the transformer
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Additive feature to represent where is a predicate
        self.add_features = 10

        #BiLSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, n_hidden, bidirectional=True, num_layers = 2, dropout=0.3, batch_first = True)

        #Linear classifier
        self.hidden1 = torch.nn.Linear(2*n_hidden+self.add_features, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, classes)

        #CRF layer
        self.crf = CRF(classes, batch_first = True)

    def forward(self, inp: torch.Tensor=None, index_p=None, tags=None):

        masks = []
        for m in inp["attention_mask"][0]:
            masks.append(m.type(torch.bool))
        masks = torch.stack(masks)

        out = self.transformer_model(**inp)
        out = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        out, _ = self.lstm(out)

        #associate '1' in correspondence with the ambiguous pronouns and '0' otherwise
        new_out = []
        o = []
        for i in range(out.shape[1]):
            if i == index_p:
                o.append(torch.cat((out[0,i,:], torch.ones(self.add_features))))
            else:
                o.append(torch.cat((out[0,i,:], torch.zeros(self.add_features))))
        out = torch.stack(o)
        new_out.append(out)
        new_out = torch.stack(new_out)

        #Linear classifier
        out = self.hidden1(new_out)
        out = torch.relu(out)
        out = nn.Dropout(0.3)(out)
        out = self.hidden2(out)
        out = F.log_softmax(out,-1)

        #CRF layer
        if tags is not None:

          loss = self.crf(out, tags, mask=masks, reduction="mean") #for training
          return -loss

        else:
          out = self.crf.decode(out) #for inference
          return out[0]

#Model for Entity Resolution
class ConferenceResolution(torch.nn.Module):

    def __init__(self, n_hidden, classes, device, language_model_name="bert-base-cased"):
        super(ConferenceResolution, self).__init__()

        #Transformer model
        self.transformer_model = AutoModel.from_pretrained(language_model_name, output_hidden_states=True)

        #Freeze Transformer
        for param in self.transformer_model.parameters():
          param.requires_grad = False

        #Number of features in output from the transformer
        self.embedding_dim = self.transformer_model.config.hidden_size

        #Additive feature to represent where is a predicate
        self.add_features = 0

        #BiLSTM layer
        self.lstm = nn.LSTM(self.embedding_dim+self.add_features, n_hidden, bidirectional=True, num_layers = 2, dropout=0.3, batch_first = True)

        #Linear classifier
        #self.clf = svm.SVC(kernel='linear', C=classes)
        self.hidden1 = torch.nn.Linear(2*n_hidden, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, classes)

    def forward(self, inp: torch.Tensor = None, index=None, lenghts=None):

        out = self.transformer_model(**inp)
        out = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0)
        out, _ = self.lstm(out)

        new_out = []
        output = out[0]
        mask = inp["attention_mask"][0]
        i = index
        l = lenghts
        new_representation = []

        #eliminate pad (if there is)
        for o,m in zip(output,mask):
            if m == 1:
                new_representation.append(o)

        i1 = i[0] #index of entity
        i2 = i[1] #index of pronoun

        #Look for entity
        if i1 is not None:
            if l>1:
                #take the mean of the representation of the features 
                #of the entity tokens if there is more than one token
                o1 = torch.stack(new_representation[i1:i1+l])
                o1 = torch.mean(o1, -2)
            else:
                #entity representation (made by only one token)
                o1 = new_representation[i1]
        else:
            #take the mean of the representation of the features 
            #of the all tokens if no entity found
            o1 = torch.mean(output, -2)

        #Look for pronoun
        if i2 is not None:
            #pronoun representation
            o2 = new_representation[i2]
        else:
            #take the mean of the representation of the features 
            #of the all tokens if no pronoun found
            o2 = torch.mean(output, -2)

        #sum between entity and pronoun
        new_out = o1+o2

        #Linear classify
        out = self.hidden1(new_out)
        out = torch.relu(out)
        out = nn.Dropout(0.3)(out)
        out = self.hidden2(out)

        #activation function
        out = torch.sigmoid(out)

        return out.squeeze(-1)
