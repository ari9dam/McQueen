import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from tqdm import tqdm
import sys
from torch.multiprocessing import Pool


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()


def get_unmasked_sentences(model,tokenizer,text,gpu_num):
    topk=3
    device='cuda:'+gpu_num
    
    text = "[CLS] "+text.strip()+" [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    try:
        masked_index = tokenized_text.index('[MASK]')
    except ValueError:
        print(tokenized_text)
        0/0
        
    val,topk3_indexes = torch.nn.functional.softmax(predictions[0,masked_index],dim=0).topk(3)
    topk3_indexes = topk3_indexes.to('cpu').tolist()
    predicted_index = torch.argmax(predictions[0,masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens(topk3_indexes)
    return [text.replace('[MASK]',token).replace('[CLS]','').replace('[SEP]','') for token in predicted_token]


if __name__ == "__main__":
    
    inpfile = sys.argv[1]
    gpu_num = sys.argv[2]
    outfile = inpfile + ".out"
    cont = -1
    model.to('cuda:'+gpu_num)
    if len(sys.argv)>2:
        cont = int(sys.argv[2])
        outfile+= "." +str(cont)
        
    with open(inpfile,"r") as ifd, open(outfile,"w+") as ofd:
        for text in tqdm(ifd.readlines(),desc="Unmasking:"):
            unmasked_texts = get_unmasked_sentences(model,tokenizer,text,gpu_num)
            for unmasked_text in unmasked_texts:
                ofd.write(unmasked_text+"\n")