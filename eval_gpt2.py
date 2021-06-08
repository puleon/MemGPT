import argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='gpt2',
    help='The model checkpoint for weights initialization.')

def main():
    args = parser.parse_args()
    
    device = 'cuda'
    model_id = args.model_name_or_path

    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    dataset = load_dataset('ptb_text_only')
    texts = [el['sentence'] for el in dataset['test']]
    encodings = tokenizer('\n\n'.join(texts), return_tensors='pt')

    max_length = model.config.n_positions
    stride = 1024

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)

    print("Perplexity:", ppl)

if __name__ == "__main__":
    main()
