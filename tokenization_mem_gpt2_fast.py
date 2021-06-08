from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast, BatchEncoding


class MemGPT2TokenizerFast(GPT2TokenizerFast):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_tokens(['[mem]'])
        self.mem_token = self.get_added_vocab()['[mem]']
        

    def __call__(self, texts, return_tensors=None) -> BatchEncoding:
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = super().__call__(texts)
        
        mem_input_ids = []
        for seq in encoded_input['input_ids']:
            mem_seq = []
            for tok in seq:
                mem_seq.append(tok)
                mem_seq.append(self.mem_token)
            mem_input_ids.append(mem_seq)

        mem_attention_mask = []
        for seq in encoded_input['attention_mask']:
            mem_attention_mask.append(2*seq)

        mem_encoded_input = {'input_ids': mem_input_ids, 'attention_mask': mem_attention_mask}

        return BatchEncoding(mem_encoded_input, tensor_type=return_tensors)        
