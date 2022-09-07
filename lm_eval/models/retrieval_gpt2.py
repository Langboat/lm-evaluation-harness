import transformers
import torch
from lm_eval.base import BaseLM
import base64
import requests
import time
import json
headers = {
    'accept': 'application/json',
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
}

config = json.load(open("./config.json"))
class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="retrieval",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        retrieval=True,
        request_server=config['request_server'],
        tokenizer_name=config['tokenizer_name'],
        chunk_size=config['chunk_size']
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.Re_gptForCausalLM.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        ).to(self.device)
        self.gpt2.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.chunk_size = chunk_size
        self.request_server = request_server
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.retrieval = retrieval
        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def split_token(self, text):
        encode_dict = self.tokenizer(text, return_offsets_mapping=True)
        offset_mapping = encode_dict['offset_mapping']
        chunk_count = int(len(offset_mapping) / self.chunk_size)
        chunk_texts = []
        for chunk_offset in range(chunk_count):
            chunk_mappings = offset_mapping[chunk_offset * self.chunk_size: chunk_offset * self.chunk_size + self.chunk_size]
            chunk_text = text[chunk_mappings[0][0]: chunk_mappings[-1][1]]
            chunk_texts.append(chunk_text)
        return chunk_texts

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        text = self.tok_decode(inps[0].tolist())
        chunk_text = self.split_token(text)
        if self.retrieval:
            with torch.no_grad():
                if len(chunk_text) > 0:
                    data = {"query": [base64.b64encode(s.encode('utf-8')).decode('utf-8') for s in chunk_text]}
                    response_text = ''
                    # try request until success
                    while response_text == '':
                        try:
                            response = requests.post(self.request_server, headers=headers, json=data)
                            response_text = json.loads(response.text)
                            retrieval = response_text
                        except Exception as e:
                            time.sleep(1)
                            print(e)
                    chunk_1 = [i[0] for i in retrieval]
                    chunk_2 = [i[1] for i in retrieval]
                    chunk_1 = torch.tensor(self.tokenizer(chunk_1, max_length=self.chunk_size * 2, padding="max_length", truncation=True).input_ids).unsqueeze(1)
                    chunk_2 = torch.tensor(self.tokenizer(chunk_2, max_length=self.chunk_size * 2, padding="max_length", truncation=True).input_ids).unsqueeze(1)
                    retrieval = torch.cat((chunk_1, chunk_2), dim=1)
                    retrieval = retrieval[:int(inps.shape[1] / self.chunk_size), :, :]
                    if retrieval.shape[0] != int(inps.shape[1] / self.chunk_size):
                        pad_ = torch.ones(int(inps.shape[1] / self.chunk_size) - retrieval.shape[0], retrieval.shape[1], retrieval.shape[2]) * self.tokenizer.pad_token_id
                        retrieval = torch.cat((retrieval, pad_), dim=0)
                    retrieval = retrieval.long()
                    return self.gpt2(inps.to(self.device), retrieval=retrieval.to(self.device).unsqueeze(0))[0][:, :, :50257]
                else:
                    return self.gpt2(inps)[0][:, :, :50257]
        else:
            return self.gpt2(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
