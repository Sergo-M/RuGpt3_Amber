import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class TextGenerator:

    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, device: str = 'cpu', do_sample: bool = True,
                 num_beams: int = 2, temperature: float = 1.5, top_p: float = 0.9, max_length: int = 75) -> None:
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.model.eval()

    def input(self, prompt: str) -> None:
        self.input_text = prompt

    def encode(self, prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def decode(self, tensor: torch.Tensor) -> str:
        return list(map(self.tokenizer.decode, tensor))[0]

    def generate(self) -> str:
        encoded_input = self.encode(self.input_text)
        with torch.no_grad():
            out = self.model.generate(encoded_input,
                                      do_sample=self.do_sample,
                                      num_beams=self.num_beams,
                                      temperature=self.temperature,
                                      top_p=self.top_p,
                                      max_length=self.max_length,
                                      )
        return self.decode(out)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    tuned_model = 'data/tuned_model'
    model = GPT2LMHeadModel.from_pretrained(tuned_model)
    tokenizer = GPT2Tokenizer.from_pretrained(tuned_model)
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    generator.input(input('Введите слово: '))
    print(generator.generate())

if __name__ == "__main__":
    main()
