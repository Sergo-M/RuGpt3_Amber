"""
this module generating text from input string
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rugpt3_amber.data.pathes import MODEL_PATH


class TextGenerator:
    """
    class that generate text from input
    """

    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                 device: str = 'cpu', max_length: int = 75) -> None:
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.do_sample = True
        self.num_beams = 2
        self.temperature = 1.5
        self.top_p = 0.9
        self.max_length = max_length
        self.model.eval()
        self.input_text = None

    def input(self, prompt: str) -> None:
        """
        getting input str
        """
        self.input_text = prompt

    def encode(self, prompt: str) -> torch.Tensor:
        """
        encoding input str
        """
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def decode(self, tensor: torch.Tensor) -> str:
        """
        decoding model output
        """
        return list(map(self.tokenizer.decode, tensor))[0]

    def generate(self) -> str:
        """
        generating output str
        """
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
    """
    running script
    """
    device = 'cuda' if torch.cuda.is_available() \
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    generator.input(input('Введите слово: '))
    print(generator.generate())

if __name__ == "__main__":
    main()
