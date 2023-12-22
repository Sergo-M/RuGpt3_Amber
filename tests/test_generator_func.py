import unittest
from rugpt3_amber.data.pathes import MODEL_PATH
from rugpt3_amber.generator import TextGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TestTextGenerator(unittest.TestCase):
    def setUp(self):
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        self.device = 'cpu'
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)

    def test_input(self):
        prompt = "Input prompt"
        self.generator.input(prompt)
        self.assertEqual(self.generator.input_text, prompt)

    def test_encode(self):
        prompt = "Input prompt"
        expected_encoding = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        encoding = self.generator.encode(prompt)
        self.assertEqual(encoding.tolist(), expected_encoding.tolist())

    def test_decode(self):
        tensor = self.tokenizer.encode("MATT ", return_tensors="pt").to(self.device)
        expected_decoding = "MATT "
        decoding = self.generator.decode(tensor)
        self.assertEqual(expected_decoding, decoding)

if __name__ == '__main__':
    unittest.main()
