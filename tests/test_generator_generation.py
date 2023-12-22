import unittest
from rugpt3_amber.generator import TextGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TestTextGenerator(unittest.TestCase):
    def setUp(self):
        self.model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
        self.device = 'cpu'
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)

    def test_generate(self):
        self.generator.input("Input prompt")
        generated_text = self.generator.generate()
        self.assertIsInstance(generated_text, str)

if __name__ == '__main__':
    unittest.main()
