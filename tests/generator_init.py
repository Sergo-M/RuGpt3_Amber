import unittest
from RuGpt3_Amber.rugpt3_amber.data.pathes import MODEL_PATH
from RuGpt3_Amber.rugpt3_amber.generator import TextGenerator
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TestTextGenerator(unittest.TestCase):
    def setUp(self):
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        self.device = 'cpu'
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)

    def test_init(self):
        self.assertEqual(self.generator.device, self.device)
        self.assertEqual(self.generator.model, self.model.to(self.device))
        self.assertEqual(self.generator.tokenizer, self.tokenizer)
        self.assertEqual(self.generator.do_sample, True)
        self.assertEqual(self.generator.num_beams, 2)
        self.assertEqual(self.generator.temperature, 1.5)
        self.assertEqual(self.generator.top_p, 0.9)
        self.assertEqual(self.generator.max_length, 75)
        self.assertEqual(self.generator.model.training, False)


if __name__ == '__main__':
    unittest.main()
