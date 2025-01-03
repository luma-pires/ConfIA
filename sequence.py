class AlphabetSequenceConverter:
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def number_to_sequence(self, num):
        result = ""
        while num >= 0:
            result = self.alphabet[num % 26] + result
            num = num // 26 - 1
            if num < 0:
                break
        return result
