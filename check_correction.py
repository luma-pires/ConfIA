
class Check_Correction:

    def __init__(self):
        super().__init__()

    def validate_user_info(self, user_correction, ai_model=None, method='ai'):
        """ Por enquanto há apenas um método de verificação das informações. Futuramente podem ser inseridos métodos
        alternativos para checar a validade de uma informação do usuário. Uma das alternativas pode ser pequisar a
        informação na internet e verificar sua similaridade (embeddings) com o afirmado pelo usuário. Uma combinação
        dos métodos também é um bom caminho para tornar a verificação mais robusta."""
        methods_for_validating_user_correction = {'ai': self.validate_with_ai}
        return methods_for_validating_user_correction[method](user_correction, ai_model)

    @staticmethod
    def validate_with_ai(user_correction, ai):
        """ Aqui utilizamos o próprio LLM para obter a validade da correção do usuário. O chatbot só aceitará/salvará
        informações  que forem verídicas."""
        prompt = (f"Essa informação é falsa: {user_correction}? Responda apenas com sim ou não. Se for algo que você "
                  f"não consegue afirmar se é falso ou não, como por exemplo informações subjetivas, características"
                  f" sobre o usuário, etc, retorne não.")
        detected_classes = ai.invoke(prompt)
        answer = detected_classes.content.replace('.', '').lower()
        print(user_correction)
        print(answer)
        corrections_labels = {'não': True, 'sim': False}
        valid_correction = corrections_labels.get(answer, False)
        return valid_correction
