from embedding import Embedding


class Check_Correction(Embedding):

    def __init__(self):
        super().__init__()

    def validate_user_correction(self, user_correction, prev_question=None, prev_answer=None, ai_model=None, method='ai'):
        methods_for_validating_user_correction = {'ai': self.validate_with_ai}
        methods_for_validating_user_correction[method](user_correction, prev_question, prev_answer, ai_model)

    def validate_with_ai(self, user_correction, prev_question, prev_answer, ai):
        prompt = (f"Considerando o diálogo abaixo, a correção do usuário está correta? Responda com Sim ou Não: \n"
                  f"Usuário: {prev_question} \n"
                  f"Resposta: {prev_answer}' \n"
                  f"Usuário: {user_correction}")
        corrections_labels = {'sim': True, 'não': False}
        detected_classes = ai.invoke(prompt)
        answer = detected_classes.content.replace('.', '').lower()
        valid_correction = corrections_labels.get(answer, False)
        return valid_correction
