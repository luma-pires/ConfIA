from embedding import Embedding


class Check_Correction(Embedding):

    def __init__(self, tool):
        super().__init__()
        self.tool = tool

    def validate_user_correction(self, user_correction):
        search_results = self.search_web_for_correction(user_correction)
        consistency_check = self.compare_with_search_results(user_correction, search_results)
        return consistency_check

    def search_web_for_correction(self, query):
        search_results = self.tool.search(query)
        return search_results

    def compare_with_search_results(self, user_correction, search_results):
        is_consistent = False
        for result in search_results:
            if user_correction.lower() in result['metadata']['original_question'].lower():
                is_consistent = True
                break
        return is_consistent

