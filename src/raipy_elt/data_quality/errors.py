


class UncoveredAssessmentError(Exception):
    def __init__(self, file_with: str, file_without: str, missing_ids: list[int], *args):
        super().__init__(args)
        self.file_with = file_with
        self.file_without = file_without
        self.missing_ids = missing_ids