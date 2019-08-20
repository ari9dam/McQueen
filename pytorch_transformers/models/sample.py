from typing import List,Union


class MCQSample(object):

    """A single training/test MCQ example"""

    def __init__(self,
                 id,
                 premises: Union[List[str],List[List[str]]],
                 choices: List[str],
                 label: int=None,
                 question : str = None):
        self.id = id
        number_of_choices = len(choices)
        if isinstance(premises[0], str):
            premises = [premises] * number_of_choices

        self.premises = premises
        self.question = question
        self.choices = choices
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.id),
            "premises: {}".format(self.premises),
            "choices: {}".format(self.choices),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        if self.question is not None:
            l.append("question: {}".format(self.question))

        return ", ".join(l)
