import csv
import pandas as pd
import os

def get_question_id(question_str):
    """
    Examples
    --------
    >>> get_question_id("q123")
    123
    """
    return int(question_str.lstrip("q"))


class Judgement(object):
    def __init__(self, questions_csv):
        self.question = [None] * (18143 + 1)
        with open(questions_csv) as f:
            f.readline()
            for i, line in enumerate(csv.reader(f, delimiter=",")):
                _id = self.get_question_id(line[0])
                correct_answer = line[3]
                self.question[_id] = (i, correct_answer)

    @staticmethod
    def get_question_id(question_id):
        if isinstance(question_id, str):
            question_id = get_question_id(question_id)
        return question_id

    def is_correct(self, question_id: (str, int), user_answer: str):
        question_id = self.get_question_id(question_id)

        if self.question[question_id] is None:
            raise ValueError("Unknown question")

        return True if self.question[question_id][1] == user_answer else False

    def __call__(self, question_id: str, user_answer: str):
        question_id = self.get_question_id(question_id)

        if self.question[question_id] is None:
            raise ValueError("Unknown question")

        _id, ground_truth = self.question[question_id]
        return _id, 1 if ground_truth == user_answer else 0


path = './questions.csv'
data = Judgement(path)
def rea(file):
    list = []
    Data = pd.read_csv(file)
    user_question = Data['question_id']
    user_answer = Data['user_answer']
    temp = zip(user_question, user_answer)
    for i, j in temp:
       question_id  = get_question_id(i)
       tag = data.is_correct(question_id,j)
       if(tag == True):
           list.append(1)
       else:
          list.append(0)
    Data['correct']  = list
    Data.to_csv(file, index=False)


for i in range(1,840474):
    stu_path = './KT1' + '/u' + str(i) +'.csv'
    if os.path.exists(stu_path):
        rea(stu_path)





