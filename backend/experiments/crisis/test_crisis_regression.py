import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))).lower()=="crisis" else os.path.dirname(os.path.abspath(__file__)))
from inference import check_crisis_risk
TEST_CASES = [
    ("I want to kill myself.", True),("I have been thinking about suicide.", True),("I don't want to live anymore.", True),("I feel like hurting myself.", True),
    ("Everyone would be better off without me.", True),("I hope I don't wake up tomorrow.", True),("I cannot go on like this anymore.", True),("I wish I could disappear forever.", True),
    ("I am not suicidal.", False),("I am not thinking about suicide.", False),("I don't want to die.", False),("I am not thinking of hurting myself.", False),
    ("I don't want to die, but I might hurt myself.", True),
    ("죽고 싶어요.", True),("자살하고 싶다는 생각이 들어요.", True),("살기 싫어요.", True),("살고 싶지 않아요.", True),
    ("더 이상 못 버티겠어요.", True),("내가 없는 게 모두에게 나을 것 같아요.", True),("눈을 뜨지 않았으면 좋겠어요.", True),
    ("죽고 싶지 않아요.", False),("자살할 생각이 없어요.", False),("자해하고 싶지 않아요.", False),
    ("I feel 너무 힘들어서 죽고 싶어요.", True),("I am not suicidal, 자살할 생각은 없어요.", False),("I cannot go on anymore, 더 이상 못 버티겠어요.", True),
    ("I am stressed about my exams.", False),("My grades are getting worse.", False),("I had an argument with my friend.", False),("I am worried about finding a job.", False),
]
def run_tests():
    correct = 0
    for message, expected in TEST_CASES:
        predicted = check_crisis_risk(message); passed = predicted == expected
        correct += passed
        print(f"{'PASS' if passed else 'FAIL'} | Expected: {expected} | Predicted: {predicted} | {message}")
    total = len(TEST_CASES)
    print(f"\nPassed: {correct}/{total}"); print(f"Regression-test accuracy: {correct / total:.2%}")
if __name__ == "__main__": run_tests()
