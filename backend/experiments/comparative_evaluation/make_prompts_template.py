"""Writes prompts.csv: 48 starter messages (24 contexts x English/Korean), strategies rotated
evenly. REVIEW/EDIT the messages (especially Korean), and add more rows (copy and reword)
to reach ~120. Add a few informational questions with emotion=neutral, stress_type=general."""
import itertools, pandas as pd
M = {
("anxious","academic"):("I have three exams next week and I can't stop worrying that I'll fail.","다음 주에 시험이 세 개나 있는데 떨어질까 봐 계속 걱정돼요."),
("anxious","career"):("I'm worried I won't find a job after graduation.","졸업하고 취업을 못 할까 봐 너무 불안해요."),
("anxious","interpersonal"):("I'm nervous about talking to my roommate about the noise.","룸메이트에게 소음 문제를 이야기해야 하는데 말을 꺼내기가 너무 불안해요."),
("anxious","general"):("I've felt anxious all the time lately and I don't know why.","요즘 이유 없이 계속 불안해요."),
("overwhelmed","academic"):("I have so many assignments due that I don't know where to start.","과제가 너무 많아서 어디서부터 시작해야 할지 모르겠어요."),
("overwhelmed","career"):("Between internship applications and classes, I feel like I'm drowning.","인턴 지원이랑 수업을 같이 하려니 감당이 안 돼요."),
("overwhelmed","interpersonal"):("My group project teammates keep asking me for help and I can't keep up.","팀플 조원들이 계속 도움을 요청해서 제가 감당이 안 돼요."),
("overwhelmed","general"):("Everything feels like too much right now.","요즘 모든 게 너무 벅차게 느껴져요."),
("sad","academic"):("I failed my midterm and I feel really down about it.","중간고사를 망쳐서 너무 우울해요."),
("sad","career"):("I got rejected from another job application and I feel discouraged.","또 입사 지원에서 떨어져서 많이 낙담했어요."),
("sad","interpersonal"):("My close friend stopped talking to me and I feel very lonely.","친한 친구가 연락을 끊어서 너무 외로워요."),
("sad","general"):("I've been feeling sad and low on energy this week.","이번 주 내내 기분이 가라앉고 기운이 없어요."),
("angry","academic"):("My professor graded my paper unfairly and I'm furious.","교수님이 제 과제를 불공정하게 채점해서 너무 화가 나요."),
("angry","career"):("My internship supervisor takes credit for my work and it makes me so angry.","인턴 담당자가 제 성과를 가로채서 정말 화가 나요."),
("angry","interpersonal"):("My roommate keeps ignoring our cleaning agreement and I'm really annoyed.","룸메이트가 청소 약속을 계속 어겨서 너무 짜증나요."),
("angry","general"):("I'm angry about everything lately and I snap at people.","요즘 사소한 일에도 화가 나서 사람들에게 짜증을 내요."),
("hopeful","academic"):("I think I can raise my GPA this semester and I want a plan.","이번 학기에 학점을 올릴 수 있을 것 같아서 계획을 세우고 싶어요."),
("hopeful","career"):("I have an interview next week and I'm excited but want to prepare.","다음 주에 면접이 있는데 기대되고, 잘 준비하고 싶어요."),
("hopeful","interpersonal"):("I just joined a club and I hope to make good friends.","동아리에 새로 들어갔는데 좋은 친구를 사귀고 싶어요."),
("hopeful","general"):("I feel better than I have in months and want to keep it going.","몇 달 만에 기분이 좋아져서 이 상태를 유지하고 싶어요."),
("neutral","academic"):("How should I organize my study schedule for finals?","기말고사 공부 일정을 어떻게 짜면 좋을까요?"),
("neutral","career"):("What should I put on my resume as a student with no work experience?","경력이 없는 학생은 이력서에 무엇을 써야 하나요?"),
("neutral","interpersonal"):("How do I start a conversation with classmates I don't know?","잘 모르는 같은 과 친구들에게 어떻게 말을 걸면 좋을까요?"),
("neutral","general"):("Can you tell me what kind of support is available for students?","학생을 위한 지원에는 어떤 것들이 있는지 알려주세요."),
}
EM = ["anxious","overwhelmed","sad","angry","hopeful","neutral"]
ST = ["academic","career","interpersonal","general"]
SS = ["validate_listen","validate_ask","validate_suggest","validate_normalize"]
rows = []
for i,(e,s,l) in enumerate(itertools.product(EM, ST, ["en","ko"])):
    rows.append(dict(id=i+1, language=l, emotion=e, stress_type=s, strategy=SS[i//2 % 4],
                     message=M[(e,s)][0 if l=="en" else 1]))
pd.DataFrame(rows).to_csv("prompts.csv", index=False, encoding="utf-8-sig")
print("Wrote prompts.csv with", len(rows), "rows")