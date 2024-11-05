import random
questions = 5
operation = ['+','-','*','/']
count = 0

for i in range(questions):
    num1= random.randint(1,10)
    num2= random.randint(1,10)
    op = random.choice(operation)
    correct_answer = 0

    if op == '+':
        correct_answer = num1 + num2
    elif op == '-':
        correct_answer = num1 - num2
    elif op == '*':
        correct_answer = num1 * num2
    elif op == '/':
        if num2 ==0:
            correct_answer= 0
            print("Error: Division by zero")
        else:
            correct_answer = num1 / num2
    user_answer = input(f"{num1} {op} {num2} = ")   
    if float(user_answer) == correct_answer:
        print("Correct!")
        count+=1
    else:
        print(f"Incorrect. The correct answer is {correct_answer}")

print(f"You got {count} out of {questions} correct.")