import random
import math




def calculate_score(guesses):
    worst_score = round(math.log2(100))
    if guesses < worst_score:
        return 100
    elif guesses == worst_score:
        return 50
    else:
        return round(100*(1-(guesses-worst_score)/(100-worst_score)))
def guessing_game(max_num,min_num):
    num = random.randint(min_num,max_num)
    guesses = 0
    while True:
        guess = int(input("Enter a number between 1 and 100: "))
        guesses += 1
        if num > guess:
            print("Your guess is too low.")
        elif num < guess:
            print("Your guess is too high.")
        elif guess == num:
            score = calculate_score(guesses)
            print(f"Congratulations! You guessed the number in {guesses} tries.\n Your score is {score}.")
            break

guessing_game(100,1)