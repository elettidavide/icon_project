def ask_question(question: str) -> bool:
    answer = input(question + " (si/no): ")
    while not is_answer(answer):
        print("Il valore inserito non è valido!")
        answer = input(question + " (si/no): ")

    answer = answer.lower()
    return answer == "si"

def is_answer(answer: str) -> bool:
    answer = answer.lower()
    return answer == "si" or answer == "no"

def user_menu(title: str, options: list[str]) -> int:
    response = 0
    while response == 0:
        print(title + ":")
        i = 1
        for option in options:
            print(str(i) + ") " + option)
            i = i + 1
	
        print("-----------------------------------------------------------------------")
        response = int(input("Seleziona (1)sistema esperto - (2)esci : "))
        if (response < 1) or (response > len(options)): response = 0

    return response

def wait_user(): input("Premi un pulsante qualsiasi per continuare...")