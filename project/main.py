import os.path
import pandas

from expertsystem import runex
from textinterface import user_menu, wait_user

def expert_system():
    print("Benvenuto nel sistema diagnostico con modello a sistema esperto!")
    print("-----------------------------------------------------------------------")
    runex()

if __name__ == '__main__':
    title = "\nMenù"
    options = [
        "Sistema esperto",
        "Esci"
    ]
    res = 0
    while res != 2:
        res = user_menu(title, options)
        if res == 1:
            expert_system()
            wait_user()

