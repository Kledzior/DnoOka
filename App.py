import tkinter as tk
from DnoOka import computeAndVerify
from Klasyfikator import classifyAndVerify
from multiprocessing import Process
import sys
import os


def main():
    root = tk.Tk()
    root.title("Dno Oka")
    root.geometry("500x500")

    label = tk.Label(root, text="Wprowadz liczbe obrazow na ktorej chcesz zbadac algorytm:")
    label.pack(pady=5)

    liczba_entry = tk.Entry(root)
    liczba_entry.pack(pady=5)


    def uruchom_funkcje(funkcja):
        try:
            liczba = int(liczba_entry.get())
            p = Process(target=funkcja,args=(liczba,))
            p.start()
            
        except ValueError:
            print("Wprowadź poprawną liczbę całkowitą.")


    
    btn1 = tk.Button(root, text="Filtr Frangiego wraz ze wstępnym i końcowym przetwarzaniem", command=lambda: uruchom_funkcje(computeAndVerify))
    btn2 = tk.Button(root, text="Przeprowadz klasyfikacje z wykorzystaniem GradientBoostingClassifier", command=lambda: uruchom_funkcje(classifyAndVerify))

    btn1.pack(pady=10)
    btn2.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()