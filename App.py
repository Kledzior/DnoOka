import tkinter as tk
from DnoOka import computeAndVerify
from Klasyfikator import classifyAndVerify
from train_unet import UNET
from multiprocessing import Process
import sys
import os
from tkinter import messagebox


def main():
    root = tk.Tk()
    root.title("Dno Oka")
    root.geometry("500x500")

    label = tk.Label(root, text="Wprowadz liczbe obrazow na ktorej chcesz zbadac algorytm:")
    label.pack(pady=5)

    liczba_entry = tk.Entry(root)
    liczba_entry.pack(pady=5)

    label2 = tk.Label(root, text="Wprowadz liczbe Epoch w UNet")
    label2.pack(pady=5)
    liczba2_entry = tk.Entry(root)
    liczba2_entry.pack(pady=5)


    def uruchom_funkcje(funkcja, *args):
        try:
            liczba = int(liczba_entry.get())
            if funkcja==UNET and trainUNET.get():
                try:
                    liczbaEpoch = int(liczba2_entry.get())
                    flagaTrening = args[0]
                    p = Process(target=funkcja, args=(liczba, flagaTrening, liczbaEpoch))
                    p.start()
                except ValueError:
                    messagebox.showerror("Błąd", "Wprowadź poprawną liczbę całkowitą epoch.")
            else:
                p = Process(target=funkcja, args=(liczba, *args))
                p.start()
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawną liczbę całkowitą dla liczby obrazów.")




    trainModelGradBoost = tk.BooleanVar(value=False)
    trainUNET = tk.BooleanVar(value=False)
    checkbtn = tk.Checkbutton(
        root,
        text="Wykonaj trening Modelu Gradient Boost",
        variable=trainModelGradBoost
    )
    checkbtn.pack(pady=5)

    checkbtnUnet = tk.Checkbutton(
    root,
    text="Wykonaj trening Modelu UNET",
    variable=trainUNET
    )
    checkbtnUnet.pack(pady=5)

    btn1 = tk.Button(root, text="Filtr Frangiego wraz ze wstępnym i końcowym przetwarzaniem", command=lambda: uruchom_funkcje(computeAndVerify))
    btn2 = tk.Button(root, text="Przeprowadz klasyfikacje z wykorzystaniem GradientBoostingClassifier", command=lambda: uruchom_funkcje(classifyAndVerify, trainModelGradBoost.get()))
    btn3 = tk.Button(root, text="UNET", command=lambda: uruchom_funkcje(UNET, trainUNET.get()))
    btn1.pack(pady=10)
    btn2.pack(pady=10)
    btn3.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()