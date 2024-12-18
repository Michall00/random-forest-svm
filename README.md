# UMA Random forest SVM

## Opis
Celem zadania jest stworzenie hybrydowego modelu klasyfikatora, który łączy drzewa ID3 i maszyny wektorów nośnych (SVM).

### Algorytmy
1. Drzewo decyzyjne (ID3): Zaimplementujemy algorytm ID3 do budowy drzew decyzyjnych, który wybiera podział w węźle na podstawie maksymalizacji zysku informacyjnego (information gain).
2. SVM (Support Vector Machine): Będziemy korzystać z dostepnej implemenacji SVM z biblioteki [scikit-learn](https://scikit-learn.org/1.5/modules/svm.html).

### Integracja w modelu hybrydowym:

- Dla każdego klasyfikatora generujemy losowy podzbiór danych treningowych.
- Co drugi klasyfikator jest zastępowany SVM.
Wynik końcowy jest określany na podstawie głosowania większościowego.

## Autorzy
Mateusz Ostaszewski
Michał Sadowski

## Instalacja
W celu instalacji projektu należy:
1. pobrać kod z repozytorium  
   `git clone https://gitlab-stud.elka.pw.edu.pl/mostasze/uma-random-forest-svm`
2. Zainstalować wymagane biblioteki w środowisku wirtualnym  
    `make requirements`

## Implementacja
[Las losowy z SVM](random_forest_svm/hybrid_random_forest.py)
[Drzewo ID3](random_forest_svm/id3_tree/id3_tree.py)
[Eksperyment z hiperparametrami](random_forest_svm/experiments/perform_hyperparameters_experiment.py)
[Eksperyment porównawczy](random_forest_svm/experiments/comparative_experiment.py)


## Uruchomienie eksperymentów

1. Przygotownie danych  
   `make prepare_data`
2. Przeprowadzenie eksprymentów  
   `make run_experiments`
3. Przegląd wyników  
   `mlflow ui`