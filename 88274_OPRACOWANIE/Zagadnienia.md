# Zagadnienia

## 0. Przepływ od zera

1. Zdefiniuj zadanie i metrykę sukcesu.
2. Podziel dane na train / val / test, zanim cokolwiek policzysz.
3. Zbuduj baseline i dopiero potem sieć.
4. Dobierz model i funkcję straty do typu zadania.
5. Trenuj, patrząc na walidację (test zostaw na koniec).
6. Zdiagnozuj underfitting / overfitting i dobierz regularyzację.
7. Oceń na teście raz, z niepewnością wyniku.
8. Zapisz wniosek i jedną propozycję usprawnienia.

## 1. Środowisko i powtarzalność
_Lab 01_

- Ustaw seedy (Python, NumPy, TensorFlow, PyTorch), aby wyniki były powtarzalne.
- Pracuj w `float32`; przed każdą operacją sprawdzaj `shape`, `dtype` i urządzenie.
- Pamiętaj o układzie: NHWC w TensorFlow, NCHW w PyTorch.
- Wektoryzuj zamiast pętli Pythona; GPU przyspiesza duże operacje macierzowe (przy pomiarach `synchronize`).

## 2. Dane
_Lab 01, 03, 05_

- Najpierw ustal typ zadania: klasyfikacja, regresja, generowanie, redukcja wymiaru, klasteryzacja.
- Podziel dane na train / val / test, zanim cokolwiek policzysz; hiperparametry wybieraj na walidacji, test użyj raz.
- Normalizuj i standaryzuj na statystykach zbioru treningowego; skalowanie trzymaj w `Pipeline`.
- Wyciek danych: przetwarzanie i selekcja cech dopasowywane w każdym foldzie, podział według jednostki wdrożenia (pacjent, ośrodek, czas), cechy tylko z informacji dostępnej w chwili predykcji.
- Standaryzacja cech poprawia uwarunkowanie problemu i tempo uczenia.

## 3. Model
_Lab 02, 03, 04, 06, 07_

- Baseline obowiązkowy: średnia, regresja liniowa lub prosty klasyfikator.
- Warstwa gęsta ma `(n_in + 1) * n_out` parametrów.
- Nieliniowe aktywacje są niezbędne - bez nich dowolnie głęboka sieć jest modelem liniowym. Ukryte: ReLU/GELU; wyjście: sigmoid (binarna), softmax (wieloklasowa), liniowe (regresja).
- CNN: wymiar wyjścia `floor((n + 2p - k) / s) + 1`; parametry `k^2 * C_in * F + F`. Lokalność i współdzielenie wag dają mniej parametrów niż warstwa gęsta; `GlobalAveragePooling` zamiast `Flatten`; blok residualny `y = ReLU(F(x) + x)`, projekcja 1x1 przy zmianie wymiarów.
- Autoenkoder: enkoder + wąskie gardło + dekoder, uczony `fit(x, x)`. Nieliniowy AE rekonstruuje lepiej niż PCA przy tym samym wymiarze; liniowy AE z MSE jest równoważny PCA. Warianty: DAE (szum -> czysty obraz), CAE (`Conv2D` ze stride do zwijania, `Conv2DTranspose` do rozwijania).
- VAE: strata = rekonstrukcja + beta * KL; trik reparametryzacji `z = mu + sigma * epsilon`; beta steruje kompromisem między regularnością latentu a jakością rekonstrukcji.
- GAN: gra generatora i dyskryminatora, trening naprzemienny, generator używa non-saturating loss; zasady DCGAN (stride zamiast pooling, BN w G, LeakyReLU w D, tanh na wyjściu, Adam 2e-4 z beta1 0.5).
- Transfer learning dwuetapowo: (1) zamrożona baza + nowa głowa, (2) fine-tuning górnych warstw z małym learning rate (ok. 1e-5). Zawsze właściwy `preprocess_input` i `training=False` dla bazy z BN.

## 4. Trening
_Lab 01, 02, 03, 04, 05_

- Uczenie = dane + model + strata + optymalizator; pętla: forward -> strata -> gradient -> krok.
- PyTorch: `zero_grad` -> forward -> `backward` -> `step`; pamiętaj o `train()` / `eval()` / `no_grad()`.
- Keras: `Sequential`/`Functional` -> `compile` -> `fit(validation_data=...)` -> `evaluate` / `predict`.
- Strata do zadania: klasyfikacja `sparse_categorical_crossentropy` (etykiety całkowite) lub `categorical_crossentropy`; binarna na logitach + `BCEWithLogitsLoss`; regresja MSE, a raportuj MAE/RMSE.
- Optymalizator: Adam (1e-3) to dobry start, SGD z momentum solidna alternatywa; learning rate jest najważniejszym hiperparametrem. Harmonogramy: `CosineDecay`, `ReduceLROnPlateau`.
- Learning rate: za mały - wolno, za duży - rozbieżność; dla straty kwadratowej próg `2 / lambda_max`.
- Mini-batch SGD: więcej tańszych, zaszumionych kroków - standard w Deep Learningu.
- Inicjalizacja wag: He dla ReLU.
- Callbacki: `EarlyStopping(restore_best_weights=True)`, `ModelCheckpoint(save_best_only=True)`, `ReduceLROnPlateau`.
- Zapis modelu: `.keras` (pełny model, wznowienie treningu) lub `.weights.h5` (same wagi).
- Gradient check (różnice centralne w `float64`) weryfikuje ręcznie liczone pochodne.

## 5. Diagnostyka i regularyzacja
_Lab 01, 02, 03, 04_

- Krzywe uczenia to podstawowa diagnostyka: rosnąca strata walidacyjna przy malejącej treningowej = przeuczenie; obie wysokie = niedouczenie lub zły learning rate.
- Regularyzacja: dropout i L2 zmniejszają lukę train-val; BatchNorm głównie stabilizuje trening; augmentacja działa tylko w treningu; najlepsze lekarstwo to więcej danych.
- Znikający gradient: sigmoid ma pochodną <= 0.25, więc głębokie sieci nim nasycone się nie uczą; pomaga ReLU plus inicjalizacja He, dalej BatchNorm i połączenia residualne.

## 6. Ewaluacja
_Lab 05, 06_

- Klasyfikacja: ROC-AUC (ranking), PR-AUC (przy niezbalansowaniu), kalibracja (Brier), metryki progowe zależne od decyzji.
- Regresja: MAE, RMSE, R^2 oraz analiza reszt (struktura, heteroskedastyczność, ucięcie celu).
- Próg decyzyjny dobieraj na walidacji według kosztów błędów; sam wybór progu nie zmienia ROC-AUC.
- Kalibracja prawdopodobieństw: skalowanie Platta.
- Walidacja krzyżowa z poprawką Nadeau-Bengio (naiwny przedział jest zbyt wąski); bootstrap; permutation importance na danych wstrzymanych (ważność dotyczy modelu, nie przyczynowości, uwaga na korelacje cech).
- Random search z rozkładami log-jednostajnymi; najlepszy wynik CV jest optymistyczny - rzetelną ocenę daje test lub zagnieżdżona CV.
- Klasteryzacja: ARI/NMI gdy są etykiety, silhouette bez etykiet (ostrożnie przy porównaniach między różnymi wymiarami).

## 7. Rzetelność i wnioski
_Lab 05, 07_

- Porównuj modele na tym samym zbiorze testowym.
- Powtarzaj eksperymenty z różnymi seedami; raportuj średnią i odchylenie.
- Katalog pułapek: wyciek przy skalowaniu i selekcji cech, wyciek przez podział bez grup, użycie testu więcej niż raz, wyciek przez strojenie hiperparametrów (potrzebna zagnieżdżona CV), ucięcie celu w regresji, korelacje cech w importance, mode collapse w GAN.
- Każdy wynik domykaj schematem: hipoteza -> eksperyment -> wniosek.
- Dodaj jedną propozycję usprawnienia opartą na istniejących bibliotekach.
