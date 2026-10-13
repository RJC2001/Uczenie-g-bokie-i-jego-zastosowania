# Sprawozdanie

## Lab 01

### 0.2 Importy, konfiguracja i tryb szybki
- `SZYBKI_TRYB = True`
- NumPy 2.2.6, TensorFlow 2.21.0 / Keras 3.12.4, PyTorch 2.14.1+cpu

### 0.3 Sprawdzenie GPU i ustawienie ziaren losowości (seedów)
- GPU niewidoczne, `DEVICE = cpu`

### Zadanie 1.1 - Standaryzacja batcha obrazów (broadcasting, NHWC vs NCHW)
- TF średnie per kanał = [0, -0, -0], std = [1, 1, 1]
- PyTorch średnie per kanał = [0, 0, 0]
- Zgodność TF vs PyTorch = `True`

### Zadanie 1.2 - Macierz odległości bez pętli (wektoryzacja)
- Zgodność z wersją pętlową: NumPy `True`, Torch `True`
- n = 150: pętla 0.120 s vs wektorowo 0.00614 s (przyspieszenie ~20x)

### 2.4 Gradient na całej siatce punktów i porównanie z pochodną analityczną
- maks. |TF - analityczna| = 3.8e-06
- maks. |PyTorch - analityczna| = 1.9e-06

![Autograd vs analityczna](img/lab01_2_4_autograd_vs_analityczna.png)

### Zadanie 2.1 - Gradient funkcji dwóch zmiennych w TF i PyTorch
- analitycznie: (0.27105, -0.65559)
- TF: (0.27105, -0.65559)
- PyTorch: (0.27105, -0.65559)

### Zadanie 2.2 - Sprawdzanie gradientu różnicami skończonymi (gradient check)
- błąd względny (h = 1e-5) = 1.47e-11

![Gradient check](img/lab01_2_2_gradient_check.png)

### 3. Regresja liniowa od zera
- GD: w = 2.5128, b = -1.0130, MSE = 0.9589
- `polyfit`: w = 2.5128, b = -1.0130

![Dane przykładu](img/lab01_3_1_dane_przyklad.png)

![GD 1D](img/lab01_3_gd_1d.png)

### 3.2 Powierzchnia straty i trajektoria spadku gradientu

![Powierzchnia straty](img/lab01_3_2_powierzchnia_straty.png)

### 3.3 Prawdziwe dane: California Housing
- cechy: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- X: (2000, 8), y: (2000,); train: (1600, 8), test: (400, 8); standaryzacja na statystykach zbioru treningowego
- baseline (średnia): MSE test = 0.9137
- najmniejsze kwadraty (odniesienie): MSE train = 0.2818, test = 0.241

### Zadanie 3.1 - Wektorowy spadek gradientu w NumPy (wiele cech)
- GD (50 epok, lr = 0.2): czas 0.03 s, MSE train = 0.2829 vs optimum = 0.2818 (0.37%)
- MSE test GD = 0.2371
- `max|w_gd - w_ls| = 0.1456`

![GD wielocechowe](img/lab01_3_1_gd_wielocechowe_a.png)

![GD wielocechowe](img/lab01_3_1_gd_wielocechowe_b.png)

### 3.4 Ta sama regresja w PyTorch - najpierw "ręcznie" z autograd
- PyTorch (autograd): MSE train = 0.2829, `max|w_torch - w_ls| = 0.1456`

### 3.5 ...i w "standardowym" stylu PyTorch: `nn.Linear` + optymalizator
- `nn.Linear` + SGD: MSE train = 0.2859, test = 0.2344

### Zadanie 3.2 - Regresja liniowa w TensorFlow z `GradientTape`
- MSE train: NumPy 0.28288 | PyTorch 0.28288 | TF 0.28288

![TensorFlow GradientTape](img/lab01_3_2_tf_gradienttape.png)

### Zadanie 4.1 - Eksperyment: za mały, dobry i za duży learning rate
- `lambda_min = 0.0480`, `lambda_max = 4.3339`, uwarunkowanie = 90.3
- `lr_kryt = 2/lambda_max = 0.4615`
- `lr = 0.4522` (0.98 * lr_kryt): zbieżność w 8 epokach do 1% od optimum
- `lr = 0.4846` (1.05 * lr_kryt): rozbieżność (MSE = 2.967) - hipoteza o progu `2/lambda_max` potwierdzona doświadczalnie

| lr | MSE po ostatniej epoce | epoki do 1% od optimum |
|---|---|---|
| 0.001 | 3.98 | nie osiągnięto |
| 0.01 | 1.155 | nie osiągnięto |
| 0.1 | 0.2837 | 24 |
| 0.2307 | 0.2830 | 10 |
| 0.4522 | 0.2824 | 8 |
| 0.4846 | 2.967 | nie osiągnięto |

![Wpływ learning rate](img/lab01_4_1_learning_rate.png)

### Zadanie 5.1 - Implementacja mini-batch SGD i porównanie rozmiarów batcha
- 4 epoki, lr = 0.05
- B = 16: MSE = 0.3558
- B = 64: MSE = 0.2928
- B = 256: MSE = 0.2974
- full-batch: MSE = 2.1097 (przy tak małym lr i 4 epokach jeszcze daleko od zbieżności)

![Mini-batch vs full-batch](img/lab01_5_1_minibatch.png)

### 5.1 Mini-batche "po bibliotecznemu": `DataLoader` w PyTorch
- `DataLoader` + SGD (B = 64): MSE train = 0.4828

### 6. Regresja wielomianowa i przeuczenie (overfitting)

![Dane wielomianowe](img/lab01_6_dane_wielomian.png)

### Zadanie 6.1 - Krzywa train vs walidacja w funkcji stopnia wielomianu
- najlepszy stopień wg walidacji: D = 3, MSE val = 0.0911

![Stopień wielomianu](img/lab01_6_1_stopien_wielomianu.png)

### Zadanie 6.2 - Regularyzacja L2 (ridge) jako "lekarstwo" na przeuczenie
- najlepsza `lambda = 3.16e-05`, MSE val = 0.1050

![Ridge](img/lab01_6_2_ridge.png)

### Zadanie dodatkowe (dla chętnych) - GD, momentum i Adam na funkcji Rosenbrocka
- start (-1.5, 2), 300 kroków
- odległość od minimum (1,1): GD (lr=1e-3) = 2.314, momentum (lr=2e-4) = 2.046, Adam (lr=0.02) = 2.065
- momentum (lr=1e-3) = 0.384 (najbliżej minimum)

![Rosenbrock](img/lab01_dodatkowe_rosenbrock.png)

### Wnioski (Lab 01)
- Poprawność implementacji weryfikowano liczbowo: kontrolowano kształty i typy tensorów pośrednich, a ręcznie wyprowadzony gradient porównano z różnicami centralnymi w `float64` (błąd względny `1.5e-11`). Wektoryzowaną macierz odległości zestawiono z wersją pętlową - wyniki zgodne, a czas skrócony z 0.120 s do 0.00614 s (ok. 20x).
- Standaryzacja cech na statystykach zbioru treningowego poprawiła uwarunkowanie zadania (`cond = 90.3`), dzięki czemu GD z learning rate bliskim `lr_kryt` osiągał zbieżność w kilka epok.
- Learning rate okazał się najważniejszym hiperparametrem: wartości znacznie poniżej `lr_kryt` nie dawały zbieżności w zadanym budżecie epok, wartości w pobliżu `lr_kryt` dawały zbieżność najszybciej, a przekroczenie progu `2/lambda_max` prowadziło do rozbieżności - zgodnie z teorią dla straty kwadratowej.
- Pojemność modelu steruje kompromisem między niedouczeniem a przeuczeniem: krzywa walidacyjna ma minimum dla stopnia D = 3, a powyżej niego błąd walidacji rośnie przy wciąż malejącym błędzie treningowym. Stopień wybierano na zbiorze walidacyjnym, a zbiór testowy posłużył do jednorazowej oceny - błąd treningowy nie może być kryterium wyboru, ponieważ maleje monotonicznie wraz z pojemnością.
- Regularyzacja L2 poprawiła dopasowanie modelu o dużej pojemności (D = 15); optymalna wartość współczynnika wyniosła `lambda = 3.16e-05`.
- Przy stałym learning rate mini-batch SGD wykonywał więcej, tańszych kroków na epokę i szybciej zmniejszał błąd niż gradient pełny; trening na całym zbiorze okazał się w tym budżecie najwolniejszy.

## Lab 02

## Lab 03

## Lab 04

## Lab 05

## Lab 06

## Lab 07
