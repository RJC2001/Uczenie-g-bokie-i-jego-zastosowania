# Odpowiedzi

## Lab 01

### Pytanie 1

Czym różni się `tf.constant` od `tf.Variable`? Jaki jest odpowiednik zmiennej, którą da się trenować, w PyTorch?

**Odpowiedź:** `tf.constant` tworzy niezmienny tensor (stałą w grafie) - nie można go modyfikować w miejscu. `tf.Variable` to zmienna trenowalna: jest mutowalna (`assign`, `assign_add`) i domyślnie śledzona przez `GradientTape`, więc to po niej liczymy gradienty. W PyTorch odpowiednikiem jest `torch.nn.Parameter` (tensor z `requires_grad=True` zarejestrowany w module) lub zwykły tensor z `requires_grad=True`.

### Pytanie 2

Masz tensor `y_pred` o kształcie `(64, 1)` i `y` o kształcie `(64,)`. Jaki kształt będzie miało `y_pred - y` i dlaczego to groźne przy liczeniu MSE?

**Odpowiedź:** Wynik ma kształt `(64, 64)`. Broadcast wyrównuje wymiary od tyłu: `(64,)` staje się `(1, 64)`, a `(64, 1)` i `(1, 64)` rozgłaszają się do `(64, 64)`. To groźne, bo MSE policzone po takiej macierzy uśrednia wszystkie `64 x 64` kombinacji zamiast 64 próbek - strata (i gradienty) są zafałszowane. Naprawa: `y_pred.squeeze(-1)` albo `y.reshape(-1, 1)`.

### Pytanie 3

Dlaczego w pętli treningowej PyTorch musimy wywoływać `optimizer.zero_grad()` (lub `x.grad.zero_()`)? Co się stanie, jeśli o tym zapomnimy?

**Odpowiedź:** W PyTorch gradienty **akumulują się**: każde `backward()` dodaje nowy gradient do już zapisanego w `.grad`. `zero_grad()` zeruje je przed kolejnym krokiem, żeby w danej iteracji gradient dotyczył tylko bieżącej straty. Jeśli o tym zapomnimy, gradienty sumują się między iteracjami, efektywny krok rośnie z każdą epoką i aktualizacje stają się błędne - trening rozbiega się.

### Pytanie 4

Wyjaśnij, dlaczego autograd używa trybu odwrotnego (backpropagation), a nie różnic skończonych, do trenowania sieci.

**Odpowiedź:** Backpropagation to autodiff w trybie odwrotnym: jedno przejście wstecz daje gradient skalarnej straty po **wszystkich** parametrach naraz, a jego koszt jest rzędu przejścia w przód (`O(liczba parametrów)`), i jest dokładny do precyzji `float`. Różnice skończone wymagałyby osobnego przejścia w przód dla każdego parametru (`O(liczba parametrów)` przebiegów) i dają wynik tylko przybliżony (błąd obcięcia, dobór kroku `h`). Dla milionów parametrów jest to niepraktyczne.

### Pytanie 5

Co się dzieje z GD dla zbyt dużego learning rate? Jak dla straty kwadratowej wyznaczyć próg rozbieżności?

**Odpowiedź:** Dla zbyt dużego learning rate kroki przestrzeliwują minimum: strata zaczyna oscylować i rośnie - GD staje się rozbieżne. Dla straty kwadratowej z hesjanem `H` (dla MSE: `H = (2/n) X^T X`) zbieżność jest wtedy i tylko wtedy, gdy `lr < 2/lambda_max`, gdzie `lambda_max` to największa wartość własna `H`. Stąd próg rozbieżności `lr_kryt = 2/lambda_max` (u nas `lambda_max = 4.3339`, czyli `lr_kryt = 0.4615`).

### Pytanie 6

Podaj po dwa przykłady **parametrów** i **hiperparametrów** z dzisiejszych zajęć.

**Odpowiedź:** Parametry (uczone przez algorytm): wagi `w` i bias `b` regresji liniowej (w PyTorch `model.weight`, `model.bias`; w TF zmienne `w`, `b`). Hiperparametry (ustawiane przez nas): learning rate `lr` oraz liczba epok (przykłady dodatkowe: rozmiar batcha, stopień wielomianu `D`, współczynnik regularyzacji `lambda`).

### Pytanie 7

Jakie są zalety i wady mini-batch SGD w porównaniu z full-batch GD?

**Odpowiedź:** Zalety mini-batch: dużo **tańszych** kroków (wiele aktualizacji na epokę), gradient zaszumiony, co działa regularyzująco i pomaga wydostać się z minimów lokalnych/płaskich, lepsza generalizacja i wykorzystanie pamięci/GPU. Wady: gradient jest tylko przybliżony (zaszumiony), zbieżność mniej gładka i trzeba dobrać dodatkowe hiperparametry (rozmiar batcha, lr), a szum utrudnia precyzyjne dojście do optimum. Full-batch: dokładny gradient i stabilna zbieżność, ale jeden drogi krok na epokę, duże zużycie pamięci i łatwo utknąć.

### Pytanie 8

Dlaczego do wyboru stopnia wielomianu nie możemy użyć błędu na zbiorze treningowym? Do czego służy zbiór walidacyjny, a do czego testowy?

**Odpowiedź:** Błąd treningowy maleje monotonicznie wraz ze wzrostem pojemności (stopnia `D`), więc jego minimum zawsze wypada dla modelu maksymalnie złożonego - nie wykrywa przeuczenia. Dlatego stopień wybieramy na **zbiorze walidacyjnym** (dane nieużywane w treningu, ale też nie testowe) - na nim widać minimum krzywej U. **Zbiór testowy** służy do jednorazowej, bezstronnej oceny już wybranego modelu i nie wolno go używać do strojenia hiperparametrów.

## Lab 02

### Pytanie 1

Zapisz regułę uczenia perceptronu. Czym różni się od spadku gradientu? Kiedy perceptron gwarantuje zbieżność?

**Odpowiedź:**

### Pytanie 2

Dlaczego pojedynczy perceptron nie rozwiąże problemu XOR, a MLP z jedną warstwą ukrytą tak?

**Odpowiedź:**

### Pytanie 3

Dlaczego sieć bez nieliniowych funkcji aktywacji (dowolnie głęboka) jest równoważna modelowi liniowemu?

**Odpowiedź:**

### Pytanie 4

Wyjaśnij zjawisko znikającego gradientu. Dlaczego sigmoid jest na nie podatny bardziej niż ReLU?

**Odpowiedź:**

### Pytanie 5

Ile parametrów ma MLP o architekturze 10 -> 32 -> 16 -> 3? Pokaż obliczenia.

**Odpowiedź:**

### Pytanie 6

Dlaczego dla sigmoidu z BCE gradient po pobudzeniu wyjścia upraszcza się do $\hat y - y$? Jaka jest korzyść z używania `BCEWithLogitsLoss` (lub `from_logits=True`)?

**Odpowiedź:**

### Pytanie 7

Czym różni się Sequential API od Functional API w Keras? Podaj przykład architektury, której nie zbudujesz w Sequential.

**Odpowiedź:**

### Pytanie 8

Wymień kroki jednej iteracji pętli treningowej PyTorch. Co robią `model.train()`, `model.eval()` i `torch.no_grad()`?

**Odpowiedź:**

## Lab 03

### Pytanie 1

Dlaczego w ostatniej warstwie klasyfikatora na 10 klas używamy softmax, a w regresji warstwy liniowej? Co by się stało, gdyby w regresji cen domów użyć sigmoidy?

**Odpowiedź:**

### Pytanie 2

Czym różni się `sparse_categorical_crossentropy` od `categorical_crossentropy`? Kiedy używamy której?

**Odpowiedź:**

### Pytanie 3

Dlaczego `StandardScaler` dopasowujemy tylko do zbioru treningowego? Jak nazywa się błąd polegający na dopasowaniu go do całych danych i jakie ma skutki?

**Odpowiedź:**

### Pytanie 4

Ile parametrów ma warstwa `Dense(64)` po wejściu o wymiarze 8? A `BatchNormalization` po niej (ile z nich jest uczonych gradientowo)?

**Odpowiedź:**

### Pytanie 5

Co robi `restore_best_weights=True` w `EarlyStopping` i dlaczego jest ważne?

**Odpowiedź:**

### Pytanie 6

Wyjaśnij, czym Adam różni się od zwykłego SGD. Dlaczego SGD bez momentum uczy się wolniej?

**Odpowiedź:**

### Pytanie 7

Na wykresie strata walidacyjna rośnie od 8. epoki, a treningowa dalej spada. Jak to nazywamy i jakie trzy działania podejmiesz?

**Odpowiedź:**

### Pytanie 8

Jak działa Dropout w trakcie treningu, a jak w predykcji? Dlaczego przeskalowuje się aktywacje przez $1/(1-p)$?

**Odpowiedź:**

## Lab 04

### Pytanie 1

Obraz 64x64x3 przechodzi przez `Conv2D(20, kernel_size=5, strides=1, padding="valid")`. Jaki jest kształt wyjścia i ile parametrów ma warstwa?

**Odpowiedź:**

### Pytanie 2

Dlaczego warstwa splotowa ma znacznie mniej parametrów niż warstwa gęsta o porównywalnej liczbie wyjść? Wymień dwie własności splotu, które o tym decydują.

**Odpowiedź:**

### Pytanie 3

Co robi `MaxPooling2D(2)` i ile ma parametrów? Czym różni się od `GlobalAveragePooling2D`?

**Odpowiedź:**

### Pytanie 4

Dlaczego warstwy augmentacji w modelu nie zmieniają obrazów podczas `evaluate` i `predict`? Podaj przykład augmentacji, która byłaby **błędem** dla CIFAR-10, i przykład błędnej augmentacji dla MNIST.

**Odpowiedź:**

### Pytanie 5

Zapisz wzór bloku residualnego. Dlaczego połączenia residualne ułatwiają trenowanie głębokich sieci? Kiedy skrót wymaga splotu 1x1?

**Odpowiedź:**

### Pytanie 6

Opisz dwa etapy transfer learningu. Dlaczego w fine-tuningu stosujemy bardzo mały learning rate i dlaczego najpierw trenujemy samą głowę?

**Odpowiedź:**

### Pytanie 7

Co się stanie, jeśli obrazy podzielimy przez 255, a potem podamy do `mobilenet_v2.preprocess_input`?

**Odpowiedź:**

### Pytanie 8

Po co wywołujemy `baza(x, training=False)`, skoro i tak ustawiamy `baza.trainable`?

**Odpowiedź:**

## Lab 05

### Pytanie 1

Dlaczego w klasyfikacji binarnej w PyTorch używamy `BCEWithLogitsLoss` zamiast `Sigmoid` + `BCELoss`? Co zwraca ostatnia warstwa sieci?

**Odpowiedź:**

### Pytanie 2

Co się stanie, jeśli w early stopping zapiszesz `best_state = model.state_dict()` bez `deepcopy`?

**Odpowiedź:**

### Pytanie 3

Dlaczego próg decyzyjny wybieramy na walidacji, a nie na teście? Czy wybór progu zmienia ROC-AUC?

**Odpowiedź:**

### Pytanie 4

Model ma ROC-AUC = 0,95. Czy to znaczy, że jego prawdopodobieństwa są dobrze skalibrowane? Jak to sprawdzić?

**Odpowiedź:**

### Pytanie 5

Podaj trzy przykłady wycieku danych, których **nie** usunie użycie `sklearn.Pipeline`.

**Odpowiedź:**

### Pytanie 6

W selekcji cech na czystym szumie procedura z wyciekiem dała accuracy ok. 0,9. Dlaczego, i jak można by tę wadę wykryć bez znajomości prawdy?

**Odpowiedź:**

### Pytanie 7

Dlaczego naiwny przedział ufności dla średniej z wyników foldów CV jest zbyt wąski? Na czym polega poprawka Nadeau-Bengio?

**Odpowiedź:**

### Pytanie 8

Permutation importance pokazuje, że `mean radius` jest prawie nieistotna, choć klinicznie wielkość guza ma znaczenie. Jak to wyjaśnić?

**Odpowiedź:**

## Lab 06

### Pytanie 1

Dlaczego LDA może zwrócić co najwyżej $C-1$ wymiarów i czym jej cel różni się od celu PCA?

**Odpowiedź:**

### Pytanie 2

Co oznacza wartość `explained_variance_ratio_` w PCA, a co w LDA?

**Odpowiedź:**

### Pytanie 3

Kiedy wolimy TruncatedSVD od PCA?

**Odpowiedź:**

### Pytanie 4

W jakich warunkach autoenkoder rozpina tę samą podprzestrzeń co PCA? Czy jego wagi są wtedy równe składowym głównym?

**Odpowiedź:**

### Pytanie 5

Dlaczego autoenkoder odszumiający może uczyć się użytecznych cech nawet wtedy, gdy wymiar latentu nie jest mały?

**Odpowiedź:**

### Pytanie 6

Dlaczego silhouette nie nadaje się do porównywania klasteryzacji wykonanych w przestrzeniach o różnym wymiarze? Czym różnią się ARI/NMI od silhouette?

**Odpowiedź:**

### Pytanie 7

Jaki kształt wyjścia da `Conv2DTranspose(16, kernel_size=4, strides=2, padding="valid")` dla wejścia 7x7?

**Odpowiedź:**

### Pytanie 8

Jak należy wybrać próg w detekcji anomalii, aby nie popełnić wycieku danych? Podaj przykład klasy, dla której metoda błędu rekonstrukcji może zawieść.

**Odpowiedź:**

## Lab 07

### Pytanie 1

Jakie dwa składniki ma strata VAE i co by się stało, gdyby usunąć każdy z nich?

**Odpowiedź:**

### Pytanie 2

Na czym polega trik reparametryzacji i dlaczego jest potrzebny?

**Odpowiedź:**

### Pytanie 3

Dlaczego enkoder VAE zwraca $\log\sigma^2$, a nie $\sigma$?

**Odpowiedź:**

### Pytanie 4

Dlaczego próbki z VAE są zwykle bardziej rozmyte niż z GAN?

**Odpowiedź:**

### Pytanie 5

Zapisz funkcję celu minimax GAN. Czym jest *non-saturating loss* generatora i dlaczego się go używa?

**Odpowiedź:**

### Pytanie 6

Dlaczego w kroku dyskryminatora w PyTorch używamy `fake.detach()`, a w Keras liczymy gradient tylko po `discriminator.trainable_weights`?

**Odpowiedź:**

### Pytanie 7

Czym jest mode collapse i jak można go wykryć ilościowo? Jakie ograniczenia ma miara oparta na rozkładzie klas?

**Odpowiedź:**

### Pytanie 8

Dlaczego wartości strat G i D nie są dobrą miarą jakości wygenerowanych obrazów? Co stosujemy zamiast nich?

**Odpowiedź:**

**Pytania do dyskusji**

### Pytanie 9

Jak sprawdzić, czy nasz GAN nie "zapamiętał" obrazów treningowych? (wskazówka: najbliższy sąsiad w zbiorze treningowym)

**Odpowiedź:**

### Pytanie 10

Czy publikacja wytrenowanego generatora twarzy powinna wymagać ograniczeń? Jakich?

**Odpowiedź:**
