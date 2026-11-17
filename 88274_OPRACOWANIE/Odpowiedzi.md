# Odpowiedzi

## Lab 01

### Pytanie 1

Czym różni się `tf.constant` od `tf.Variable`? Jaki jest odpowiednik zmiennej, którą da się trenować, w PyTorch?

**Odpowiedź:** `tf.constant` tworzy niezmienny tensor (stałą w grafie) - nie można go modyfikować w miejscu. `tf.Variable` to zmienna trenowalna: jest mutowalna (`assign`, `assign_add`) i domyślnie śledzona przez `GradientTape`, więc to po niej liczymy gradienty. W PyTorch odpowiednikiem jest `torch.nn.Parameter` (tensor z `requires_grad=True` zarejestrowany w module) lub zwykły tensor z `requires_grad=True`.

### Pytanie 2

Masz tensor `y_pred` o kształcie `(64, 1)` i `y` o kształcie `(64,)`. Jaki kształt będzie miało `y_pred - y` i dlaczego to groźne przy liczeniu MSE?

**Odpowiedź:** Wynik ma kształt `(64, 64)`. Broadcast wyrównuje wymiary od tyłu: `(64,)` staje się `(1, 64)`, a `(64, 1)` i `(1, 64)` rozgłaszają się do `(64, 64)`:
$$(64, 1) - (64,) \;\longrightarrow\; (64, 1) - (1, 64) \;\longrightarrow\; (64, 64).$$
To groźne, bo MSE policzone po takiej macierzy uśrednia wszystkie `64 x 64` kombinacji zamiast 64 próbek - strata (i gradienty) są zafałszowane. Naprawa: `y_pred.squeeze(-1)` albo `y.reshape(-1, 1)`.

### Pytanie 3

Dlaczego w pętli treningowej PyTorch musimy wywoływać `optimizer.zero_grad()` (lub `x.grad.zero_()`)? Co się stanie, jeśli o tym zapomnimy?

**Odpowiedź:** W PyTorch gradienty **akumulują się**: każde `backward()` dodaje nowy gradient do już zapisanego w `.grad`. Po $t$ krokach bez zerowania mamy więc
$$g_t = \sum_{k=1}^{t}\nabla_\theta L_k \quad\text{zamiast}\quad \nabla_\theta L_t,$$
a aktualizacja używa tego sumarycznego gradientu. `zero_grad()` zeruje go przed kolejnym krokiem, żeby w danej iteracji gradient dotyczył tylko bieżącej straty. Jeśli o tym zapomnimy, gradienty sumują się między iteracjami, efektywny krok rośnie z każdą epoką i aktualizacje stają się błędne - trening rozbiega się.

### Pytanie 4

Wyjaśnij, dlaczego autograd używa trybu odwrotnego (backpropagation), a nie różnic skończonych, do trenowania sieci.

**Odpowiedź:** Backpropagation to autodiff w trybie odwrotnym: jedno przejście wstecz daje gradient skalarnej straty po **wszystkich** parametrach naraz, a jego koszt jest stałym czynnikiem rzędu jednego przejścia w przód - nie zależy od liczby parametrów $P$:
$$\nabla_\theta L = \left[\frac{\partial L}{\partial \theta_1}, \dots, \frac{\partial L}{\partial \theta_P}\right] \quad\text{- koszt } O(1)\text{ przejść w przód}.$$
Różnice skończone wymagałyby osobnego przejścia w przód dla każdego parametru:
$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + h e_i) - L(\theta - h e_i)}{2h} \quad\text{- koszt } O(P)\text{ przejść w przód},$$
i dają wynik tylko przybliżony (błąd obcięcia, dobór kroku $h$). Dla milionów parametrów jest to niepraktyczne.

### Pytanie 5

Co się dzieje z GD dla zbyt dużego learning rate? Jak dla straty kwadratowej wyznaczyć próg rozbieżności?

**Odpowiedź:** Dla zbyt dużego learning rate kroki przestrzeliwują minimum: strata zaczyna oscylować i rośnie - GD staje się rozbieżne. Dla straty kwadratowej z hesjanem $H$ (dla MSE: $H = \frac{2}{n}X^\top X$) zbieżność jest wtedy i tylko wtedy, gdy
$$\eta < \frac{2}{\lambda_{max}(H)},$$
gdzie $\lambda_{max}$ to największa wartość własna $H$. Stąd próg rozbieżności
$$\eta_{crit} = \frac{2}{\lambda_{max}}$$
(u nas $\lambda_{max} = 4.3339$, czyli $\eta_{crit} = 0.4615$).

### Pytanie 6

Podaj po dwa przykłady **parametrów** i **hiperparametrów** z dzisiejszych zajęć.

**Odpowiedź:** Parametry (uczone przez algorytm): wagi `w` i bias `b` regresji liniowej (w PyTorch `model.weight`, `model.bias`; w TF zmienne `w`, `b`). Hiperparametry (ustawiane przez nas): learning rate `lr` oraz liczba epok (przykłady dodatkowe: rozmiar batcha, stopień wielomianu `D`, współczynnik regularyzacji `lambda`).

### Pytanie 7

Jakie są zalety i wady mini-batch SGD w porównaniu z full-batch GD?

**Odpowiedź:** Pełny gradient i jego estymator z mini-batcha $B$:
$$g = \frac1n\sum_{i=1}^n\nabla_\theta\ell_i(\theta), \qquad g_B = \frac1{|B|}\sum_{i\in B}\nabla_\theta\ell_i(\theta).$$
Zalety mini-batch: dużo **tańszych** kroków (wiele aktualizacji na epokę), gradient zaszumiony, co działa regularyzująco i pomaga wydostać się z minimów lokalnych/płaskich, lepsza generalizacja i wykorzystanie pamięci/GPU. Wady: gradient jest tylko przybliżony (zaszumiony), zbieżność mniej gładka i trzeba dobrać dodatkowe hiperparametry (rozmiar batcha, lr), a szum utrudnia precyzyjne dojście do optimum. Full-batch: dokładny gradient i stabilna zbieżność, ale jeden drogi krok na epokę, duże zużycie pamięci i łatwo utknąć.

### Pytanie 8

Dlaczego do wyboru stopnia wielomianu nie możemy użyć błędu na zbiorze treningowym? Do czego służy zbiór walidacyjny, a do czego testowy?

**Odpowiedź:** Błąd treningowy maleje monotonicznie wraz ze wzrostem pojemności (stopnia `D`), więc jego minimum zawsze wypada dla modelu maksymalnie złożonego - nie wykrywa przeuczenia. Dlatego stopień wybieramy na **zbiorze walidacyjnym** (dane nieużywane w treningu, ale też nie testowe) - na nim widać minimum krzywej U. **Zbiór testowy** służy do jednorazowej, bezstronnej oceny już wybranego modelu i nie wolno go używać do strojenia hiperparametrów.

## Lab 02

### Pytanie 1

Zapisz regułę uczenia perceptronu. Czym różni się od spadku gradientu? Kiedy perceptron gwarantuje zbieżność?

**Odpowiedź:** Dla próbki $(\mathbf x_i, y_i)$, $y_i \in \{0, 1\}$, i predykcji $\hat y_i = \text{step}(\mathbf w^\top \mathbf x_i + b)$ reguła ma postać
$$\mathbf w \leftarrow \mathbf w + \eta\,(y_i - \hat y_i)\,\mathbf x_i, \qquad b \leftarrow b + \eta\,(y_i - \hat y_i).$$
Aktualizacja następuje tylko dla błędnie sklasyfikowanych próbek: przy fałszywie negatywnej dodajemy $\eta\mathbf x_i$ do wag, przy fałszywie pozytywnej odejmujemy. To **nie jest** spadek gradientu - funkcja progowa ma pochodną zero prawie wszędzie, więc nie ma tu gradientu straty; jest to reguła heurystyczna z krokiem o stałej długości. Spadek gradientu minimalizuje różniczkowalną stratę i przesuwa parametry w kierunku przeciwnym do gradientu, a perceptron tylko obraca/przesuwa hiperpłaszczyznę po każdym błędzie. Twierdzenie Novikoffa gwarantuje zbieżność w skończonej liczbie błędnych aktualizacji (ograniczonej przez $(R/\gamma)^2$, gdzie $R$ to maksymalna norma próbki, a $\gamma$ margines) **tylko wtedy, gdy dane są liniowo separowalne**. Dla danych nieseparowalnych (np. XOR) algorytm nigdy się nie zatrzyma - w naszym eksperymencie błędy nie zniknęły i oscylowały przez wszystkie 100 epok.

### Pytanie 2

Dlaczego pojedynczy perceptron nie rozwiąże problemu XOR, a MLP z jedną warstwą ukrytą tak?

**Odpowiedź:** Pojedynczy perceptron dzieli płaszczyznę **jedną prostą** (hiperpłaszczyzną) $\mathbf w^\top\mathbf x + b = 0$. W XOR punkty $(0,0)$ i $(1,1)$ należą do klasy 0, a $(0,1)$ i $(1,0)$ do klasy 1 - leżą naprzemiennie na przekątnych kwadratu. Każda prosta pozostawia po jednej stronie parę punktów różnych klas, więc nie da się osiągnąć 100% trafności; w eksperymencie perceptron utknął na 50% (4 punkty) i 53% (XOR zaszumiony).

MLP z warstwą ukrytą najpierw **przekształca wejście** $\mathbf h = \tanh(W_1\mathbf x + \mathbf b_1)$ w nową przestrzeń cech, a dopiero potem dzieli ją warstwa wyjściowa. Neurony ukryte mogą wykrywać dwie różne przekątne (dwie proste), a wyjście łączy ich odpowiedzi - granica decyzyjna staje się **kawałkami liniowa** i może oddzielić klasy XOR. W naszym zadaniu sieć 2-4-1 z tanh osiągnęła BCE = 0.00047 i predykcje `[0, 1, 1, 0]`.

### Pytanie 3

Dlaczego sieć bez nieliniowych funkcji aktywacji (dowolnie głęboka) jest równoważna modelowi liniowemu?

**Odpowiedź:** Złożenie przekształceń afinicznych jest przekształceniem afinicznym. Już dla dwóch warstw:
$$W_2(W_1\mathbf x + \mathbf b_1) + \mathbf b_2 = (W_2W_1)\mathbf x + (W_2\mathbf b_1 + \mathbf b_2),$$
czyli cała sieć redukuje się do jednej macierzy $W = W_2W_1$ i jednego biasu. Przez indukcję to samo zachodzi dla dowolnej liczby warstw: produkt macierzy jest macierzą, a suma przesunięć - przesunięciem. Taka sieć ma dokładnie taką samą pojemność jak regresja liniowa/pojedynczy perceptron (z liniową granicą decyzyjną) i nie rozwiąże XOR. Dopiero nieliniowa aktywacja między warstwami przerywa to uproszczenie i pozwala aproksymować funkcje nieliniowe.

### Pytanie 4

Wyjaśnij zjawisko znikającego gradientu. Dlaczego sigmoid jest na nie podatny bardziej niż ReLU?

**Odpowiedź:** Podczas backpropu gradient straty względem wczesnej warstwy jest iloczynem pochodnych kolejnych warstw (reguła łańcuchowa):
$$\frac{\partial L}{\partial W_1} \propto \prod_{l} f'(z_l)\, W_l.$$
Jeśli każdy czynnik ma moduł istotnie mniejszy od 1, iloczyn maleje **wykładniczo** z głębokością - pierwsze warstwy dostają gradient praktycznie równy zero i przestają się uczyć (analogicznie może wystąpić eksplozja gradientu, gdy czynniki są > 1).

Sigmoid ma pochodną $\sigma(z)(1-\sigma(z)) \le 0.25$, a dodatkowo nasyca się: dla $|z| \gtrsim 4$ pochodna jest bliska zeru. Iloczyn wielu czynników $\le 0.25$ daje np. $0.25^{20} \approx 10^{-12}$. ReLU ma pochodną **1** dla $z > 0$ i 0 dla $z < 0$, więc dla aktywnych neuronów gradient nie jest tłumiony; z inicjalizacją He (dobraną do wariancji ReLU) skala aktywacji i gradientów pozostaje w przybliżeniu stała. W naszym eksperymencie stosunek normy gradientu pierwszej warstwy do ostatniej wyniósł `7.5e-14` dla sigmoidu, `0.56` dla tanh i `1.4` dla ReLU. Tanh jest lepszy od sigmoidu (pochodna do 1, wycentrowanie wokół 0), ale nadal nasyca się dla dużych $|z|$.

### Pytanie 5

Ile parametrów ma MLP o architekturze 10 -> 32 -> 16 -> 3? Pokaż obliczenia.

**Odpowiedź:** Warstwa gęsta ma $(n_{in} + 1)\,n_{out}$ parametrów (macierz wag plus bias). Kolejno:
- 10 -> 32: $(10 + 1)\cdot 32 = 352$,
- 32 -> 16: $(32 + 1)\cdot 16 = 528$,
- 16 -> 3: $(16 + 1)\cdot 3 = 51$.

Razem: $352 + 528 + 51 = \mathbf{931}$ parametrów. Funkcje aktywacji nie mają parametrów.

### Pytanie 6

Dlaczego dla sigmoidu z BCE gradient po pobudzeniu wyjścia upraszcza się do $\hat y - y$? Jaka jest korzyść z używania `BCEWithLogitsLoss` (lub `from_logits=True`)?

**Odpowiedź:** Dla $\hat y = \sigma(z)$ i binarnej entropii krzyżowej
$$L = -\big[y\log\hat y + (1-y)\log(1-\hat y)\big]$$
pochodna po wyjściu wynosi
$$\frac{\partial L}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y} = \frac{\hat y - y}{\hat y(1-\hat y)}.$$
Pochodna sigmoidu to $\frac{\partial \hat y}{\partial z} = \hat y(1-\hat y)$, więc po pomnożeniu mianownik się skraca:
$$\frac{\partial L}{\partial z} = \frac{\hat y - y}{\hat y(1-\hat y)}\cdot \hat y(1-\hat y) = \hat y - y.$$
Dzięki temu gradient jest prosty i dobrze uwarunkowany nawet przy dużych błędach.

`BCEWithLogitsLoss` (PyTorch) i `from_logits=True` (Keras) łączą sigmoidę i BCE w jedną operację liczoną numerycznie stabilnie (w praktyce przez stabilną postać softplus, bez jawnego liczenia sigmoidy i logarytmu). Unika się w ten sposób liczenia `log(0)` i przepełnienia/utraty precyzji przy najpierw `sigmoid`, a potem `log`; wynik jest dokładniejszy i szybszy. Dlatego w PyTorch model zwraca **logity** (bez sigmoidy), a dopiero `BCEWithLogitsLoss` robi resztę.

### Pytanie 7

Czym różni się Sequential API od Functional API w Keras? Podaj przykład architektury, której nie zbudujesz w Sequential.

**Odpowiedź:** `keras.Sequential` to **liniowy stos warstw**: każda warstwa ma dokładnie jedno wejście i jedno wyjście, a wyjście poprzedniej jest wejściem następnej. Nie da się w nim rozgałęziać, łączyć tensorów ani mieć wielu wejść/wyjść. **Functional API** traktuje warstwy jak funkcje wywoływane na tensorach symbolicznych i pozwala zbudować dowolny graf (DAG): wiele wejść i wyjść, konkatenacje, rozgałęzienia, współdzielone warstwy, połączenia pomijające. Model definiuje się przez `keras.Model(inputs=..., outputs=...)`.

Przykład, którego nie zbudujesz w Sequential, jest w notebooku: wejście jest doklejane przez `Concatenate` do wyjścia drugiej warstwy ukrytej (`[x, h2]`), a dopiero potem trafia do warstwy wyjściowej - to połączenie pomijające (skip connection). Inne przykłady: residual block $y = \text{ReLU}(F(x) + x)$, model z dwoma osobnymi wejściami (np. obraz + metadane) albo z dwoma wyjściami (klasa + regresja pomocnicza).

### Pytanie 8

Wymień kroki jednej iteracji pętli treningowej PyTorch. Co robią `model.train()`, `model.eval()` i `torch.no_grad()`?

**Odpowiedź:** Jedna iteracja (dla batcha):
1. `opt.zero_grad()` - wyzerowanie gradientów z poprzedniego kroku (w PyTorch `.grad` się akumuluje),
2. `logity = model(xb)` - przejście w przód,
3. `loss = loss_fn(logity, yb)` - policzenie straty,
4. `loss.backward()` - backpropagation i wypełnienie `.grad` wszystkich parametrów,
5. `opt.step()` - aktualizacja parametrów przez optymalizator.

`model.train()` ustawia tryb treningowy: aktywne są dropout i augmentacje zależne od trybu, a BatchNorm używa statystyk bieżącego batcha. `model.eval()` przełącza tryb ewaluacji: dropout jest wyłączony, a BatchNorm korzysta z uśrednionych statystyk zgromadzonych podczas treningu. `torch.no_grad()` wyłącza budowanie grafu obliczeniowego (nie zapisuje pośrednich aktywacji), co oszczędza pamięć i przyspiesza ewaluację/predykcję; stosuje się go razem z `model.eval()` przy walidacji i inferencji.

## Lab 03

### Pytanie 1

Dlaczego w ostatniej warstwie klasyfikatora na 10 klas używamy softmax, a w regresji warstwy liniowej? Co by się stało, gdyby w regresji cen domów użyć sigmoidy?

**Odpowiedź:** Softmax jest potrzebny w klasyfikacji, bo chcemy **rozkładu prawdopodobieństw** nad $K$ klas: zamienia logity $z_k$ na $p_k = e^{z_k}/\sum_j e^{z_j}$, więc $p_k \in (0,1)$ i $\sum_k p_k = 1$. Dokładnie tego wymaga entropia krzyżowa $L = -\log p_y$ i na tym opiera się decyzja `argmax`. W regresji nie ma klas - cel jest dowolną liczbą rzeczywistą (u nas mediana ceny w setkach tysięcy dolarów, zakres 0.15-5.0), więc wyjście musi być liniowe. Sigmoida ścisnęłaby wyjście do $(0,1)$: model mógłby przewidzieć najwyżej 1.0, czyli 100 tys. $, a nigdy 5.0 (500 tys. $), więc cele powyżej 1 byłyby nieosiągalne. Dodatkowo sigmoida nasyca się - dla dużych błędów jej pochodna jest bliska zeru, gradient znika i trening z MSE staje się bardzo wolny.

### Pytanie 2

Czym różni się `sparse_categorical_crossentropy` od `categorical_crossentropy`? Kiedy używamy której?

**Odpowiedź:** Obie funkcje liczą tę samą stratę $L = -\log p_y$, różnią się formatem etykiet. `categorical_crossentropy` oczekuje **one-hot** o kształcie `(n, K)`, np. klasa 3 -> `[0,0,0,1,0,0,0,0,0,0]`; `sparse_categorical_crossentropy` przyjmuje **liczby całkowite** `(n,)` i sama wybiera odpowiednią kolumnę. Wersja `sparse` oszczędza pamięć (brak macierzy `n x K`, przy 50 000 próbek i 10 klasach to 500 000 dodatkowych floatów) i jest wygodniejsza, gdy etykiety są pojedynczymi indeksami - tak jest w Fashion-MNIST. Wersji `categorical` używamy, gdy etykiety są rozkładami prawdopodobieństwa (label smoothing, mixup, dystylacja wiedzy) albo gdy mamy już one-hot. Typowy błąd to podanie etykiet całkowitych do `categorical_crossentropy` - Keras zgłasza wtedy niezgodność kształtów.

### Pytanie 3

Dlaczego `StandardScaler` dopasowujemy tylko do zbioru treningowego? Jak nazywa się błąd polegający na dopasowaniu go do całych danych i jakie ma skutki?

**Odpowiedź:** `StandardScaler` uczy się parametrów $\mu$ i $\sigma$ z danych, więc dopasowanie go do całego zbioru oznacza, że statystyki zbioru testowego (a w walidacji krzyżowej - statystyki foldów walidacyjnych) wpływają na przygotowanie danych treningowych. Informacja o rozkładzie danych testowych "przecieka" do treningu. Błąd ten nazywamy **wyciekiem danych** (*data leakage*). Skutki: wynik na zbiorze testowym jest sztucznie optymistyczny (model "widział" skalę i rozkład testu), ocena generalizacji przestaje być uczciwa, a w produkcji nowe dane są przekształcane parametrami, których przy treningu nie było. Poprawnie: `fit` tylko na `X_train`, a `transform` (te same $\mu$, $\sigma$) na train, walidacji i teście. W naszym przebiegu średnie cech po standaryzacji na train są równe 0, a na teście już nie (od -0.021 do 0.030) - i tak ma być.

### Pytanie 4

Ile parametrów ma warstwa `Dense(64)` po wejściu o wymiarze 8? A `BatchNormalization` po niej (ile z nich jest uczonych gradientowo)?

**Odpowiedź:** Warstwa `Dense(64)` po wejściu o wymiarze 8 ma $8\cdot64 + 64 = \mathbf{576}$ parametrów (512 wag + 64 biasy). `BatchNormalization` po niej ma 4 parametry na cechę: $\gamma$ i $\beta$ (przeskalowanie i przesunięcie) oraz `moving_mean` i `moving_var` (statystyki do wnioskowania). Razem $4\cdot64 = 256$ parametrów, z czego **uczone gradientowo jest 128** ($\gamma$, $\beta$), a pozostałe 128 to bufor aktualizowany średnią kroczącą podczas `fit`, a nie gradientem.

### Pytanie 5

Co robi `restore_best_weights=True` w `EarlyStopping` i dlaczego jest ważne?

**Odpowiedź:** `restore_best_weights=True` sprawia, że po zatrzymaniu treningu Keras przywraca wagi z epoki, w której monitorowana metryka (np. `val_loss`) była najlepsza - a nie wagi z ostatniej epoki. Ma to znaczenie, bo `EarlyStopping` zatrzymuje trening dopiero po `patience` epokach bez poprawy; w tym czasie model zdążył się już przeuczyć i jego ostatnie wagi są gorsze od najlepszych. Bez tej opcji "wygrywa" model z końca treningu, czyli zwykle gorszy na walidacji. Wagi najlepszej epoki są trzymane w pamięci (dla zapisu na dysk służy `ModelCheckpoint(save_best_only=True)`).

### Pytanie 6

Wyjaśnij, czym Adam różni się od zwykłego SGD. Dlaczego SGD bez momentum uczy się wolniej?

**Odpowiedź:** SGD robi krok w kierunku przeciwnym do gradientu, ze **wspólnym** learning rate: $w \leftarrow w - \eta g_t$. Adam łączy dwa mechanizmy: momentum (pierwszy moment $m_t$, średnia krocząca gradientów) i RMSprop (drugi moment $v_t$, średnia krocząca kwadratów gradientów) z korekcją obciążenia:
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad w \leftarrow w - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
Dzięki temu każdy parametr ma **własny, adaptacyjny krok**: kierunki o dużym gradiencie są tłumione, a rzadko aktualizowane przyspieszane. SGD bez momentum uczy się wolniej, bo: (1) ma jeden krok dla wszystkich parametrów, więc w źle uwarunkowanych "wąskich dolinach" oscyluje w poprzek i wolno posuwa się wzdłuż, (2) nie wygładza szumu mini-batchy, (3) tempo ogranicza największa krzywizna - mniejszy lr daje stabilność, ale wolniejszy postęp. W naszym porównaniu (10 epok, ten sam MLP) SGD osiągnął `val_accuracy` 0.8354, SGD+momentum 0.8722, RMSprop 0.8859, a Adam 0.8825.

### Pytanie 7

Na wykresie strata walidacyjna rośnie od 8. epoki, a treningowa dalej spada. Jak to nazywamy i jakie trzy działania podejmiesz?

**Odpowiedź:** To **przeuczenie** (*overfitting*): model dopasowuje szum zbioru treningowego, więc strata treningowa maleje, a walidacyjna rośnie - rośnie luka między treningiem a walidacją. Trzy działania: (1) **regularyzacja** - Dropout, L2/weight decay, ewentualnie ograniczenie pojemności modelu; (2) **early stopping** na `val_loss` z `restore_best_weights=True`, żeby zatrzymać się w minimum walidacyjnym (w eksperymencie z sekcji 5 minimum wypadło już w 6. epoce); (3) **więcej danych / augmentacja** (albo walidacja krzyżowa przy małym zbiorze). Dodatkowo można zmniejszyć learning rate lub zastosować harmonogram, ale to leczy objaw, nie przyczynę.

### Pytanie 8

Jak działa Dropout w trakcie treningu, a jak w predykcji? Dlaczego przeskalowuje się aktywacje przez $1/(1-p)$?

**Odpowiedź:** W treningu Dropout($p$) losowo zeruje każdą aktywację z prawdopodobieństwem $p$ w każdym kroku, a pozostałe skaluje przez $1/(1-p)$ (tzw. *inverted dropout*). Sieć nie może polegać na pojedynczych neuronach - uczy się reprezentacji redundantnych, co działa jak uśrednianie wielu "podsieci" i utrudnia zapamiętanie szumu. W predykcji Dropout jest **wyłączony**: wszystkie neurony pracują, nic nie jest zerowane ani skalowane. Skalowanie w treningu przez $1/(1-p)$ wyrównuje **wartość oczekiwaną** aktywacji: jeśli w treningu neuron jest aktywny z prawdopodobieństwem $1-p$, to mnożąc przez $1/(1-p)$ dostajemy oczekiwaną aktywację równą tej w predykcji. Bez tego przeskalowania aktywacje (a więc i logity) w predykcji byłyby średnio $1/(1-p)$ razy większe niż oczekiwane aktywacje w treningu i model dawałby przesunięte, źle skalibrowane wyjścia. W zad. 5.1 Dropout 0.5 zmniejszył lukę `train-val` z 0.181 do 0.099.

## Lab 04

### Pytanie 1

Obraz 64x64x3 przechodzi przez `Conv2D(20, kernel_size=5, strides=1, padding="valid")`. Jaki jest kształt wyjścia i ile parametrów ma warstwa?

**Odpowiedź:** `Conv2D(20, kernel_size=5, strides=1, padding="valid")` na wejściu $64\times64\times3$: wymiar wyjścia liczymy ze wzoru $n_{out}=\lfloor (n + 2p - k)/s \rfloor + 1 = \lfloor (64 - 5)/1 \rfloor + 1 = 60$, więc wyjście ma kształt **60×60×20** (`valid` nie dopisuje zer, więc wysokość i szerokość maleją o $k-1 = 4$). Parametry: każdy z 20 filtrów ma $5\cdot5\cdot3 = 75$ wag plus 1 bias, razem $20\cdot(75+1) = \mathbf{1520}$ parametrów (1500 wag + 20 biasów). Liczba parametrów **nie zależy** od rozmiaru obrazu - ten sam filtr jest przesuwany po wszystkich 3600 pozycjach.

### Pytanie 2

Dlaczego warstwa splotowa ma znacznie mniej parametrów niż warstwa gęsta o porównywalnej liczbie wyjść? Wymień dwie własności splotu, które o tym decydują.

**Odpowiedź:** Decydują dwie własności splotu: (1) **współdzielenie wag** (*weight sharing*) - jeden filtr o rozmiarze $k\times k\times C_{in}$ ma $k^2C_{in}+1$ parametrów i jest używany w **każdym** położeniu, więc liczba wag nie rośnie wraz z $H\cdot W$; (2) **lokalność połączeń** (rzadkie połączenia) - każdy neuron wyjściowy widzi tylko $k\times k\times C_{in}$ sąsiednich wejść, a nie wszystkie piksele jak w `Dense`.

Liczby z notebooka: `Conv2D(32, 3)` na wejściu $32\times32\times3$ ma $3\cdot3\cdot3\cdot32 + 32 = \mathbf{896}$ parametrów, a `Dense(32)` na spłaszczonym obrazie (3072 wejścia) ma $3072\cdot32 + 32 = \mathbf{98\ 336}$ - 110 razy więcej przy tych samych 32 wyjściach. Porównanie przy **równej liczbie wyjść** wypada jeszcze mocniej: splot daje $32\cdot32\cdot32 = 32\ 768$ wyjść za 896 parametrów, a warstwa gęsta z 32 768 wyjściami potrzebowałaby $3072\cdot32\ 768 + 32\ 768 \approx 100{,}7$ mln wag. W zad. 4.1 widać to samo w głowie klasyfikatora: `Flatten` + `Dense(10)` po mapie $8\times8\times128$ to 81 930 parametrów, a `GlobalAveragePooling2D` + `Dense(10)` - 1 290.

### Pytanie 3

Co robi `MaxPooling2D(2)` i ile ma parametrów? Czym różni się od `GlobalAveragePooling2D`?

**Odpowiedź:** `MaxPooling2D(2)` bierze **maksimum** z każdego okna $2\times2$ i przesuwa się o 2 piksele, więc wysokość i szerokość maleją dwukrotnie ($28\times28\to14\times14$), a liczba kanałów się nie zmienia. Ma **0 parametrów** - niczego się nie uczy, to stała operacja. Daje częściową **niezmienniczość na przesunięcia** (małe przesunięcie obrazu prawie nie zmienia maksimów), zachowuje najsilniejsze aktywacje i zmniejsza liczbę obliczeń w kolejnych warstwach.

`GlobalAveragePooling2D` uśrednia **całą** mapę cech $H\times W$ do jednej liczby, zamieniając tensor $H\times W\times C$ na wektor długości $C$ - też ma 0 parametrów, ale **usuwa informację przestrzenną** (zostaje "ile" danej cechy, a nie "gdzie"). Dlatego stosuje się go tuż przed klasyfikatorem: działa przy dowolnej rozdzielczości wejścia i zastępuje `Flatten` + dużą warstwę gęstą (81 930 → 1 290 parametrów w zad. 4.1). MaxPooling zachowuje siatkę przestrzenną i może być powtarzany w środku sieci; GAP agreguje globalnie i zamyka część splotową.

### Pytanie 4

Dlaczego warstwy augmentacji w modelu nie zmieniają obrazów podczas `evaluate` i `predict`? Podaj przykład augmentacji, która byłaby **błędem** dla CIFAR-10, i przykład błędnej augmentacji dla MNIST.

**Odpowiedź:** Warstwy augmentacji (podobnie jak `Dropout` i `BatchNormalization`) są **zależne od trybu pracy**: sprawdzają flagę `training`. W `fit` Keras wywołuje model z `training=True` i wtedy losowo przekształcają obrazy, ale w `evaluate`, `predict` oraz w walidacji w trakcie `fit` - z `training=False` - są **identycznością** (zwracają wejście bez zmian). Dzięki temu metryki walidacyjne i testowe są liczone na czystych danych, a augmentacja rozszerza wyłącznie zbiór treningowy.

Błędna augmentacja to taka, która **zmienia znaczenie klasy**. Dla CIFAR-10: odbicie pionowe (`RandomFlip("vertical")`) - pies do góry nogami, samolot "do góry podwoziem" czy przewrócony samochód nie występują w rzeczywistości, więc etykieta przestaje być prawdziwa (podobnie duże obroty, np. o 90°). Dla MNIST: odbicie poziome - cyfry 2, 3, 5 i 7 po odbiciu nie są już poprawnymi cyframi, a obrót o 180° zamienia 6 z 9. Bezpieczne są natomiast małe przesunięcia, przycięcia i obroty o kilka stopni.

### Pytanie 5

Zapisz wzór bloku residualnego. Dlaczego połączenia residualne ułatwiają trenowanie głębokich sieci? Kiedy skrót wymaga splotu 1x1?

**Odpowiedź:** Blok residualny: $y = \mathrm{ReLU}\big(F(x) + x\big)$, gdzie $F$ to kilka warstw (u nas `Conv2D → BN → ReLU → Conv2D → BN`), a $x$ jest dodawane do wyjścia $F(x)$ przez połączenie pomijające (*skip connection*).

Połączenie residualne ułatwia trening, bo w Jakobianie przejścia pojawia się **jedynka**: $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$. Gradient ma więc "autostradę" przez $I$ nawet wtedy, gdy $\partial F/\partial x$ jest bliskie zeru - znikający gradient w głębokich sieciach przestaje być krytyczny. Drugi powód: jeśli optymalnym przekształceniem jest "nic nie zmieniaj", wystarczy wyzerować wagi $F$, zamiast uczyć się odwzorowania tożsamościowego od zera. W zad. 4.1 widać tego efekty: mini-ResNet ma **309 994** parametry (1.8x mniej niż 551 914 w `model_cifar`), osiąga zbliżoną dokładność testową (0.7952 vs 0.8003), ale **wyższą dokładność treningową** (0.8269 vs 0.7947), czyli uczy się szybciej przy mniejszej pojemności.

Skrót wymaga splotu $1\times1$, gdy $F(x)$ i $x$ mają **inny kształt**: inną liczbę kanałów (np. $32\to64$) albo inną rozdzielczość (gdy splot w $F$ ma `strides=2`). Wtedy $x$ przepuszczamy przez `Conv2D(1, 1, strides=..., use_bias=False)` + `BatchNormalization`, żeby dopasować liczbę kanałów i rozmiar - tak jak w funkcji `blok_rezydualny` w notebooku.

### Pytanie 6

Opisz dwa etapy transfer learningu. Dlaczego w fine-tuningu stosujemy bardzo mały learning rate i dlaczego najpierw trenujemy samą głowę?

**Odpowiedź:** **Etap 1 - ekstrakcja cech:** baza (MobileNetV2 bez głowy klasyfikacyjnej) jest zamrożona (`baza.trainable = False`, 0 trenowalnych parametrów w bazie), a uczymy tylko nową głowę `GlobalAveragePooling2D → Dropout(0.2) → Dense(10)`. W naszym przebiegu to **12 810** parametrów (0.6% całego modelu), 6 epok, 171 s i test accuracy **0.8315**. **Etap 2 - fine-tuning:** odmrażamy górną część bazy (30 ostatnich warstw, m.in. `Conv_1`, `Conv_1_bn`, `out_relu`; 1 539 210 trenowalnych parametrów), rekompilujemy model i **kontynuujemy** trening od epoki, na której skończył się etap 1 (stąd `initial_epoch`), ale z learning rate **1e-5** - dało to 0.8405.

Bardzo mały learning rate jest konieczny, bo wagi ImageNet są już blisko dobrego rozwiązania: duży krok "przestawiłby" wyuczone filtry i zniszczył cechy, które chcemy wykorzystać (*catastrophic forgetting*). Widać to na krzywej: w pierwszej epoce etapu 2 (epoka 7) dokładność treningowa spada z 0.876 do 0.782, ale dzięki małemu lr model odrabia stratę w 2-3 epoki i wychodzi powyżej etapu 1. Najpierw trenujemy samą głowę, bo jest **losowa** - jej gradienty są duże i zaszumione; gdyby baza była wtedy trenowalna, te gradienty spłynęłyby do pretrenowanych wag i rozstroiły je, zanim głowa nauczy się czegokolwiek sensownego. Po etapie 1 głowa daje sensowne predykcje, poprawki propagowane do bazy są małe i fine-tuning tylko je dopracowuje.

### Pytanie 7

Co się stanie, jeśli obrazy podzielimy przez 255, a potem podamy do `mobilenet_v2.preprocess_input`?

**Odpowiedź:** `mobilenet_v2.preprocess_input` oczekuje pikseli w zakresie **0-255** i liczy $x/127.5 - 1$, czyli mapuje $[0,255]\to[-1,1]$. Jeśli najpierw podzielimy obrazy przez 255 (zakres $[0,1]$), to po `preprocess_input` dostaniemy $[0/127.5-1,\ 1/127.5-1] = [-1,\ -0.9922]$: cały obraz to **prawie stała wartość ok. $-1$**, a kontrast (różnica między najciemniejszym i najjaśniejszym pikselem) kurczy się z 2.0 do 0.008. Sieć dostaje praktycznie jednolity tensor, aktywacje są niemal identyczne dla wszystkich klas, więc dokładność spada do poziomu losowego - i **żaden błąd nie jest zgłaszany**, bo kształty i typy danych się zgadzają. To cichy błąd preprocessingu; poprawnie podajemy surowe obrazy 0-255 (u nas `xc_te_raw`, `uint8`) i wywołujemy `preprocess_input` dokładnie raz.

### Pytanie 8

Po co wywołujemy `baza(x, training=False)`, skoro i tak ustawiamy `baza.trainable`?

**Odpowiedź:** `baza.trainable` i `baza(x, training=False)` sterują **dwiema różnymi rzeczami**. `trainable` mówi optymalizatorowi, czy wagi warstwy mają być aktualizowane gradientem (przy `False` są pomijane przy aktualizacji, ale nadal biorą udział w przejściu w przód). `training=False` ustawia **tryb działania** warstw, które zachowują się inaczej w treningu i w inferencji: `BatchNormalization` używa wtedy zapamiętanych statystyk (`moving_mean`, `moving_var`) z ImageNet zamiast statystyk bieżącego batcha, a `Dropout` jest wyłączony.

W fine-tuningu chcemy **obu rzeczy naraz**: wagi bazy mają się uczyć, ale normalizacja ma zostać w trybie inferencji. Gdyby BN liczył statystyki z batcha, to przy batchu 64 obrazów z innego rozkładu (CIFAR-10 w 96x96, a nie ImageNet 224x224) średnie i wariancje byłyby głośne i rozstroiłyby statystyki ruchome - na starcie etapu 2 dokładność gwałtownie by spadła, a model musiałby je odbudowywać od zera. Samo `baza.trainable = False` **nie wystarcza**, bo nie zatrzymuje aktualizacji statystyk BN; dlatego bazę wywołujemy zawsze jako `baza(x, training=False)` - także po odmrożeniu warstw.

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
