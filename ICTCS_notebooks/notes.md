1. **GENERATE DATASETS**

- *Hyperparametes*: SIZE = 10000
                    MAX_DEPTH = 5
                    NUM_LETTERS = 7
                    SEED = 42

- *Dataset Synthetic + Taut* : Tot. 13,000 - Percentage of tautologies in the dataset: 26.29%
- *Dataset Synth + Taut + Dimacs*: Tot. 13,000 -Percentage of tautologies in the composed dataset: 41.68%

-----------------------------
2. **Preparing Data**

- *Hyperparameters*: TEST_SIZE = 0.2
                     BATCH_SIZE = 16
                     SEED = 42
  
- *Dataloader & Train-Test split using (Synt + taut) Dataset*:

(train_dataloader, 
 test_dataloader, 
 X_train, X_test, 
 y_train, y_test, 
 idx_test)  = prepare_formula_dataset(dataset = dataset,
                                      test_size=TEST_SIZE,
                                      batch_size=BATCH_SIZE,
                                      seed=SEED)
                                      
- *Training set*: 10400 samples | *Test set*: 2600 samples

- *Tokenization data*: Number of unique letters: 8
                       Number of unique connectives: 4
                       Number of spacial tokens: 2

- *Train and Test distribution*: Train set (10400 samples): False = 73.67 % and True = 26.33 %
                                 Test set  (2600samples):   False = 73.62 % and True = 26.38 %

-----------------------------
3. **Agnostic-Code**

-----------------------------
4. **Build, Train and Test Models on Synthetic Dataset**

- *Hyperparameters*: VOCAB_SIZE (including padding token) = 108
                     EMBEDDING_DIM = 32
                     LR = 0.0005
                     EPOCHS = 5

- **Model_1** Vanilla RNN model:
RNN_V1(
  (embedding): Embedding(108, 32)
  (rnn): RNN(32, 64, batch_first=True)
  (linear): Linear(in_features=64, out_features=1, bias=True)
)

*Trainable params*: 9,793

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=LR)

*Final score*: Epoch: 5 | train_loss: 0.5770 | train_acc: 0.7369 | test_loss: 0.5764 | test_acc: 0.7370
Given the data distribution and Model performances, model_1 predicts False every time — and that would still be right ~74% of the time.


- **Model_2** Bidirectional GRU: 
GRU(
  (embedding): Embedding(108, 32, padding_idx=0)
  (gru1): GRU(32, 128, batch_first=True, bidirectional=True)
  (gru2): GRU(256, 64, batch_first=True, bidirectional=True)
  (fc1): Linear(in_features=128, out_features=32, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=32, out_features=1, bias=True)

*Trainable params*: 255,681

*Focal Loss*: L(y, ŷ) = - α_pos * y * (1 - ŷ)^γ_pos * log(ŷ) - α_neg * (1 - y)^γ_neg * log(1 - ŷ)
loss_fn = AsymmetricFocalLoss(alpha_pos=0.3,  # minority (tautology)
                              alpha_neg=0.7,  # majority
                              gamma_pos=3.0,
                              gamma_neg=1.5
)
optimizer = torch.optim.Adam(params=model_2.parameters(), lr=LR)

*Final score*: Epoch: 5 | train_loss: 0.0177 | train_acc: 0.9593 | test_loss: 0.0160 | test_acc: 0.9624


-----------------------------
5. **LSTM Tree**
The logical formulas are represented as trees. This will serve as the recursive structure on which the TreeLSTM will operate.

**Syntactic Tree Representation of Formulas**
Each node of the tree will correspond to a `FormulaTreeNode` object, which represents a node of the logical formula.

**Assigning embedding indices to the nodes**

The `CustomTokenizer` already maps each `Formula` object to an integer. We the helper function `assign_embedding_indices` that:
-  Recursively visits the `FormulaTreeNode` tree
- Assigns to each node the corresponding `embedding_index` using `tokenizer.formula_to_token`.

**Dataset, Custom collate_fn for tree structures, and Representative Subset of Formulas for experiments**

We use: 
- The datset `TreeFormulaDataset` that returns `(FormulaTreeNode, label)`
- The `tree_collate_fn` that creates a batch (actually a list) of trees
- The `DataLoader` that uses the `collate_fn`
- The adaptation `tree_train_step()` and `tree_test_step()` for train and test the TreeLSTM model.
- The function `prepare_balanced_tree_dataloaders()` that:
  - Takes as input: all formulas and labels 
  - Selects a balanced subset with the same original proportion (~74% / 26%)
  - Returns `DataLoaders` ready for training.

Train set (1000 samples): False = 74.00 % and True = 26.00 %
Test set (200 samples): False = 74.00 % and True = 26.00 %

-  **Sweep config — Step 1: `hidden_size` and `fully_conected_size`**
  - Best conf. hidden states: 128
  - Best conf. fully conn. layers: 32

- **Sweep config — Step 2: `alpha_pos, alpha_neg, gamma_pos, gamma_neg`**
  - Best conf. alpha pos: 0.25
  - Best conf. alpha neg: 0.7
  - Best conf. gamma pos: 3.0
  - Best conf. gamma neg: 2.5

- **Sweep config — Step 3: `learning_rate`**
  - Best conf. learning rate: 0.0009
  - Best conf. alpha pos: 0.25
  - Best conf. alpha neg: 0.7
  - Best conf. gamma pos: 3
  - Best conf. gamma neg: 2.5

**Tree LSTM Model Training and Testing on the full dataset with best hyperparameters**

# --- Hyperparameters ---
TEST_SIZE = 0.2
BATCH_SIZE = 16
SEED = 42

# --- Best WandB Hyperparameters ---
VOCAB_SIZE = compute_vocab_size(tokenizer)
print(f"Vocabulary size (including padding token): {VOCAB_SIZE}")

EMBEDDING_DIM = 32
HIDDEN_SIZE = 128
FC_SIZE = 32
LR = 0.0009

ALPHA_POS = 0.25
ALPHA_NEG = 0.7
GAMMA_POS = 3
GAMMA_NEG = 2.5

**First Tree LSTM Model**
tree_model = TreeLSTMClassifierV1(vocab_size=VOCAB_SIZE,
                                  embedding_dim=EMBEDDING_DIM,
                                  hidden_size=HIDDEN_SIZE,
                                  fc_size=FC_SIZE).to(device)

# --- Loss and Optimizer ---
loss_fn = AsymmetricFocalLoss(alpha_pos=ALPHA_POS, 
                              alpha_neg=ALPHA_NEG,
                              gamma_pos=GAMMA_POS,
                              gamma_neg=GAMMA_NEG
                             )

optimizer = torch.optim.Adam(tree_model.parameters(), lr=LR)

TreeLSTMClassifierV1(
  (encoder): TreeLSTMEncoder(
    (embedding): Embedding(108, 32, padding_idx=0)
    (cell): BinaryTreeLSTMCell(
      (W_iou): Linear(in_features=32, out_features=384, bias=True)
      (U_iou): Linear(in_features=256, out_features=384, bias=True)
      (W_f): Linear(in_features=32, out_features=256, bias=True)
      (U_f): Linear(in_features=256, out_features=256, bias=True)
    )
  )
  (fc1): Linear(in_features=128, out_features=32, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=32, out_features=1, bias=True)
)

*Trainable parameters:* 193,217

train_loss: 0.0000 |train_acc=1.0000 |test_loss=0.0047 |test_acc=0.9892


**Second Tree LSTM Model (with Dropout)**
tree_model_V2 = TreeLSTMClassifierV2(vocab_size=VOCAB_SIZE,
                                     embedding_dim=EMBEDDING_DIM,
                                     hidden_size=HIDDEN_SIZE,
                                     fc_size=FC_SIZE).to(device)


# --- Loss and Optimizer ---
loss_fn = AsymmetricFocalLoss(alpha_pos=ALPHA_POS, 
                              alpha_neg=ALPHA_NEG,
                              gamma_pos=GAMMA_POS,
                              gamma_neg=GAMMA_NEG
                             )

optimizer = torch.optim.Adam(tree_model_V2.parameters(), lr=LR)

TreeLSTMClassifierV2(
  (encoder): TreeLSTMEncoder(
    (embedding): Embedding(108, 32, padding_idx=0)
    (cell): BinaryTreeLSTMCell(
      (W_iou): Linear(in_features=32, out_features=384, bias=True)
      (U_iou): Linear(in_features=256, out_features=384, bias=True)
      (W_f): Linear(in_features=32, out_features=256, bias=True)
      (U_f): Linear(in_features=256, out_features=256, bias=True)
    )
  )
  (fc1): Linear(in_features=128, out_features=32, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.3, inplace=False)
  (fc2): Linear(in_features=32, out_features=1, bias=True)
)

*Trainable parameters*: 193,217

train_loss: 0.0011 |train_acc=0.9930 |test_loss=0.0016 |test_acc=0.9919

**Confusion Matrix Second Model**
[[1902   18]  -> 18 False Positive
 [  6  674]]  -> 6 false Negative
 
 
=======================================================================

**4.1 Rappresentazione delle formule e generazione del dataset.**
Nel nostro framework, le formule della logica proposizionale sono codificate da oggetti con struttura ad albero, definiti dalla classe astratta Formula e da una serie di sottoclassi che rappresentano variabili porposizionali, costanti e connettivi logici. Questa rappresentazione consente una manipolazione flessibile delle formule, mantenendone la struttura sintattica e semantica, inclusa la gestione della priorità degli operatori e della loro associatività.
Le formule atomiche sono rappresentate da istanze della classe Letter, ciascuna associata a un identificatore numerico. Le formule composte vengono costruite ricorsivamente utilizzando le sottoclassi UnaryConnectiveFormula e BinaryConnectiveFormula, che modellano rispettivamente il connettivo unario (¬) e i connettivi binari (∧, ∨, →). Queste classi incorporano le proprietà sintattiche dei connettivi, come l’associatività e, dove applicabile, la commutatività. Inoltre, il sovraccarico (overloading) di alcuni operatori Python (&, |, ~) consente di definire formule in modo leggibile e conciso, mantenendo una notazione simile a quella logica tradizionale.

Per addestrare e valutare i modell Recurrent Neural Network, abbiamo costruito un dataset sintetico di formule proposizionali normalizzate, generate in modo controllato attraverso un processo ricorsivo. Il generatore produce formule casuali controllando due parametri principali: 
- La profondità massima (max_depth = 5) definisce il numero massimo di livelli di annidamento nella struttura ad albero della formula. Ogni volta che un connettivo viene applicato a una o più sottoformule, si aggiunge un livello alla profondità complessiva. Limitare la profondità consente di generare formule dalla complessità controllata, evitando annidamenti eccessivi e mantenendo gestibile lo spazio delle formule possibili.
- Il numero massimo di lettere proposizionali (num_letters = 7) stabilisce il numero di simboli distinti (es. A0, A1, ..., A6) che possono essere utilizzati nella costruzione delle formule. 
- 
La generazione segue una logica ricorsiva: a ciascun livello, la formula può essere la costante booeana falso (⊥), una lettera proposizionale (Ai) oppure una formula composta. Le formule composte sono costruite applicando il connettivo logici unario (¬) o i connettivi ligici binari (∧, ∨, →) a sottostrutture generate ricorsivamente. Le lettere proposizionali vengono assegnate tramite un generatore ciclico che restituisce oggetti Letter(i), con i ∈ [0, ..., num_letters - 1].

Una volta generata, ogni formula viene normalizzata utilizzando la classe Normalizer, che rinumera le lettere in base all’ordine di apparizione, assegnando indici crescenti a partire da 0. In questo modo, formule sintatticamente diverse come A2 ∧ A7 e A7 ∧ A2 vengono entrambe trasformate nella formula normalizzata A0 ∧ A1. Questa procedura riduce la ridondanza sintattica superficiale nel dataset, permettendo al modello di concentrarsi sulla struttura logica piuttosto che sui nomi arbitrari delle lettere.

L’etichetta associata a ciascuna formula (tautologia o non-tautologia) è determinata tramite l'enumerazione esaustiva di tutte le possibili assegnazioni di valori di verità. Una formula è etichettata come tautologia se risulta vera per ogni interpretazione possibile. Il dataset finale è costituito da 10.000 formule distinte, ciascuna accompagnata dalla propria etichetta booleana.


Tuttavia, la distribuzione iniziale del dataset generato risultava fortemente sbilanciata: solo il 4,18% delle formule era una tautologia. Questo squilibrio è una conseguenza diretta del processo di generazione casuale: poiché le tautologie costituiscono una porzione estremamente ridotta dello spazio delle formule proposizionali, è altamente improbabile ottenerle tramite generazione casuale. Un dataset così sbilanciato rischia di compromettere l’addestramento del modello, inducendolo a trascurare la classe minoritaria e a generalizzare male nei confronti delle formule tautologiche. Un’analisi del dataset ha inoltre evidenziato l’assenza di istanze di tautologie canoniche, come ad esempio il principio del terzo escluso (A ∨ ¬A) o la doppia negazione (¬(A ∧ ¬A)). Allo scopo di rendere il dataset più rappresentativo, abbiamo introdotto una fase di data augmentation basata sull’instanziazione controllata di tautologie note. A questo scopo, abbiamo definito una serie di tautologie template, contenenti metavariabili (come A, B, C), rappresentate dalla classe Metavariable. Queste formule includono esempi fondamentali di logica classica, come:
- Principio del terzo escluso: A ∨ ¬A
- Doppia negazione (forma contraddittoria): ¬(A ∧ ¬A)
- Leggi di De Morgan (congiunzione): (¬(A ∧ B) → ¬A ∨ ¬B) ∧ (¬A ∨ ¬B → ¬(A ∧ B))
- Leggi di De Morgan (disgiunzione): (¬(A ∨ B) → ¬A ∧ ¬B) ∧ (¬A ∧ ¬B → ¬(A ∨ B))
- Distribuzione della congiunzione sulla disgiunzione:
(A ∧ (B ∨ C) → (A ∧ B) ∨ (A ∧ C)) ∧ ((A ∧ B) ∨ (A ∧ C) → A ∧ (B ∨ C))
- Distribuzione della disgiunzione sulla congiunzione:
(A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)) ∧ ((A ∨ B) ∧ (A ∨ C) → A ∨ (B ∧ C))

A partire da ciascuna tautologia, il modulo Instantiator genera istanze uniche sostituendo le metavariabili con sottoformule casuali, normalizzate e costruite fino a una profondità massima di 5, utilizzando al più 7 lettere proposizionali. Questa procedura assicura la coerenza strutturale tra le formule instanziate e quelle generate casualmente. Il processo di instanziazione è inoltre controllato per evitare duplicati, e ogni formula generata viene verificata con la funzione is_tautology per garantire che conservi la proprietà tautologica.
In totale, sono state generate e aggiunte al dataset 3.000 nuove formule tautologiche, portando la percentuale della classe tautologica al 26,29% dell’intero dataset.


**4.2 Preprocessing e Architettura del Modello TreeLSTM**
Dopo la generazione del dataset, ogni formula proposizionale viene trasformata in un albero sintattico binario che ne riflette la struttura logica. A questo scopo utilizziamo la classe FormulaTreeNode, che costruisce ricorsivamente una rappresentazione ad albero in cui ogni nodo corrisponde a una sottoformula, e i figli rappresentano gli argomenti del connettivo logico associato al nodo.

Dopo la generazione del dataset, ogni formula proposizionale viene trasformata in un albero sintattico binario che ne riflette la struttura logica composizionale. Questo albero è costruito ricorsivamente dalla classe FormulaTreeNode: ogni nodo 
n dell’albero rappresenta una sottoformula f∈F, dove F è l’insieme delle formule ben formate, e i suoi figli rappresentano gli argomenti del connettivo principale applicato in f. La struttura è binaria: il connettioi unario (e.g. ¬f ) genera un solo figlio, mentre i connettivi binari (e.g. f1 -> f2, ecc.) generano due figli.
Per consentire l’elaborazione numerica da parte del modello neurale, ogni nodo n dell’albero viene annotato con un indice intero e(n)∈N, che identifica simbolicamente la sottoformula f contenuta in n. Questa annotazione è prodotta da un tokenizer T:F→N, che implementa la seguente logica di mappatura:

- per ongi lettera propoizionale Ak, si assegna il token T(Ak) = k+1, con k ≥ 0;
- per ogni connettivo logico ∘∈{¬,∧,∨,→}, si assegna un token T(∘)∈{100,101,102,103}, specifico per il tipo di connettivo.
- per la costante ⊥, si assegna il token T(⊥) = 105,
- per i simboli di parentsi, si assegnano i token T('(') = 106, T(')') = 107

Nel dettaglio:

- se f è una formula atomica, il tokenizer assegna direttamente il token e(n) = T(f)
- se invece f è una formula composta, costruita tramite un connettivo, allotara il tokenizer assegna un token basato sulla classe sintattica del connettivo principale della formula f, usando e(n) 0 T(Type(f))

Questo valore intero e(n) ∈ ℕ viene memorizzato localmente nel campo embedding_index del nodo n, e viene poi trasformato in un vettore numerico dal modello tramite una matrice di embedding appresa. In questo modo, ogni simbolo logico è rappresentato da un vettore continuo, utilizzabile dal modello neurale come input numerico.

Esempio di albero con indici di embedding
Consideriamo la formula: ¬¬(A0 ∨ A1 ∨ A2 ∨ A3)

Questa formula viene dapprima convertita in un albero binario dove ogni sottoformula è rappresentata da un nodo. Dopo l’annotazione con il tokenizer, ogni nodo contiene anche un indice intero. La struttura risultante è la seguente (riportata con notazione ad albero e indici numerici):

└── [Negation] ¬¬(A0 ∨ A1 ∨ A2 ∨ A3)        (embedding_index = 102)
    └── [Negation] ¬(A0 ∨ A1 ∨ A2 ∨ A3)     (embedding_index = 102)
        └── [Disjunction] A0 ∨ A1 ∨ A2 ∨ A3 (embedding_index = 101)
            ├── [Disjunction] A0 ∨ A1       (embedding_index = 101)
            │   ├── [Letter] A0             (embedding_index = 1)
            │   └── [Letter] A1             (embedding_index = 2)
            └── [Disjunction] A2 ∨ A3       (embedding_index = 101)
                ├── [Letter] A2             (embedding_index = 3)
                └── [Letter] A3             (embedding_index = 4)



Ogni nodo contiene:

la sottoformula f di cui è radice,
il tipo di f, e il token assegnato e(n), che identifica simbolicamente quella struttura.
Questa rappresentazione consente al modello TreeLSTM di navigare e comporre ricorsivamente i vettori corrispondenti a ciascun nodo, riflettendo esattamente la composizione sintattica della formula.


Dataset e DataLoader per strutture ad albero
Per l’addestramento del modello, abbiamo definito una classe TreeFormulaDataset che implementa un dataset personalizzato, in cui ogni esempio è composto da:
- un albero sintattico binario (FormulaTreeNode) annotato con indici di embedding, che rappresenta la struttura logica della formula;
- un’etichetta booleana che indica se la formula è una tautologia o meno.
A causa della variabilità strutturale tra le formule — che possono differire significativamente in profondità e ampiezza — non è possibile impiegare meccanismi di batching standard basati su tensori di dimensione fissa. Per questo motivo, utilizziamo una funzione di collation personalizzata, tree_collate_fn, che aggrega dinamicamente un batch di esempi come:
- una lista di alberi (List[FormulaTreeNode]) da processare individualmente dal modello,
- un tensore PyTorch contenente le etichette booleane corrispondenti.
Questa soluzione mantiene la compatibilità con l’interfaccia DataLoader di PyTorch, consentendo un'integrazione efficiente con modelli che operano direttamente su strutture ad albero.

Architettura del TreeLSTM

Il classificatore adottato si basa sul modello TreeLSTM binario proposto da Tai et al. (2015), progettato per strutture gerarchiche e non sequenziali. Il modello ricorsivo processa l’albero dal basso verso l’alto, combinando le rappresentazioni dei figli tramite gate LSTM modificati.

Ogni nodo riceve un embedding x∈R^d e i due stati dei figli:
- sinistro: (h_l, c_l)
- destro: (h_r, c_r)
Le equazioni della Binary TreeLSTM cell sono: ..... vedi appunti......

Per i nodi unari (come la negazione), lo stesso figlio è passato due volte al posto di sinistro e destro.

Classificatore

L’output del TreeLSTM (lo stato h della radice) viene processato da un piccolo MLP:
un layer fully connected con RelU, un dropout per regolarizzazione un layer di output a neurone singolo per la classificazione binaria.
L’intera architettura è quindi:

TreeLSTMClassifier(x)=sigmoid(fc2(Dropout(ReLU(fc1(h))))) 

Dove h è l’output della TreeLSTM sulla radice dell’albero.

Funzione di Loss

Dato lo sbilanciamento del dataset (26.29% tautologie), abbiamo utilizzato una variante della Focal Loss denominata Asymmetric Focal Loss. Tale loss penalizza maggiormente i falsi negativi nella classe minoritaria (tautologie), con due parametri indipendenti per ciascuna classe:

vedi appunti ..........


Gli iperparametri principali dell’architettura sono i seguenti:


Dimensione embedding	32
Hidden size TreeLSTM	128
Dimensione FC layer	32
Dropout rate	0.3
Ottimizzatore	Adam
Batch size	32
Epoche max	20


EMBEDDING_DIM = 32
HIDDEN_SIZE = 128
FC_SIZE = 32
LR = 0.0009

ALPHA_POS = 0.25
ALPHA_NEG = 0.7
GAMMA_POS = 3
GAMMA_NEG = 2.5


Strategia di valutazione

I modelli sono stati valutati usando accuracy, precision, recall, e F1-score, con particolare attenzione al recall della classe tautologia, data la sua importanza nei contesti logici o critici. In fase di training, i dati sono stati suddivisi in train/test con una proporzione dell’80/20, assicurando che la distribuzione delle classi fosse mantenuta anche nello split (stratificazione).

















































