# Architecture and algorithm notes

```mermaid
flowchart TD
  Person[User] --> Browser[HTML CSS JavaScript dashboard]
  Browser --> API[Flask REST API]
  API --> Classifier[MobileNetV2 transfer learning]
  API --> Decision[Decision service: four 0–100 measurements]
  Decision --> Fuzzy[Gaussian fuzzification + category-aware Sugeno rule aggregation]
  ANFIS[Hybrid ANFIS fitting module] --> Fuzzy
  GP[Tree genetic programming] --> Rules[Executable boolean decision trees]
  Rules --> Decision
  GA[Real-valued genetic algorithm] --> MF[Ordered Gaussian parameters]
  MF --> Fuzzy
  API --> SQLite[(SQLite analyses and evolution runs)]
```

```mermaid
flowchart LR
  Upload --> Classify --> Correct[User category correction] --> Inputs[Condition sliders]
  Inputs --> Fuzzify --> Fire[Rule firing and normalization] --> Scores[Suitability and priority]
  Scores --> Explain[Activated rule explanation and recommendation]
  Scores --> Store[(SQLite)]
```

```mermaid
flowchart LR
  User((User)) --> Upload[Upload waste photo]
  User --> Inspect[Inspect analysis and reports]
  User --> Evolve[Run GP or GA]
  Upload --> Choose[Correct material class]
  Choose --> Analyze[Analyze material condition]
  Analyze --> Result[Read score and explanation]
```

```mermaid
flowchart TB
  UI[Frontend] --> Routes[Flask routes]
  Routes --> Classifier[WasteClassifier]
  Routes --> Engine[Fuzzy rules / Sugeno ANFIS]
  Routes --> GP[GeneticProgramming]
  Routes --> GA[GeneticAlgorithm]
  Routes --> DB[SQLite store]
```

## Methods

- **Gaussian membership:** `μ(x;c,σ) = exp(-0.5 ((x-c)/σ)^2)` with `c∈[0,100]`, `σ∈[1,60]`.
- **Fuzzy inference:** antecedent conjunction uses minimum firing, normalized firing strengths form the weights, and first-order weighted consequents yield crisp suitability and priority independently. Rules with a category condition have zero firing for other categories.
- **ANFIS:** three selected rules avoid a 3⁴ grid. Consequent coefficients are estimated by least squares; premise parameters receive bounded finite-difference gradient updates. Training requires an explicit numeric target, not labels invented by the application.
- **GP:** a tree node is an executable boolean or comparison expression. Tournament selection and elitism preserve strong trees; generated subtrees supply variation. Fitness combines accuracy and a size penalty.
- **GA:** chromosomes contain LOW/MEDIUM/HIGH center and width pairs. Decoding bounds widths and sorts/enforces center separation. Fitness is calculated against labeled synthetic teaching scenarios until a real target dataset is supplied.
- **Classifier:** MobileNetV2 is fine-tuned by `experiments/train_classifier.py`. Without the resulting checkpoint, image inference returns an unavailable status and no predicted class.
