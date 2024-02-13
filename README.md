# Champions League 2022-2023 Final

## Feature Extracting


```python
import numpy as np
import pandas as pd
import random as rd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
```

Each feature vector $\mathbf{U_j} \in \mathbb{R}^d$ contains the following statistics taken from the 2022-23 Champions League season:

- Games won
- Games lost
- Games tied

as well as the following per-game statistics:

- Goals scored 
- Goals conceded
- Possession %
- Passing accuracy
- Balls recovered
- Tackles won
- Clean sheets
- Saves
- Distance covered (km)
- Yellow cards
- Red cards
- Shots on goal
- Corners


```python
features = [
    ["Ajax", 2, 0, 4, 1.84, 2.67, 53.67, 85, 41, 7, 0.17, 3.67, 113.37, 2.5, 0.17, 23/6, 26/6],
    ["Atletico", 1, 2, 3, 0.84, 1.5, 54, 86.5, 39.34, 4.67, 0.17, 3.17, 118.39, 2.17, 0, 35/6, 39/6],
    ["Barcelona", 2, 1, 3, 2, 2, 62.5, 88.84, 38.5, 4, 0, 3.17, 118.64, 1.67, 0, 34/6, 40/6],
    ["Bayern", 8, 1, 1, 2.21, 0.6, 54.6, 88.3, 42.6, 5.5, 0.7, 3.1, 120.19, 2.3, 0.2, 72/10, 38/10],
    ["Benfica", 6, 3, 1, 2.6, 1.3, 52.5, 86.7, 41.1, 5.4, 0.2, 1.9, 108.89, 1.9, 0, 52/10, 49/10],
    ["Celtic", 0, 2, 4, 0.67, 2.5, 44.17, 87.5, 36.34, 4.84, 0, 2.67, 101.37, 1.17, 0, 29/6, 32/6],
    ["Chelsea", 5, 1, 4, 1.2, 0.9, 55.3, 87.3, 44.2, 3.8, 0.3, 2.7, 107.63, 2.5, 0.1, 54/10, 53/10],
    ["Club Brugge", 3, 2, 3, 1, 1.38, 46.88, 82, 36.88, 3.25, 0.63, 4.38, 118.57, 3.38, 0.13, 23/8, 28/8],
    ["Copenhagen", 0, 3, 3, 0.17, 2, 38.67, 82, 36.84, 2.67, 0.34, 4.5, 120.97, 2, 0.17, 16/6, 21/6],
    ["Dinamo Zagreb", 1, 1, 4, 0.67, 1.84, 47.5, 82.67, 39.34, 5, 0.17, 4.17, 109.08, 2.17, 0, 20/6, 15/6],
    ["Dortmund", 3, 3, 2, 1.38, 0.88, 48.88, 84, 40.38, 4.75, 0.38, 3.75, 113.16, 2.13, 0, 25/8, 27/8],
    ["Frankfurt", 3, 1, 4, 0.88, 1.63, 42.88, 79.88, 47, 4.63, 0.25, 3.63, 119.72, 2.25, 0.25, 27/8, 26/8],
    ["Inter", 7, 3, 2, 1.59, 0.84, 46.42, 82.92, 38.42, 6.92, 0.67, 3.75, 117.54, 2, 0.09, 57/12, 61/12],
    ["Juventus", 1, 0, 5, 1.5, 2.17, 49.17, 86.34, 36.67, 5.84, 0, 3.34, 102.95, 2.34, 0, 26/6, 27/6],
    ["Leipzig", 4, 1, 3, 1.75, 2.13, 49.63, 86.75, 39.63, 3.88, 0.25, 2.75, 103.74, 1, 0, 34/8, 32/8],
    ["Leverkusen", 1, 2, 3, 0.67, 1.34, 52.34, 84.67, 44.34, 4.5, 0.34, 3, 118.5, 3.17, 0.17, 29/6, 31/6],
    ["Liverpool", 5, 0, 3, 2.38, 1.5, 53.38, 85.5, 40.25, 5.88, 0.38, 2.13, 115.1, 1.38, 0, 56/8, 59/8],
    ["M. Haifa", 1, 0, 5, 1.17, 3.5, 45.67, 84.5, 40.67, 3.34, 0.17, 3.5, 98.86, 2.34, 0, 22/6, 5],
    ["Man City", 7, 5, 0, 2.59, 0.42, 59.84, 90.42, 37.42, 3.42, 0.59, 2.17, 116.48, 1.67, 0.09, 76/12, 85/12], 
    ["Marseille", 2, 0, 4, 1.34, 1.34, 59, 87, 40.5, 1.84, 0.17, 2, 113.04, 1.84, 0.17, 4, 5],
    ["Milan", 5, 3, 4, 1.25, 0.92, 50.42, 83, 37.67, 5.92, 0.42, 3.42, 112.39, 2.5, 0.09, 45/12, 36/12],
    ["Napoli", 7, 1, 2, 2.6, 0.8, 54.6, 86.3, 41.9, 4, 0.4, 2.3, 114.79, 2, 0.1, 71/10, 58/10],
    ["Paris", 4, 2, 2, 2, 1.25, 53.75, 89.88, 41.63, 5, 0, 3.25, 108.52, 1.75, 0, 43/8, 39/8],
    ["Plzen", 0, 0, 6, 0.84, 4, 32.67, 76.17, 37.17, 3.34, 0, 5.34, 110.13, 2, 0.17, 20/6, 22/6],
    ["Porto", 4, 1, 3, 1.5, 1, 48, 80.25, 41.13, 4.38, 0.5, 3.88, 120.83, 3.13, 0.38, 40/8, 35/8],
    ["Rangers", 0, 0, 6, 0.34, 3.67, 40.67, 79.5, 36.5, 6, 0, 4.84, 110.94, 1.67, 0.17, 2, 20/6],
    ["Real Madrid", 8, 2, 2, 2.17, 1.09, 52.75, 89.92, 35.92, 5.09, 0.42, 3.75, 101.36, 1.25, 0, 76/12, 64/12],
    ["Salzburg", 1, 3, 2, 0.84, 1.5, 40.5, 71.84, 45.34, 7.67, 0.17, 3.67, 116.08, 1.84, 0, 26/6, 4],
    ["Sevilla", 1, 2, 3, 1, 2, 50.5, 85.34, 34.5, 5.84, 0.34, 3, 114.72, 2.17, 0, 4, 17/6],
    ["Shaktar Donetsk", 1, 3, 2, 1.34, 1.67, 44.5, 86.5, 39, 5.67, 0, 5.17, 117.22, 2.67, 0, 15/6, 15/6],
    ["Sporting CP", 2, 1, 3, 1.34, 1.5, 47.67, 81.84, 37.67, 4, 0.34, 3.67, 109.16, 2.84, 0.5, 21/6, 20/6],
    ["Tottenham", 3, 3, 2, 1, 0.88, 48.88, 85, 40.75, 3.25, 0.38, 2.88, 120.38, 2.75, 0.25, 31/8, 45/8],
]

# Reformat data
features = np.array(features)
features = np.transpose(features)
features = pd.DataFrame(features)
features.rename(columns=features.iloc[0], inplace = True)
features.drop(features.index[0], inplace = True)
features = features.astype(float)
features = features.rename(index={
    1: "Games won",
    2: "Games lost",
    3: "Games tied",
    4: "Goals scored", 
    5: "Goals conceded", 
    6: "Possession %", 
    7: "Passing accuracy", 
    8: "Balls recovered", 
    9: "Tackles won", 
    10: "Clean sheets", 
    11: "Saves", 
    12: "Distance covered (km)", 
    13: "Yellow cards", 
    14: "Red cards", 
    15: "Shots on goal", 
    16: "Corners"              
})
```

Since the data contains really large and really small numbers, it makes sense to normalize the features.


```python
# Normalize data
for idx, row in features.iterrows():
    mean = np.mean(row)
    var = np.var(row)

    for team in features:
        features[team][idx] = (features[team][idx] - mean) / var
```

Additionally, the first three rows (Games won, Games lost, Games tied) introduce bias. It makes sense to remove them too.


```python
features = features.iloc[3:]
```

---

## Match Outcomes

The following cell contains the outcomes of every win/loss game in the 2022-23 season. A $0$ means that the first (home) team won, and a $1$ means the second (away) team won. 


```python
comparisons = [
    ["Man City", "Real Madrid", 0],
    ["Inter", "Milan", 0],
    ["Milan", "Inter", 1],
    ["Chelsea", "Real Madrid", 1],
    ["Man City", "Bayern", 0],
    ["Benfica", "Inter", 1],
    ["Real Madrid", "Chelsea", 0],
    ["Milan", "Napoli", 0],
    ["Benfica", "Club Brugge", 0],
    ["Chelsea", "Dortmund", 0],
    ["Bayern", "Paris", 0],
    ["Man City", "Leipzig", 0],
    ["Napoli", "Frankfurt", 0],
    ["Real Madrid", "Liverpool", 0],
    ["Paris", "Bayern", 1],
    ["Milan", "Tottenham", 0],
    ["Benfica", "Club Brugge", 1],
    ["Dortmund", "Chelsea", 0],
    ["Liverpool", "Real Madrid", 0],
    ["Frankfurt", "Napoli", 1],
    ["Inter", "Porto", 0],
    ["Bayern", "Inter", 0],
    ["Liverpool", "Napoli", 0],
    ["Marseille", "Tottenham", 1],
    ["Real Madrid", "Celtic", 0],
    ["Shaktar Donetsk", "Leipzig", 1],
    ["Milan", "Salzburg", 0],
    ["M. Haifa", "Benfica", 1],
    ["Man City", "Sevilla", 0],
    ["Juventus", "Paris", 1],
    ["Chelsea", "Dinamo Zagreb", 0],
    ["Porto", "Atletico", 0],
    ["Rangers", "Ajax", 1],
    ["Plzen", "Barcelona", 1],
    ["Sporting CP", "Frankfurt", 1],
    ["Salzburg", "Chelsea", 1],
    ["Sevilla", "Copenhagen", 0],
    ["Paris", "M. Haifa", 0],
    ["Dinamo Zagreb", "Milan", 1],
    ["Benfica", "Juventus", 0],
    ["Leipzig", "Real Madrid", 0],
    ["Inter", "Plzen", 0],
    ["Club Brugge", "Porto", 1],
    ["Frankfurt", "Marseille", 0],
    ["Barcelona", "Bayern", 1],
    ["Ajax", "Liverpool", 1],
    ["Napoli", "Rangers", 0],
    ["M. Haifa", "Juventus", 0],
    ["Milan", "Chelsea", 1],
    ["Celtic", "Leipzig", 1],
    ["Napoli", "Ajax", 0],
    ["Leverkusen", "Porto", 1],
    ["Rangers", "Liverpool", 1],
    ["Plzen", "Bayern", 1],
    ["Tottenham", "Frankfurt", 0],
    ["Sporting CP", "Marseille", 1],
    ["Bayern", "Plzen", 0],
    ["Marseille", "Sporting CP", 0],
    ["Porto", "Leverkusen", 0],
    ["Club Brugge", "Atletico", 0],
    ["Ajax", "Napoli", 1],
    ["Inter", "Barcelona", 0],
    ["Liverpool", "Rangers", 0],
    ["Salzburg", "Dinamo Zagreb", 0],
    ["Leipzig", "Celtic", 0],
    ["Chelsea", "Milan", 0],
    ["Juventus", "M. Haifa", 0],
    ["Real Madrid", "Shaktar Donetsk", 0],
    ["Sevilla", "Dortmund", 1],
    ["Man City", "Copenhagen", 0],
    ["Plzen", "Inter", 1],
    ["Sporting CP", "Tottenham", 0],
    ["Liverpool", "Ajax", 0],
    ["Bayern", "Barcelona", 0],
    ["Porto", "Club Brugge", 1],
    ["Marseille", "Frankfurt", 1],
    ["Leverkusen", "Atletico", 0],
    ["Milan", "Dinamo Zagreb", 0],
    ["Rangers", "Napoli", 1],
    ["Real Madrid", "Leipzig", 0],
    ["Juventus", "Benfica", 1],
    ["M. Haifa", "Paris", 1],
    ["Man City", "Dortmund", 0],
    ["Dinamo Zagreb", "Chelsea", 0],
    ["Dortmund", "Copenhagen", 0],
    ["Benfica", "M. Haifa", 0],
    ["Sevilla", "Man City", 1],
    ["Celtic", "Real Madrid", 1],
    ["Leipzig", "Shaktar Donetsk", 1],
    ["Paris", "Juventus", 0],
    ["Ajax", "Rangers", 0],
    ["Frankfurt", "Sporting CP", 1],
    ["Inter", "Bayern", 1],
    ["Barcelona", "Plzen", 0],
    ["Napoli", "Liverpool", 0],
    ["Atletico", "Porto", 0],
    ["Club Brugge", "Leverkusen", 0],
    ["Tottenham", "Marseille", 0]
]

for match in comparisons:
    match[2] = 1 - match[2]
```

---

## Model Training



```python
### Construct logistic regression features matrix X and labels y
X = []
y = []
for comp in comparisons:
    team1 = comp[0]
    team2 = comp[1]
    who_won = comp[2]

    home_team = list(features[team1])
    away_team = list(features[team2])

    home_team.append(1)
    away_team.append(0)

    input_vector = np.asarray(home_team, dtype='float64') - np.asarray(away_team, dtype='float64')

    X.append(input_vector)
    y.append(who_won)

X = np.array(X)
y = np.array(y)

### Search for optimal lambda

# Helper function
def split(lst, k):
    n, m = divmod(len(lst), k)
    return (lst[i*n+min(i, m):(i+1)*n+min(i+1, m)] for i in range(k))

# Declare variables
search_space = np.linspace(0.1, 2, 20)
k = 6
num_games = X.shape[0]
indices = [i for i in range(num_games)]
min_loss = float('inf')
Lambda = 0

# Search parameter space
for lam in search_space:
    # Repeat for less variance
    for i in range(10):
        # Random partition of data
        shuffle = np.array(indices)
        rd.shuffle(shuffle)
        partition = list(split(shuffle, k))

        # Cross validation
        for p in partition:
            idx = np.delete(shuffle, p)
            sample_x = X[idx, :]
            sample_y = y[idx]

            LR = LogisticRegression(C=lam, penalty='l2').fit(X, y)
            y_pred = LR.predict(X[p, :])
            mse = mean_squared_error(y_pred, y[p])

            if mse < min_loss:
                min_loss = mse
                Lambda = lam

### Fit model
weights = np.reshape(LogisticRegression(C=Lambda, penalty='l2').fit(X, y).coef_, (len(features.index)+1,))
print("Weights:", weights)
```

    Weights: [ 3.22762477e-01 -4.28347724e-01  2.87938458e-02  6.58029026e-02
      1.79935566e-01  4.13596290e-02  2.39909103e-01  3.49497391e-02
      6.04708499e-02 -2.21959969e-02 -4.55325989e-02  1.43537320e-01
      3.18883686e-02  3.91885805e-06]


It appears that any home field advantage is negligible.

---

## Ranking

Here are the strength rankings of the teams:


```python
# Predict quality of each team
pred_strength = {}
for column in features:
    # Prepare feature vector
    team = list(features[column])
    team.append(0)
    team = np.asarray(team, dtype='float64')

    # Predict
    pred_strength[column] = np.exp(np.dot(team, weights))

# Sort scores
pred_strength = dict(sorted(pred_strength.items(), key=lambda x : x[1], reverse=True))

# Print rankings
rank = 1
print("-------- Rankings --------")
for team in pred_strength:
    output = str(rank) + ". " + str(team)
    print(output)
    rank += 1
```

    -------- Rankings --------
    1. Bayern
    2. Man City
    3. Inter
    4. Napoli
    5. Real Madrid
    6. Liverpool
    7. Club Brugge
    8. Dortmund
    9. Milan
    10. Benfica
    11. Porto
    12. Chelsea
    13. Tottenham
    14. Leipzig
    15. Sevilla
    16. Leverkusen
    17. Paris
    18. Salzburg
    19. Atletico
    20. Marseille
    21. Barcelona
    22. Frankfurt
    23. Sporting CP
    24. Ajax
    25. Dinamo Zagreb
    26. Copenhagen
    27. Shaktar Donetsk
    28. Juventus
    29. M. Haifa
    30. Celtic
    31. Plzen
    32. Rangers


---

# Final Match Prediction

The probability that Man City beats Inter Milan is:

$$ P(i > j) = \frac{s_i}{s_i + s_j}$$

where $i = \text{Man City}$ and $j = \text{Inter}$.


```python
prob = pred_strength["Man City"] / (pred_strength["Man City"] + pred_strength["Inter"])
print(prob)
```

    0.6414839183116747


---

# Proofs

## Logistic Regression

Suppose we have training data $x_1, x_2, ..., x_n$ and labels $y_1, y_2, ..., y_n$. To solve for the weights vector $\theta \in \mathbb{R}^d$, we use maximum likelihood estimation. The log-likelihood is then defined as:

$$
\begin{align}
    L &= \log(\prod_{i=1}^{n} \mathbb{P}(y_i | x_i, \theta) ) \\
    &= \log(\prod_{i=1}^{n} \mathbb{P}(y_i = 1 | x_i, \theta)^{y_i} \mathbb{P}(y_i = 0 | x_i, \theta)^{1 - y_i} ) \\
    &= \sum_{i=n}^{m} [y_i \log(\mathbb{P}(y_i = 1 | x_i, \theta)) + (1-y_i)\log(\mathbb{P}(y_i = 0 | x_i, \theta))] \\
    &= \sum_{i=n}^{m} [y_i \log(\frac{1}{1 + e^{-\langle \theta, x_i \rangle}}) + (1-y_i)\log(\frac{e^{-\langle \theta, x_i \rangle}}{1 + e^{-\langle \theta, x_i \rangle}})] \\
    &= \sum_{i=n}^{m} y_i[\log(\frac{1}{1 + e^{-\langle \theta, x_i \rangle}}) - \log(\frac{e^{-\langle \theta, x_i \rangle}}{1 + e^{-\langle \theta, x_i \rangle}})] + \log(\frac{e^{-\langle \theta, x_i \rangle}}{1 + e^{-\langle \theta, x_i \rangle}}) \\
    &= \sum_{i=n}^{m} y_i[\log(e^{\langle \theta, x_i \rangle})] + \log(\frac{e^{-\langle \theta, x_i \rangle}}{1 + e^{-\langle \theta, x_i \rangle}} * \frac{e^{\langle \theta, x_i \rangle}}{e^{\langle \theta, x_i \rangle}}) \\
    &= \sum_{i=n}^{m} y_i\langle \theta, x_i \rangle - \log(1 + e^{\langle \theta, x_i \rangle}) \\
\end{align}
$$

## Bradley-Terry-Luce

Let the following definitions hold:

- $\mathbf{U_j} \in \mathbb{R}^d$ is the feature vector of team $j$.
-  $\mathbf{w} \in \mathbb{R}^d$ is the weights vector.
- $m$ be the number of matches recorded.
- $y_\ell$ be the outcome of the $\ell$'th game. Note that $y_\ell \sim \text{Bern}(\mathbb{P}(i_\ell \text{ beats } j_\ell))$, so $y_\ell = 0$ if team $j_\ell$ wins and $y_\ell = 1$ if team $i_\ell$ wins.

The log-likehood of the Bradley-Terry-Luce model is 

$$
\begin{align}
    L &= \log(\prod_{\ell=1}^{m} (\mathbb{P}(y_\ell=1)^{y_\ell}   \mathbb{P}(y_\ell=0)^{1-y_\ell})) \\
    &= \sum_{\ell=1}^{m} [y_\ell \log(\mathbb{P}(y_\ell = 1)) + (1-y_\ell)\log(\mathbb{P}(y_\ell = 0))] \\
    &= \sum_{\ell=1}^{m} [y_\ell \log(\frac{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle}}{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle} + e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}}) 
    + (1-y_\ell)\log(\frac{e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}}{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle} + e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}})] \\
    &= \sum_{\ell=1}^{m} y_\ell[\log(\frac{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle}}{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle} + e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}}) 
    - \log(\frac{e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}}{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle} + e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}})] 
    + \log(\frac{e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}}{e^{\langle \mathbf{U_{i_\ell}} , \mathbf{w} \rangle} + e^{\langle \mathbf{U_{j_\ell}} , \mathbf{w} \rangle}}) \\
    &= \sum_{\ell=1}^{m} y_\ell\langle \mathbf{U_{i_\ell}} - \mathbf{U_{j_\ell}}, \mathbf{w} \rangle - \log(1+e^{\langle \mathbf{U_{i_\ell}} - \mathbf{U_{j_\ell}}, \mathbf{w} \rangle}) \\
\end{align}
$$

This is logistic regression with $x_\ell = \mathbf{U_{i_\ell}} - \mathbf{U_{j_\ell}}$.


