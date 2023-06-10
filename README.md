# Champions League 2022-2023 Final
#### by Kevin Wang, June 6, 2023

On June 10 2023, Manchester City will face off against Inter Milan in the Champions League final. Manchester City defeated last year's winner Real Madrid en route to the final while Inter fended off city rivals AC Milan. Inter will be fighting for their 4th title while Manchester City have never won the Champions League. 

Using statistics and the Bradley-Terry-Luce model (BTL), we can assess the probability that Manchester City wins against Inter.

Given teams $i$ and $j$, the BTL model estimages the probability that the pairwise comparison $i > j$ turns out true

$$ P(i > j) = \frac{p_i}{p_i + p_j}$$

where $p_i$ is a score assigned to team $i$ and measures the overall "quality" of the team.

---

## The Model

Now assume that we have $n = 32$ teams where each team $j$ has a known feature vector $\mathbf{U_j} \in \mathbb{R}^d$. We also assume that there is a *universal ranking* where each team can be ranked according to how good they are. This ranking is determined by an unknown weights vector $\mathbf{w} \in \mathbb{R}^d$ where the weights signify the importance of each feature in the ranking. The predicted match outcomes are independent Bernoulli are defined as

$$ P(i \text{ beats } j) = \frac{e^{\langle \mathbf{U_i} , \mathbf{w} \rangle}}{e^{\langle \mathbf{U_i} , \mathbf{w} \rangle} + e^{\langle \mathbf{U_j} , \mathbf{w} \rangle}} $$

We learn each $p_i = e^{\langle \mathbf{U_i} , \mathbf{w} \rangle}$ from data with maximum likelihood estimation. This can be solved with Logistic Regression.

The model can predict hypothetical match-ups, which we will employ to predict the outcome of the final.

---

## Feature Extracting

First, let's import our necessary packages.


```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
```

Each feature vector $\mathbf{U_j} \in \mathbb{R}^d$ contains the following statistics:

- Games won
- Games lost
- Games tied

as well as the following *per-game* statistics:

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

All statistics were scraped from the offical UEFA Champions League website. The following cell loads in the data and reformats it into a Pandas dataframe:


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

The following cell prints out the raw features:


```python
features
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ajax</th>
      <th>Atletico</th>
      <th>Barcelona</th>
      <th>Bayern</th>
      <th>Benfica</th>
      <th>Celtic</th>
      <th>Chelsea</th>
      <th>Club Brugge</th>
      <th>Copenhagen</th>
      <th>Dinamo Zagreb</th>
      <th>...</th>
      <th>Paris</th>
      <th>Plzen</th>
      <th>Porto</th>
      <th>Rangers</th>
      <th>Real Madrid</th>
      <th>Salzburg</th>
      <th>Sevilla</th>
      <th>Shaktar Donetsk</th>
      <th>Sporting CP</th>
      <th>Tottenham</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Games won</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>8.00</td>
      <td>6.00</td>
      <td>0.000000</td>
      <td>5.00</td>
      <td>3.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.000</td>
      <td>0.000000</td>
      <td>4.000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00</td>
      <td>2.000000</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>Games lost</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>2.000000</td>
      <td>1.00</td>
      <td>2.000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000</td>
      <td>0.000000</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.00</td>
      <td>1.000000</td>
      <td>3.000</td>
    </tr>
    <tr>
      <th>Games tied</th>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>4.000000</td>
      <td>4.00</td>
      <td>3.000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>2.000</td>
      <td>6.000000</td>
      <td>3.000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.00</td>
      <td>3.000000</td>
      <td>2.000</td>
    </tr>
    <tr>
      <th>Goals scored</th>
      <td>1.840000</td>
      <td>0.840000</td>
      <td>2.000000</td>
      <td>2.21</td>
      <td>2.60</td>
      <td>0.670000</td>
      <td>1.20</td>
      <td>1.000</td>
      <td>0.170000</td>
      <td>0.670000</td>
      <td>...</td>
      <td>2.000</td>
      <td>0.840000</td>
      <td>1.500</td>
      <td>0.340000</td>
      <td>2.170000</td>
      <td>0.840000</td>
      <td>1.000000</td>
      <td>1.34</td>
      <td>1.340000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>Goals conceded</th>
      <td>2.670000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>0.60</td>
      <td>1.30</td>
      <td>2.500000</td>
      <td>0.90</td>
      <td>1.380</td>
      <td>2.000000</td>
      <td>1.840000</td>
      <td>...</td>
      <td>1.250</td>
      <td>4.000000</td>
      <td>1.000</td>
      <td>3.670000</td>
      <td>1.090000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>1.67</td>
      <td>1.500000</td>
      <td>0.880</td>
    </tr>
    <tr>
      <th>Possession %</th>
      <td>53.670000</td>
      <td>54.000000</td>
      <td>62.500000</td>
      <td>54.60</td>
      <td>52.50</td>
      <td>44.170000</td>
      <td>55.30</td>
      <td>46.880</td>
      <td>38.670000</td>
      <td>47.500000</td>
      <td>...</td>
      <td>53.750</td>
      <td>32.670000</td>
      <td>48.000</td>
      <td>40.670000</td>
      <td>52.750000</td>
      <td>40.500000</td>
      <td>50.500000</td>
      <td>44.50</td>
      <td>47.670000</td>
      <td>48.880</td>
    </tr>
    <tr>
      <th>Passing accuracy</th>
      <td>85.000000</td>
      <td>86.500000</td>
      <td>88.840000</td>
      <td>88.30</td>
      <td>86.70</td>
      <td>87.500000</td>
      <td>87.30</td>
      <td>82.000</td>
      <td>82.000000</td>
      <td>82.670000</td>
      <td>...</td>
      <td>89.880</td>
      <td>76.170000</td>
      <td>80.250</td>
      <td>79.500000</td>
      <td>89.920000</td>
      <td>71.840000</td>
      <td>85.340000</td>
      <td>86.50</td>
      <td>81.840000</td>
      <td>85.000</td>
    </tr>
    <tr>
      <th>Balls recovered</th>
      <td>41.000000</td>
      <td>39.340000</td>
      <td>38.500000</td>
      <td>42.60</td>
      <td>41.10</td>
      <td>36.340000</td>
      <td>44.20</td>
      <td>36.880</td>
      <td>36.840000</td>
      <td>39.340000</td>
      <td>...</td>
      <td>41.630</td>
      <td>37.170000</td>
      <td>41.130</td>
      <td>36.500000</td>
      <td>35.920000</td>
      <td>45.340000</td>
      <td>34.500000</td>
      <td>39.00</td>
      <td>37.670000</td>
      <td>40.750</td>
    </tr>
    <tr>
      <th>Tackles won</th>
      <td>7.000000</td>
      <td>4.670000</td>
      <td>4.000000</td>
      <td>5.50</td>
      <td>5.40</td>
      <td>4.840000</td>
      <td>3.80</td>
      <td>3.250</td>
      <td>2.670000</td>
      <td>5.000000</td>
      <td>...</td>
      <td>5.000</td>
      <td>3.340000</td>
      <td>4.380</td>
      <td>6.000000</td>
      <td>5.090000</td>
      <td>7.670000</td>
      <td>5.840000</td>
      <td>5.67</td>
      <td>4.000000</td>
      <td>3.250</td>
    </tr>
    <tr>
      <th>Clean sheets</th>
      <td>0.170000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.70</td>
      <td>0.20</td>
      <td>0.000000</td>
      <td>0.30</td>
      <td>0.630</td>
      <td>0.340000</td>
      <td>0.170000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.500</td>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.170000</td>
      <td>0.340000</td>
      <td>0.00</td>
      <td>0.340000</td>
      <td>0.380</td>
    </tr>
    <tr>
      <th>Saves</th>
      <td>3.670000</td>
      <td>3.170000</td>
      <td>3.170000</td>
      <td>3.10</td>
      <td>1.90</td>
      <td>2.670000</td>
      <td>2.70</td>
      <td>4.380</td>
      <td>4.500000</td>
      <td>4.170000</td>
      <td>...</td>
      <td>3.250</td>
      <td>5.340000</td>
      <td>3.880</td>
      <td>4.840000</td>
      <td>3.750000</td>
      <td>3.670000</td>
      <td>3.000000</td>
      <td>5.17</td>
      <td>3.670000</td>
      <td>2.880</td>
    </tr>
    <tr>
      <th>Distance covered (km)</th>
      <td>113.370000</td>
      <td>118.390000</td>
      <td>118.640000</td>
      <td>120.19</td>
      <td>108.89</td>
      <td>101.370000</td>
      <td>107.63</td>
      <td>118.570</td>
      <td>120.970000</td>
      <td>109.080000</td>
      <td>...</td>
      <td>108.520</td>
      <td>110.130000</td>
      <td>120.830</td>
      <td>110.940000</td>
      <td>101.360000</td>
      <td>116.080000</td>
      <td>114.720000</td>
      <td>117.22</td>
      <td>109.160000</td>
      <td>120.380</td>
    </tr>
    <tr>
      <th>Yellow cards</th>
      <td>2.500000</td>
      <td>2.170000</td>
      <td>1.670000</td>
      <td>2.30</td>
      <td>1.90</td>
      <td>1.170000</td>
      <td>2.50</td>
      <td>3.380</td>
      <td>2.000000</td>
      <td>2.170000</td>
      <td>...</td>
      <td>1.750</td>
      <td>2.000000</td>
      <td>3.130</td>
      <td>1.670000</td>
      <td>1.250000</td>
      <td>1.840000</td>
      <td>2.170000</td>
      <td>2.67</td>
      <td>2.840000</td>
      <td>2.750</td>
    </tr>
    <tr>
      <th>Red cards</th>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.10</td>
      <td>0.130</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.170000</td>
      <td>0.380</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.500000</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>Shots on goal</th>
      <td>3.833333</td>
      <td>5.833333</td>
      <td>5.666667</td>
      <td>7.20</td>
      <td>5.20</td>
      <td>4.833333</td>
      <td>5.40</td>
      <td>2.875</td>
      <td>2.666667</td>
      <td>3.333333</td>
      <td>...</td>
      <td>5.375</td>
      <td>3.333333</td>
      <td>5.000</td>
      <td>2.000000</td>
      <td>6.333333</td>
      <td>4.333333</td>
      <td>4.000000</td>
      <td>2.50</td>
      <td>3.500000</td>
      <td>3.875</td>
    </tr>
    <tr>
      <th>Corners</th>
      <td>4.333333</td>
      <td>6.500000</td>
      <td>6.666667</td>
      <td>3.80</td>
      <td>4.90</td>
      <td>5.333333</td>
      <td>5.30</td>
      <td>3.500</td>
      <td>3.500000</td>
      <td>2.500000</td>
      <td>...</td>
      <td>4.875</td>
      <td>3.666667</td>
      <td>4.375</td>
      <td>3.333333</td>
      <td>5.333333</td>
      <td>4.000000</td>
      <td>2.833333</td>
      <td>2.50</td>
      <td>3.333333</td>
      <td>5.625</td>
    </tr>
  </tbody>
</table>
<p>16 rows × 32 columns</p>
</div>



Since our data contains really large and really small numbers, it would be nice to normalize our features to be of the same magnitude. Note that this does not change the distribution of the features, so our model should still learn the same patterns in the data.


```python
# Normalize data
for idx, row in features.iterrows():
    mean = np.mean(row)
    var = np.var(row)

    for team in features:
        features[team][idx] = (features[team][idx] - mean) / var
```

Additionally, we don't want to include the first three rows (Games won, Games lost, Games tied) as they introduce bias into the model.


```python
features = features.iloc[3:]
```

---

## Match Outcomes from the 2022-23 Champions League Season

The following cell contains the match outcomes of every game played in the Champions League this season. Since the Bradley-Terry-Luce model does not use tie games, such games were omitted. A $0$ means that the first team won, and a $1$ means the second team won.

Additionally, the first team is the home team.


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

# Oops I have it backwards. The 1's need to be 0's and vice versa for logistic regression
for match in comparisons:
    match[2] = 1 - match[2]
```

---

## Model Training

Here we implement a Logistic Regression. We first construct the feature matrix $\mathbf{X}$ and the labels $\mathbf{y}$. We then compute the weights vector $\mathbf{w} \in \mathbb{R}^d$.

To prevent overfitting, we add $\lambda {\lVert \mathbf{w} \rVert}^2_2$ to the maximum likelihood equation. Here, $\lambda=5$ which means $C=0.1$ in sklearn. To incorporate home field advantage, we extend each feature vector by a $1$ or $0$ depending on if the team was home or away.


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

weights = np.reshape(LogisticRegression(C=0.1, penalty='l2').fit(X, y).coef_, (len(features.index)+1,))
print("Weights:", weights)
```

    Weights: [ 3.22762477e-01 -4.28347724e-01  2.87938458e-02  6.58029026e-02
      1.79935566e-01  4.13596290e-02  2.39909103e-01  3.49497391e-02
      6.04708499e-02 -2.21959969e-02 -4.55325989e-02  1.43537320e-01
      3.18883686e-02  3.91885805e-06]
    

It appears that the most important features when predicting who will win a match are goals per game, clean sheets per game, and corners per game. It also appears that home field advantage is negligible.

---

## Universal Rankings

Since we know $\mathbf{U_i} \in \mathbb{R}^d$ and $\mathbf{w} \in \mathbb{R}^d$, we can compute the quality of each team, $p_i = \exp(\mathbf{U_i} \cdot \mathbf{w})$. Here we rank each team according to their score.


```python
# Predict quality of each team
pred_score = {}
for column in features:
    # Prepare feature vector
    team = list(features[column])
    team.append(0)
    team = np.asarray(team, dtype='float64')

    # Predict
    pred_score[column] = np.exp(np.dot(team, weights))

# Sort scores
pred_score = dict(sorted(pred_score.items(), key=lambda x : x[1], reverse=True))

# Print rankings
rank = 1
print("-------- Rankings --------")
for team in pred_score:
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

Now let's estimate the chance that Man City defeats Inter. This is defined as

$$ P(i > j) = \frac{p_i}{p_i + p_j}$$

where $i = \text{Man City}$ and $j = \text{Inter}$. Since the final is played on neutral ground, there is no home field advantage.


```python
# Probability that Man City beats Inter
prob = pred_score["Man City"] / (pred_score["Man City"] + pred_score["Inter"])
print(prob)
```

    0.6414839183116584
    

Manchester City has a 64.1% chance of winning the Champions League.

---

# Why can this be solved with Logistic Regression?

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

## Summary

The log-likelihood of both Logistic Regression and the Bradley-Terry-Luce model can be written in the form:

$$
 L = \log(\prod_{i=1}^{n} \mathbb{P}(y_i = 1 | x_i, \theta)^{y_i} \mathbb{P}(y_i = 0 | x_i, \theta)^{1 - y_i} )
$$

so Bradley-Terry-Luce can be solved with Logistic Regression.

# Biases

The model suffers from bias. The first is a form of look-ahead bias. The feature vectors $\mathbf{U_J} \in \mathbb{R}^d$ contain information from the later stages of the Champions League, which was not present in the earlier games. Another form of bias is that the model does not consider home-field advantages. But because the final is played on neutral ground, this may not be an issue. 


