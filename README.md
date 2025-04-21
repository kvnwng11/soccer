# Champions League 2022-2023 Final

A prediction of the 2022-23 Champions League final.

# Rankings

Using the Bradley-Terry model, here are the strength rankings of all 32 teams:

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


[top](#champions-league-2022-2023-final)

---

# Prediction

The probability that Man City beats Inter Milan in the final is:


```python
prob = pred_strength["Man City"] / (pred_strength["Man City"] + pred_strength["Inter"])
print(prob)
```

    0.6414839183116747



