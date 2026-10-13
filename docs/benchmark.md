| Dataset | Noise | Rate | k | Precision@k | Recall@k | Review n | Precision@review | Recall@review | 95% CI | Minority false positives | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| iris | uniform | 5% | 7 | 1.000 | 1.000 | 8 | 0.875 | 1.000 | 0.529–0.978 | setosa: 0, versicolor: 1, virginica: 0 | 1.000 |
| iris | uniform | 10% | 15 | 0.933 | 0.933 | 8 | 1.000 | 0.533 | 0.676–1.000 | setosa: 0, versicolor: 0, virginica: 0 | 0.994 |
| iris | uniform | 20% | 32 | 0.781 | 0.781 | 8 | 1.000 | 0.250 | 0.676–1.000 | setosa: 0, versicolor: 0, virginica: 0 | 0.968 |
| iris | class_conditional | 5% | 7 | 0.857 | 0.857 | 8 | 0.750 | 0.857 | 0.409–0.929 | setosa: 0, versicolor: 2, virginica: 0 | 0.996 |
| iris | class_conditional | 10% | 15 | 0.933 | 0.933 | 8 | 1.000 | 0.533 | 0.676–1.000 | setosa: 0, versicolor: 0, virginica: 0 | 0.998 |
| iris | class_conditional | 20% | 32 | 0.844 | 0.844 | 8 | 0.875 | 0.219 | 0.529–0.978 | setosa: 1, versicolor: 0, virginica: 0 | 0.966 |
| wine | uniform | 5% | 9 | 1.000 | 1.000 | 9 | 1.000 | 1.000 | 0.701–1.000 | class_2: 0 | 1.000 |
| wine | uniform | 10% | 17 | 1.000 | 1.000 | 9 | 1.000 | 0.529 | 0.701–1.000 | class_2: 0 | 1.000 |
| wine | uniform | 20% | 38 | 0.921 | 0.921 | 9 | 1.000 | 0.237 | 0.701–1.000 | class_2: 0 | 0.982 |
| wine | class_conditional | 5% | 9 | 0.778 | 0.778 | 9 | 0.778 | 0.778 | 0.453–0.937 | class_2: 0 | 0.995 |
| wine | class_conditional | 10% | 17 | 0.882 | 0.882 | 9 | 1.000 | 0.529 | 0.701–1.000 | class_2: 0 | 0.996 |
| wine | class_conditional | 20% | 38 | 0.842 | 0.842 | 9 | 0.889 | 0.211 | 0.565–0.980 | class_2: 0 | 0.970 |
| digits | uniform | 5% | 81 | 0.951 | 0.951 | 90 | 0.889 | 0.988 | 0.807–0.939 | 8: 1 | 0.999 |
| digits | uniform | 10% | 165 | 0.945 | 0.945 | 90 | 1.000 | 0.545 | 0.959–1.000 | 8: 0 | 0.999 |
| digits | uniform | 20% | 372 | 0.962 | 0.962 | 90 | 1.000 | 0.242 | 0.959–1.000 | 8: 0 | 0.997 |
| digits | class_conditional | 5% | 81 | 0.889 | 0.889 | 90 | 0.856 | 0.951 | 0.768–0.914 | 8: 2 | 0.997 |
| digits | class_conditional | 10% | 165 | 0.879 | 0.879 | 90 | 0.944 | 0.515 | 0.876–0.976 | 8: 1 | 0.990 |
| digits | class_conditional | 20% | 372 | 0.806 | 0.806 | 90 | 0.922 | 0.223 | 0.848–0.962 | 8: 1 | 0.956 |
| synthetic_text | uniform | 5% | 7 | 1.000 | 1.000 | 8 | 0.875 | 1.000 | 0.529–0.978 | garden: 0, harbor: 0, kitchen: 0, orbit: 1 | 1.000 |
| synthetic_text | uniform | 10% | 15 | 0.933 | 0.933 | 8 | 1.000 | 0.533 | 0.676–1.000 | garden: 0, harbor: 0, kitchen: 0, orbit: 0 | 0.998 |
| synthetic_text | uniform | 20% | 34 | 0.941 | 0.941 | 8 | 1.000 | 0.235 | 0.676–1.000 | garden: 0, harbor: 0, kitchen: 0, orbit: 0 | 0.995 |
| synthetic_text | class_conditional | 5% | 7 | 0.857 | 0.857 | 8 | 0.875 | 1.000 | 0.529–0.978 | garden: 0, harbor: 1, kitchen: 0, orbit: 0 | 0.999 |
| synthetic_text | class_conditional | 10% | 15 | 0.867 | 0.867 | 8 | 0.875 | 0.467 | 0.529–0.978 | garden: 0, harbor: 1, kitchen: 0, orbit: 0 | 0.993 |
| synthetic_text | class_conditional | 20% | 34 | 0.912 | 0.912 | 8 | 1.000 | 0.235 | 0.676–1.000 | garden: 0, harbor: 0, kitchen: 0, orbit: 0 | 0.976 |
| synthetic_ambiguous_minority | uniform | 5% | 9 | 0.778 | 0.778 | 9 | 0.778 | 0.778 | 0.453–0.937 | minority: 2 | 0.970 |
| synthetic_ambiguous_minority | uniform | 10% | 17 | 0.765 | 0.765 | 9 | 0.778 | 0.412 | 0.453–0.937 | minority: 2 | 0.987 |
| synthetic_ambiguous_minority | uniform | 20% | 38 | 0.737 | 0.737 | 9 | 0.889 | 0.211 | 0.565–0.980 | minority: 1 | 0.930 |
| synthetic_ambiguous_minority | class_conditional | 5% | 9 | 0.667 | 0.667 | 9 | 0.667 | 0.667 | 0.354–0.879 | minority: 3 | 0.978 |
| synthetic_ambiguous_minority | class_conditional | 10% | 17 | 0.765 | 0.765 | 9 | 0.667 | 0.353 | 0.354–0.879 | minority: 3 | 0.975 |
| synthetic_ambiguous_minority | class_conditional | 20% | 38 | 0.711 | 0.711 | 9 | 0.556 | 0.132 | 0.267–0.811 | minority: 4 | 0.932 |
