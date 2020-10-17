


### Dataset

- http://wdoor.c.u-tokyo.ac.jp/shogi/

### Data Cleansing

```
tf2 ❯ python src/utils/filter_csa.py dataset/2016/                                                                                                                                      ✘ 1 
kifu count:  29758
rate mean:   3063.189260030916
rate median: 3066.0
rate max:    3825.0
rate min:    2502.0
```

### Split train and valid


```
tf2 ❯ python src/utils/make_kifu_list.py dataset/2016 kifulist
total kifu size: 29758
train kifu size: 26782
valid kifu size: 2976
```

