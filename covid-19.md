
For this project, I will build and train a simple neural network, using
[this
dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)
uploaded to Kaggle by user Meir Nizri. I will build a model that, given
a COVID-19 patient’s current symptom, status, and medical history, will
predict whether the patient is in high risk or not.

``` r
covid_data <- read_csv("covid_data.csv", show_col_types = FALSE)
```

As mentioned in the about page for this dataset, the raw dataset
consists of 21 unique features and 1,048,576 unique patients. In the
Boolean features, 1 means “yes” and 2 means “no”. Values as 97 and 99
are missing data.

-   sex: 1 for female and 2 for male.
-   age: of the patient.
-   classification: covid test findings. Values 1-3 mean that the
    patient was diagnosed with covid in different degrees. 4 or higher
    means that the patient is not a carrier of covid or that the test is
    inconclusive.
-   patient type: type of care the patient received in the unit. 1 for
    returned home and 2 for hospitalization.
-   pneumonia: whether the patient already have air sacs inflammation or
    not.
-   pregnancy: whether the patient is pregnant or not.
-   diabetes: whether the patient has diabetes or not.
-   copd: Indicates whether the patient has Chronic obstructive
    pulmonary disease or not.
-   asthma: whether the patient has asthma or not.
-   inmsupr: whether the patient is immunosuppressed or not.
-   hypertension: whether the patient has hypertension or not.
-   cardiovascular: whether the patient has heart or blood vessels
    related disease.
-   renal chronic: whether the patient has chronic renal disease or not.
-   other disease: whether the patient has other disease or not.
-   obesity: whether the patient is obese or not.
-   tobacco: whether the patient is a tobacco user.
-   usmr: Indicates whether the patient treated medical units of the
    first, second or third level.
-   medical unit: type of institution of the National Health System that
    provided the care.
-   intubed: whether the patient was connected to the ventilator.
-   icu: Indicates whether the patient had been admitted to an Intensive
    Care Unit.
-   date died: If the patient died indicate the date of death, and
    9999-99-99 otherwise.

## Data Cleaning and Exploratory Data Analysis

I will change `DATE_DIED` to a Boolean feature, replacing it with `DIED`
which indicates whether the patient died or not.

``` r
covid_data <- covid_data %>% 
  mutate(DATE_DIED = ifelse(DATE_DIED == "9999-99-99", 2, 1)) %>% rename(DIED = DATE_DIED)
```

``` r
covid_data %>% select(-AGE) %>% apply(2, table)
```

    ## $USMER
    ## 
    ##      1      2 
    ## 385672 662903 
    ## 
    ## $MEDICAL_UNIT
    ## 
    ##      1      2      3      4      5      6      7      8      9     10     11 
    ##    151    169  19175 314405   7244  40584    891  10399  38116   7873   5577 
    ##     12     13 
    ## 602995    996 
    ## 
    ## $SEX
    ## 
    ##      1      2 
    ## 525064 523511 
    ## 
    ## $PATIENT_TYPE
    ## 
    ##      1      2 
    ## 848544 200031 
    ## 
    ## $DIED
    ## 
    ##      1      2 
    ##  76942 971633 
    ## 
    ## $INTUBED
    ## 
    ##      1      2     97     99 
    ##  33656 159050 848544   7325 
    ## 
    ## $PNEUMONIA
    ## 
    ##      1      2     99 
    ## 140038 892534  16003 
    ## 
    ## $PREGNANT
    ## 
    ##      1      2     97     98 
    ##   8131 513179 523511   3754 
    ## 
    ## $DIABETES
    ## 
    ##      1      2     98 
    ## 124989 920248   3338 
    ## 
    ## $COPD
    ## 
    ##       1       2      98 
    ##   15062 1030510    3003 
    ## 
    ## $ASTHMA
    ## 
    ##       1       2      98 
    ##   31572 1014024    2979 
    ## 
    ## $INMSUPR
    ## 
    ##       1       2      98 
    ##   14170 1031001    3404 
    ## 
    ## $HIPERTENSION
    ## 
    ##      1      2     98 
    ## 162729 882742   3104 
    ## 
    ## $OTHER_DISEASE
    ## 
    ##       1       2      98 
    ##   28040 1015490    5045 
    ## 
    ## $CARDIOVASCULAR
    ## 
    ##       1       2      98 
    ##   20769 1024730    3076 
    ## 
    ## $OBESITY
    ## 
    ##      1      2     98 
    ## 159816 885727   3032 
    ## 
    ## $RENAL_CHRONIC
    ## 
    ##       1       2      98 
    ##   18904 1026665    3006 
    ## 
    ## $TOBACCO
    ## 
    ##      1      2     98 
    ##  84376 960979   3220 
    ## 
    ## $CLASIFFICATION_FINAL
    ## 
    ##      1      2      3      4      5      6      7 
    ##   8601   1851 381527   3122  26091 128133 499250 
    ## 
    ## $ICU
    ## 
    ##      1      2     97     99 
    ##  16858 175685 848544   7488

I notice that there are an equal number of male identified patients and
patients with missing pregnancy values denoted by 97. I assume male
identified patients are not pregnant and change those values to 2.

``` r
covid_data <- covid_data %>% 
  mutate(PREGNANT = ifelse(covid_data$SEX == 2, 2, covid_data$PREGNANT))
```

Now, I will replace all missing data with `NA` and change `2`s to `0` to
represent `FALSE`.

``` r
covid_data <- covid_data %>% 
  mutate_at(vars(-AGE), function(.){ifelse(. == 97|. == 98|. == 99, NA, .)})
covid_data <- covid_data %>% 
  mutate_at(vars(-USMER, -MEDICAL_UNIT, -AGE, -CLASIFFICATION_FINAL), function(.){ifelse(. == 2, 0, .)})
```

## Feature Selection

Dropping features with many missing values:

``` r
covid_data %>% apply(2, function(.){sum(is.na(.))/nrow(covid_data)})
```

    ##                USMER         MEDICAL_UNIT                  SEX 
    ##          0.000000000          0.000000000          0.000000000 
    ##         PATIENT_TYPE                 DIED              INTUBED 
    ##          0.000000000          0.000000000          0.816221062 
    ##            PNEUMONIA                  AGE             PREGNANT 
    ##          0.015261665          0.000000000          0.003580097 
    ##             DIABETES                 COPD               ASTHMA 
    ##          0.003183368          0.002863887          0.002840998 
    ##              INMSUPR         HIPERTENSION        OTHER_DISEASE 
    ##          0.003246310          0.002960208          0.004811292 
    ##       CARDIOVASCULAR              OBESITY        RENAL_CHRONIC 
    ##          0.002933505          0.002891543          0.002866748 
    ##              TOBACCO CLASIFFICATION_FINAL                  ICU 
    ##          0.003070834          0.000000000          0.816376511

``` r
covid_data <- covid_data %>% select(-INTUBED, -ICU)
```

``` r
fs <- data.frame(DIED = abs(t(cor(covid_data$DIED, covid_data, use = 'complete.obs'))), 
                 row.names = colnames(cor(covid_data$DIED, covid_data, use = 'complete.obs'))) %>% 
  arrange(DIED)
```

I will use the features that are significantly correlated with whether
or not the patient died. (I arbitrarily chose $\alpha = 0.1$ to limit
the number of features.)

``` r
features <- rownames(fs)[-(1:9)]
covid_data <- covid_data[features]
```

Normalizing Numeric Features

``` r
covid_data <- covid_data %>% mutate(AGE = (AGE-mean(AGE))/sd(AGE))
```

Changing Categorical Variables to Factors

``` r
covid_data <- covid_data %>% 
  mutate_at(vars(DIED, MEDICAL_UNIT, CLASIFFICATION_FINAL), as.factor)
```

Impute missing values

``` r
covid_data <- covid_data %>% 
  mutate_at(vars(RENAL_CHRONIC, HIPERTENSION, DIABETES, PNEUMONIA), 
            function(.){ifelse(is.na(.), mean(., na.rm = TRUE), .)})
```

Training and Testing Data Split

``` r
set.seed(2023)
n <- dim(covid_data)[1]
rows <- sample(1:n, 0.8*n)
train <- covid_data[rows,]
test <- covid_data[-rows,]
```

Logistic Regression

``` r
# Train a logistic regression model
covidmodel <- logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(DIED ~ ., data = train)

# Model summary
tidy(covidmodel)
```

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## Loaded glmnet 4.1-7

    ## # A tibble: 26 × 3
    ##    term          estimate penalty
    ##    <chr>            <dbl>   <dbl>
    ##  1 (Intercept)    -2.21         0
    ##  2 USMER          -0.128        0
    ##  3 RENAL_CHRONIC   0.442        0
    ##  4 MEDICAL_UNIT2   0.336        0
    ##  5 MEDICAL_UNIT3  -0.223        0
    ##  6 MEDICAL_UNIT4   0.404        0
    ##  7 MEDICAL_UNIT5  -0.0172       0
    ##  8 MEDICAL_UNIT6  -0.0871       0
    ##  9 MEDICAL_UNIT7  -0.770        0
    ## 10 MEDICAL_UNIT8  -0.544        0
    ## # ℹ 16 more rows

``` r
results <- predict(covidmodel, new_data = test, type = "class")
results_df <- data.frame(results, test["DIED"])
confusion <- table(results_df) #confusion matrix
f1 <- function(tp, fp, fn){
  tp/(tp+(fp+fn)/2)
}
f1_1 <- f1(sum(test["DIED"] == 1), confusion[2, 1], confusion[1, 2])
confusion
```

    ##            DIED
    ## .pred_class      0      1
    ##           0 190981   9613
    ##           1   3294   5827

An $F_1$ score of 0.7052321 shows room for improvement!

Undersampling to handle imbalanced dataset

I can see that this is an imbalanced dataset.

``` r
ggplot(data = covid_data, aes(fill = DIED)) +
  geom_bar(aes(x = DIED))+
  ggtitle("Number of samples in each class")+
  xlab("DIED")+
  ylab("Samples")+
  scale_y_continuous(expand = c(0,0))+
  scale_x_discrete(expand = c(0,0))+
  theme(legend.position = "none", 
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())
```

![](covid-19_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

I will use undersampling to handle the unequal distribution of high and
low risk patients.

``` r
class_counts <- table(covid_data["DIED"])
undersample_false <- covid_data %>% filter(DIED == 0) %>% slice_sample(n = class_counts[2])
undersample_true <- covid_data %>% filter(DIED == 1)
undersample <- rbind(undersample_false, undersample_true)
ggplot(data = undersample, aes(fill = DIED)) +
  geom_bar(aes(x = DIED))+
  ggtitle("Number of samples in each class after undersampling")+
  xlab("DIED")+
  ylab("Samples")+
  scale_y_continuous(expand = c(0,0))+
  scale_x_discrete(expand = c(0,0))+
  theme(legend.position = "none", 
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())
```

![](covid-19_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

Lets try again!

Training and Testing Data Split

``` r
set.seed(2023)
n2 <- dim(undersample)[1]
rows2 <- sample(1:n2, 0.8*n2)
train2 <- undersample[rows2,]
test2 <- undersample[-rows2,]
```

``` r
# Train a logistic regression model
covidmodel2 <- logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(DIED ~ ., data = train2)

# Model summary
tidy(covidmodel2)
```

    ## # A tibble: 26 × 3
    ##    term          estimate penalty
    ##    <chr>            <dbl>   <dbl>
    ##  1 (Intercept)     0.202        0
    ##  2 USMER          -0.176        0
    ##  3 RENAL_CHRONIC   0.512        0
    ##  4 MEDICAL_UNIT2   0.210        0
    ##  5 MEDICAL_UNIT3  -0.207        0
    ##  6 MEDICAL_UNIT4   0.399        0
    ##  7 MEDICAL_UNIT5   0.0707       0
    ##  8 MEDICAL_UNIT6  -0.0351       0
    ##  9 MEDICAL_UNIT7  -0.700        0
    ## 10 MEDICAL_UNIT8  -0.360        0
    ## # ℹ 16 more rows

``` r
results2 <- predict(covidmodel2, new_data = test2, type = "class")
results_df2 <- data.frame(results2, test2["DIED"])
confusion2 <- table(results_df2) #confusion matrix
f1_2 <- f1(sum(test["DIED"] == 1), confusion2[2, 1], confusion2[1, 2])
confusion2
```

    ##            DIED
    ## .pred_class     0     1
    ##           0 13734  1180
    ##           1  1663 14200

An $F_1$ score of 0.9156955 is much better!
