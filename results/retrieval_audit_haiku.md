# Retrieval quality audit

Skim for junk retrievals before eval runs. Firing mechanism for the similarity-threshold revisit trigger in docs/RESEARCH_DESIGN.md.

## spooky-author-identification

### Condition B (flat, single query)

### B: metadata

*query:* `# Overview ## Description As I scurried across the candlelit chamber, manuscripts in hand, I thought I'd made it. Nothing would be able to hurt me anymore. Litt`

*5 docs | similarity max=0.510 median=0.467 min=0.457*

- `0.510` **mlebench_text-normalization-challenge-english-language_1** (text-normalization-challenge-english-language) You are provided with a large corpus of text. Each sentence has a **sentence_id**. Each token within a sentence has a **token_id**. The **before** column contains the raw text, the
- `0.470` **code4ml_ykc-2nd_0** (ykc-2nd) '` multiclass metric : f1 micro NLP 7/719:00 7/1419:00 7/1419:00 private:public = 50:50 trainpublic, private submit20 2 word embedding `'
- `0.467` **mlebench_jigsaw-toxic-comment-classification-challenge_0** (jigsaw-toxic-comment-classification-challenge) # Overview ## Description Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up
- `0.464` **code4ml_text-regression-nnfl-lab-3_0** (text-regression-nnfl-lab-3) '`Regression Task to predict a value from [0,3] to determine the degree of offence in the sentences. The required output can be a Double value ranging from 0 to 3 -> [0,3]. Use nlp
- `0.457` **mlebench_text-normalization-challenge-russian-language_1** (text-normalization-challenge-russian-language) You are provided with a large corpus of text. Each sentence has a **sentence_id**. Each token within a sentence has a **token_id**. The **before** column contains the raw text, the

### B: notebook summaries

*query:* `# Overview ## Description As I scurried across the candlelit chamber, manuscripts in hand, I thought I'd made it. Nothing would be able to hurt me anymore. Litt`

*24 docs | similarity max=0.575 median=0.512 min=0.506*

- `0.575` **nb_2336315** (sona-speeches) This notebook addresses a multi-class text classification problem: identifying which of six South African presidents wrote a given speech based on textual features. The task involv
- `0.537` **nb_22831352** (feedback-prize-2021) This notebook addresses the task of classifying discourse types in student feedback essays using traditional text classification methods. The problem involves assigning one of seve
- `0.535` **nb_597830** (jigsaw-toxic-comment-classification-challenge) This notebook tackles a multi-label text classification problem on online comments, classifying them across six toxic categories: toxic, severe_toxic, obscene, threat, insult, and 
- `0.526` **nb_2279813** (sona-speeches) This notebook addresses the problem of classifying speeches by their author among six South African presidents: de Klerk, Mandela, Mbeki, Motlanthe, Zuma, and Ramaphosa. The approa
- `0.521` **nb_551084** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label toxicity classification on comment text, where each comment must be independently classified across six categories: toxic, severe_toxic, obscene
- `0.518` **nb_4563839** (edsa-mbti) This notebook tackles multiclass text classification to predict Myers-Briggs personality types from user forum posts. The dataset contains 6506 training samples split across 16 per
- `0.515` **nb_678620** (jigsaw-toxic-comment-classification-challenge) This notebook tackles multi-label toxicity classification on comment text using a Naive Bayes feature weighting strategy combined with logistic regression. The dataset contains app
- `0.513` **nb_4474914** (edsa-mbti) This notebook tackles Myers-Briggs personality type prediction from text posts. The problem decomposes the 16 personality types into four independent binary classification tasks (I
- `0.513` **nb_2391012** (jigsaw-toxic-comment-classification-challenge) This notebook tackles toxic comment classification, a multi-label text classification problem where comments must be tagged across six categories: toxic, severe toxic, obscene, thr
- `0.512` **nb_4498126** (edsa-mbti) This notebook tackles Myers-Briggs personality type classification from social media posts, decomposing a 16-class multi-label problem into four independent binary classification t
- `0.512` **nb_2300805** (sona-speeches) This notebook tackles multiclass text classification for attributing presidential speeches to one of six South African presidents (de Klerk, Mandela, Mbeki, Motlanthe, Zuma, and Ra
- `0.512` **nb_716125** (jigsaw-toxic-comment-classification-challenge) This notebook addresses toxic comment classification across six categories (toxic, severe_toxic, obscene, threat, insult, identity_hate) on a dataset of roughly 160k training and 1
- `0.512` **nb_499816** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on online comments, where each comment can simultaneously belong to multiple toxicity categories: toxic, severe_toxic, obsce
- `0.512` **nb_605036** (jigsaw-toxic-comment-classification-challenge) This notebook tackles toxic comment classification across six multi-label categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The author's approach emphasi
- `0.510` **nb_729600** (jigsaw-toxic-comment-classification-challenge) This notebook tackles multi-label toxic comment classification where each of six toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate) must be independe
- `0.508` **nb_14432420** (donorschoose-application-screening) This notebook addresses binary classification of school fundraising project applications—predicting whether a DonorsChoose project will be approved based on project metadata, text 
- `0.507` **nb_6517333** (jigsaw-toxic-comment-classification-challenge) This notebook tackles a multi-label text classification problem on toxic comment detection. The dataset contains comments labeled across six toxicity types (toxic, severe_toxic, ob
- `0.507` **nb_13797424** (jigsaw-toxic-comment-classification-challenge) This notebook tackles multi-label text classification on the Jigsaw toxic comment challenge, where individual comments must be simultaneously classified across six toxicity categor
- `0.507` **nb_4563028** (edsa-mbti) This notebook tackles a multi-label text classification problem on MBTI personality type prediction from written posts. The target consists of four independent binary dimensions (I
- `0.507` **nb_979935** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label toxicity classification of user comments, where each comment can belong to zero or more of six toxic categories: toxic, severe_toxic, obscene, t
- `0.506` **nb_9801178** (jigsaw-multilingual-toxic-comment-classification) This notebook tackles toxic comment classification on a multilingual dataset by combining Naive Bayes feature weighting with logistic regression. The author works with roughly 500,
- `0.506` **nb_4490621** (edsa-mbti) This notebook addresses personality type classification from social media posts, treating it as four independent binary classification problems rather than a single 16-class multic
- `0.506` **nb_705116** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on toxic comment detection, where each comment can belong to zero or more of six categories: toxic, severe_toxic, obscene, t
- `0.506` **nb_1224603** (movie-review-sentiment-analysis-kernels-only) This notebook addresses sentiment classification of movie review phrases, predicting one of five sentiment classes (0-4) from text. The author employs a two-stage stacking approach

### Condition C (staged, directed queries)

### C: parse (metadata)

*query:* `# Overview ## Description As I scurried across the candlelit chamber, manuscripts in hand, I thought I'd made it. Nothing would be able to hurt me anymore. Litt`

*5 docs | similarity max=0.510 median=0.467 min=0.457*

- `0.510` **mlebench_text-normalization-challenge-english-language_1** (text-normalization-challenge-english-language) You are provided with a large corpus of text. Each sentence has a **sentence_id**. Each token within a sentence has a **token_id**. The **before** column contains the raw text, the
- `0.470` **code4ml_ykc-2nd_0** (ykc-2nd) '` multiclass metric : f1 micro NLP 7/719:00 7/1419:00 7/1419:00 private:public = 50:50 trainpublic, private submit20 2 word embedding `'
- `0.467` **mlebench_jigsaw-toxic-comment-classification-challenge_0** (jigsaw-toxic-comment-classification-challenge) # Overview ## Description Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up
- `0.464` **code4ml_text-regression-nnfl-lab-3_0** (text-regression-nnfl-lab-3) '`Regression Task to predict a value from [0,3] to determine the degree of offence in the sentences. The required output can be a Double value ranging from 0 to 3 -> [0,3]. Use nlp
- `0.457` **mlebench_text-normalization-challenge-russian-language_1** (text-normalization-challenge-russian-language) You are provided with a large corpus of text. Each sentence has a **sentence_id**. Each token within a sentence has a **token_id**. The **before** column contains the raw text, the

### C: surface

*query:* `Multi-class text classification Multi-class logarithmic loss (log loss), with probability rescaling per row Predict the author of text excerpts from horror lite`

*8 docs | similarity max=0.598 median=0.542 min=0.537*

- `0.598` **nb_2336315** (sona-speeches) This notebook addresses a multi-class text classification problem: identifying which of six South African presidents wrote a given speech based on textual features. The task involv
- `0.546` **nb_4563028** (edsa-mbti) This notebook tackles a multi-label text classification problem on MBTI personality type prediction from written posts. The target consists of four independent binary dimensions (I
- `0.544` **nb_597830** (jigsaw-toxic-comment-classification-challenge) This notebook tackles a multi-label text classification problem on online comments, classifying them across six toxic categories: toxic, severe_toxic, obscene, threat, insult, and 
- `0.543` **nb_4563839** (edsa-mbti) This notebook tackles multiclass text classification to predict Myers-Briggs personality types from user forum posts. The dataset contains 6506 training samples split across 16 per
- `0.540` **nb_4490621** (edsa-mbti) This notebook addresses personality type classification from social media posts, treating it as four independent binary classification problems rather than a single 16-class multic
- `0.539` **nb_4474914** (edsa-mbti) This notebook tackles Myers-Briggs personality type prediction from text posts. The problem decomposes the 16 personality types into four independent binary classification tasks (I
- `0.539` **nb_499816** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on online comments, where each comment can simultaneously belong to multiple toxicity categories: toxic, severe_toxic, obsce
- `0.537` **nb_4564645** (edsa-mbti) This notebook addresses multi-label personality type classification on the Myers-Briggs Type Indicator dataset, decomposing a 16-class problem into four independent binary classifi

### C: flag

*query:* `validation leakage overfitting pitfalls Multi-class text classification Multi-class logarithmic loss (log loss), with probability rescaling per row Predict the `

*13 docs | similarity max=0.574 median=0.558 min=0.550*

- `0.574` **nb_499816** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on online comments, where each comment can simultaneously belong to multiple toxicity categories: toxic, severe_toxic, obsce
- `0.573` **nb_2336315** (sona-speeches) This notebook addresses a multi-class text classification problem: identifying which of six South African presidents wrote a given speech based on textual features. The task involv
- `0.571` **nb_551084** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label toxicity classification on comment text, where each comment must be independently classified across six categories: toxic, severe_toxic, obscene
- `0.567` **nb_4563028** (edsa-mbti) This notebook tackles a multi-label text classification problem on MBTI personality type prediction from written posts. The target consists of four independent binary dimensions (I
- `0.564` **nb_2240051** (quora-insincere-questions-classification) This notebook addresses text classification on a question dataset using a straightforward bag-of-words approach paired with logistic regression. The problem involves binary or mult
- `0.560` **nb_597830** (jigsaw-toxic-comment-classification-challenge) This notebook tackles a multi-label text classification problem on online comments, classifying them across six toxic categories: toxic, severe_toxic, obscene, threat, insult, and 
- `0.558` **nb_4564645** (edsa-mbti) This notebook addresses multi-label personality type classification on the Myers-Briggs Type Indicator dataset, decomposing a 16-class problem into four independent binary classifi
- `0.557` **nb_3483360** (bad-comments) This notebook tackles multi-label text classification for toxic comment detection, where each document can belong to multiple categories simultaneously: toxic, severe_toxic, obscen
- `0.556` **nb_4068705** (jigsaw-toxic-comment-classification-challenge) This notebook tackles multi-label text classification on toxic comment detection, where individual comments can be tagged with zero or more of six violation categories: toxic, seve
- `0.556` **nb_2016588** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on the Jigsaw toxic comment dataset, where comments must be simultaneously classified across six toxicity categories: toxic,
- `0.552` **nb_605470** (jigsaw-toxic-comment-classification-challenge) This notebook addresses a multilabel text classification task on toxic comment detection, where the goal is to predict six binary toxicity categories simultaneously across test com
- `0.552` **nb_6206311** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label toxic comment classification, where the goal is to predict six independent toxicity categories (toxic, severe_toxic, obscene, threat, insult, id
- `0.550` **nb_3907124** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on a corpus of online comments, identifying six types of toxic content: toxic, severe_toxic, obscene, threat, insult, and id

### C: advise

*query:* `model architecture training approach Multi-class text classification Multi-class logarithmic loss (log loss), with probability rescaling per row Predict the aut`

*12 docs | similarity max=0.603 median=0.587 min=0.584*

- `0.603` **nb_2360909** (quora-insincere-questions-classification) This notebook tackles a multi-category text classification problem using a hierarchical attention mechanism that processes text at two levels: word-level and sentence-level. The wo
- `0.603` **nb_2336315** (sona-speeches) This notebook addresses a multi-class text classification problem: identifying which of six South African presidents wrote a given speech based on textual features. The task involv
- `0.594` **nb_6739628** (google-quest-challenge) This notebook addresses a multi-label text classification problem from the Google Quest Challenge, predicting 30 binary labels for question-answer quality attributes. The work comb
- `0.591` **nb_4563028** (edsa-mbti) This notebook tackles a multi-label text classification problem on MBTI personality type prediction from written posts. The target consists of four independent binary dimensions (I
- `0.588` **nb_4490621** (edsa-mbti) This notebook addresses personality type classification from social media posts, treating it as four independent binary classification problems rather than a single 16-class multic
- `0.588` **nb_499816** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on online comments, where each comment can simultaneously belong to multiple toxicity categories: toxic, severe_toxic, obsce
- `0.585` **nb_4900265** (jigsaw-toxic-comment-classification-challenge) This notebook tackles multi-label text classification on Wikipedia comments, predicting six toxicity categories simultaneously. The author combined TF-IDF feature extraction with a
- `0.585` **nb_2193907** (quora-insincere-questions-classification) This notebook tackles binary classification of insincere questions in Quora text data, approaching it through multi-level sequence processing with a custom hierarchical architectur
- `0.585` **nb_9774054** (jigsaw-toxic-comment-classification-challenge) This notebook addresses multi-label text classification on the Jigsaw toxic comment dataset, which contains approximately 160,000 comments labeled across six toxicity categories (t
- `0.584` **nb_2672548** (quora-insincere-questions-classification) This notebook addresses binary text classification on a corpus of questions, likely detecting a specific problematic category (such as insincere or offensive questions). The author
- `0.584` **nb_2309757** (quora-insincere-questions-classification) This notebook addresses a binary text classification problem on questions, using multi-task learning with gated recurrent architectures and pre-trained embeddings. The task involve
- `0.584` **nb_987160** (donorschoose-application-screening) The notebook builds a classification model to predict whether projects receive funding approval using TensorFlow's Estimator API, working with structured data that combines project

*distinct summaries across C's stages: 24 (parity target 24)*

## nomad2018-predict-transparent-conductors

### Condition B (flat, single query)

### B: metadata

*query:* `# Overview ## Description Innovative materials design is needed to tackle some of the most important health, environmental, energy, social, and economic challen`

*5 docs | similarity max=0.479 median=0.478 min=0.474*

- `0.479` **code4ml_facial-emotion-recognition_0** (facial-emotion-recognition) '` , . , submission-. Submission- - .csv - id . jupyter- . -, 4848. 28709 , - 7178. . , submission-, . , baseline. , , strong-baseline , `'
- `0.479` **code4ml_hotel-reviews-classification_0** (hotel-reviews-classification) '` train.csv . , . test.csv, sampleSubmission.csv. submit .`'
- `0.478` **code4ml_exam-for-students20200129_0** (exam-for-students20200129) '`AI Academy SubmissionKernel (Python or R) csvKernelnotebooksubmit KernelPublic Kernel Rules EvaluationRMSLE `'
- `0.475` **code4ml_who-is-rich?_0** (who-is-rich?) '`Please view the associated notebook.`'
- `0.474` **code4ml_thousand-facial-landmarks_0** (thousand-facial-landmarks) '` 971 . , , - , , . : https://github.com/BorisLestsov/MADE/tree/master/contest1/unsupervised-landmarks-thousand-landmarks-contest : -5 (50). 6 , , 49 1 ; . , , , 1 . 0. , ( Notebo

### B: notebook summaries

*query:* `# Overview ## Description Innovative materials design is needed to tackle some of the most important health, environmental, energy, social, and economic challen`

*24 docs | similarity max=0.525 median=0.506 min=0.495*

- `0.525` **nb_4106253** (champs-scalar-coupling) This notebook predicts scalar coupling constants in molecular structures, a fundamental quantity in computational chemistry. The problem combines geometric and chemical information
- `0.524` **nb_5507194** (champs-scalar-coupling) This notebook addresses molecular property prediction specifically: predicting scalar coupling constants between atom pairs in molecules given their 3D structures. This is a regres
- `0.517` **nb_4079065** (champs-scalar-coupling) This notebook addresses the molecular property prediction task of estimating scalar coupling constants between pairs of atoms in molecules. The core challenge is predicting a conti
- `0.514` **nb_4423628** (champs-scalar-coupling) This notebook addresses the problem of predicting scalar coupling constants between atom pairs in molecules, a quantum chemistry regression task. The author combined domain-specifi
- `0.513` **nb_4539091** (champs-scalar-coupling) This notebook addresses the prediction of scalar coupling constants in molecules from structural data, treating it as a regression problem on molecular geometry and composition fea
- `0.511` **nb_4081339** (champs-scalar-coupling) This notebook tackles predicting scalar coupling constants in molecules from structural and quantum chemical data. The core problem is a regression task: given atomic coordinates a
- `0.510` **nb_4077747** (champs-scalar-coupling) This notebook addresses predicting scalar coupling constants in molecular structures, a problem that requires mapping atomic coordinate data and learning a regression model from st
- `0.510` **nb_4818806** (champs-scalar-coupling) This notebook addresses predicting scalar coupling constants in molecules from structural and electronic properties. The problem involves merging multiple data sources—training lab
- `0.509` **nb_4983787** (champs-scalar-coupling) This notebook tackles predicting scalar coupling constants in molecular structures by engineering a comprehensive feature set from atomic coordinates and chemical properties. The p
- `0.509` **nb_4076932** (champs-scalar-coupling) This notebook addresses the problem of predicting scalar coupling constants between atom pairs in molecules given their 3D structural coordinates. The dataset contains molecular st
- `0.506` **nb_4108882** (champs-scalar-coupling) This notebook tackles the molecular property prediction problem of estimating scalar coupling constants from atomic structure data. The problem provides pairs of atoms within molec
- `0.506` **nb_4093067** (champs-scalar-coupling) This notebook addresses the prediction of scalar coupling constants in molecules by merging structural coordinates with coupling measurements and engineering distance-based feature
- `0.506` **nb_4281736** (champs-scalar-coupling) This notebook addresses prediction of scalar coupling constants in molecules—a quantum chemistry property essential for molecular characterization. The author took a feature-engine
- `0.505` **nb_5683529** (champs-scalar-coupling) This notebook addresses the molecular property prediction problem of estimating scalar coupling constants from atomic structure data using gradient boosting. The dataset comprises 
- `0.505` **nb_4086752** (champs-scalar-coupling) This notebook addresses a molecular property prediction problem: predicting scalar coupling constants from atomic structures. The dataset contains training examples linking pairs o
- `0.505` **nb_4076991** (champs-scalar-coupling) This notebook tackles prediction of scalar coupling constants in molecular systems—a quantum chemistry problem where the target is a continuous coupling strength value between atom
- `0.504` **nb_5334485** (champs-scalar-coupling) This notebook tackles predicting scalar coupling constants in molecules using structural data, framing the problem as a regression task on quantum chemistry features. The dataset c
- `0.503` **nb_4089918** (champs-scalar-coupling) This notebook addresses molecular property prediction for scalar coupling constants between atom pairs, a physics-based regression problem derived from quantum chemistry. The autho
- `0.500` **nb_4477558** (champs-scalar-coupling) This notebook tackles a regression problem predicting scalar coupling constants in molecules, a quantum chemistry property. The dataset contains training records indexed by molecul
- `0.500` **nb_4080021** (champs-scalar-coupling) This notebook tackles molecular property prediction—specifically predicting scalar coupling constants from atomic structures—using a structured machine learning pipeline centered o
- `0.500` **nb_5284104** (champs-scalar-coupling) This notebook addresses the prediction of scalar coupling constants in molecular structures, a quantum chemistry regression task that combines structural geometry with supplementar
- `0.499` **nb_4184451** (champs-scalar-coupling) This notebook tackles the CHAMPS scalar coupling prediction problem, where the goal is to predict quantum mechanical scalar coupling constants between atom pairs in molecules using
- `0.497` **nb_4441781** (champs-scalar-coupling) This notebook addresses the problem of predicting scalar coupling constants between atoms in molecules, using a dataset containing molecular structure information indexed by coupli
- `0.495` **nb_4082794** (champs-scalar-coupling) This notebook addresses a molecular property prediction problem, specifically predicting scalar coupling constants between atom pairs in molecules. The dataset contains molecular s

### Condition C (staged, directed queries)

### C: parse (metadata)

*query:* `# Overview ## Description Innovative materials design is needed to tackle some of the most important health, environmental, energy, social, and economic challen`

*5 docs | similarity max=0.479 median=0.478 min=0.474*

- `0.479` **code4ml_facial-emotion-recognition_0** (facial-emotion-recognition) '` , . , submission-. Submission- - .csv - id . jupyter- . -, 4848. 28709 , - 7178. . , submission-, . , baseline. , , strong-baseline , `'
- `0.479` **code4ml_hotel-reviews-classification_0** (hotel-reviews-classification) '` train.csv . , . test.csv, sampleSubmission.csv. submit .`'
- `0.478` **code4ml_exam-for-students20200129_0** (exam-for-students20200129) '`AI Academy SubmissionKernel (Python or R) csvKernelnotebooksubmit KernelPublic Kernel Rules EvaluationRMSLE `'
- `0.475` **code4ml_who-is-rich?_0** (who-is-rich?) '`Please view the associated notebook.`'
- `0.474` **code4ml_thousand-facial-landmarks_0** (thousand-facial-landmarks) '` 971 . , , - , , . : https://github.com/BorisLestsov/MADE/tree/master/contest1/unsupervised-landmarks-thousand-landmarks-contest : -5 (50). 6 , , 49 1 ; . , , , 1 . 0. , ( Notebo

### C: surface

*query:* `multi-target regression Root Mean Squared Logarithmic Error (RMSLE), column-wise averaged Predict formation energy and bandgap energy for novel transparent cond`

*8 docs | similarity max=0.532 median=0.515 min=0.509*

- `0.532` **nb_18402214** (tabular-playground-series-jul-2021) This notebook demonstrates building an ensemble-based regression solution for air quality pollutant prediction using LightAutoML on a multi-target tabular dataset. The problem invo
- `0.521` **nb_18452923** (tabular-playground-series-jul-2021) This notebook addresses a multi-output regression problem predicting three correlated air quality targets (carbon monoxide, benzene, nitrogen oxides) from sensor data. The distinct
- `0.518` **nb_18964729** (tabular-playground-series-jul-2021) This notebook tackles a multi-target air quality prediction problem with three continuous regression targets: carbon monoxide, benzene, and nitrogen oxides. The core challenge was 
- `0.516` **nb_18368860** (tabular-playground-series-jul-2021) This notebook addresses a multi-output air quality regression problem predicting three pollutants—carbon monoxide, benzene, and nitrogen dioxide—from sensor and meteorological feat
- `0.515` **nb_9533389** (trends-assessment-prediction) This notebook addresses multi-target regression on neuroimaging data (brain functional connectivity and component loadings) to predict five continuous outcomes: age and four domain
- `0.514` **nb_18261372** (tabular-playground-series-jul-2021) This notebook addresses a multi-target air-quality prediction problem where three pollutant concentrations (carbon monoxide, benzene, nitrogen oxides) must be predicted from sensor
- `0.511` **nb_18396582** (tabular-playground-series-jul-2021) This notebook addresses multi-target regression for air quality prediction, forecasting three pollutant concentrations (carbon monoxide, benzene, nitrogen oxides) from environmenta
- `0.509` **nb_18506994** (tabular-playground-series-jul-2021) This notebook addresses multioutput regression for air quality prediction, where the task is to predict three pollutant concentrations—carbon monoxide, benzene, and nitrogen oxides

### C: flag

*query:* `validation leakage overfitting pitfalls multi-target regression Root Mean Squared Logarithmic Error (RMSLE), column-wise averaged Predict formation energy and b`

*10 docs | similarity max=0.565 median=0.538 min=0.517*

- `0.565` **nb_1480310** (santander-value-prediction-challenge) This notebook tackles a regression value prediction problem by combining feature leakage detection with model stacking and ensemble blending. The author faces a dataset where certa
- `0.548` **nb_1312489** (santander-value-prediction-challenge) This notebook addresses a tabular regression problem where the target variable is partially recoverable from the feature set through data leakage. Rather than building a predictive
- `0.547` **nb_1478336** (santander-value-prediction-challenge) This notebook addresses a regression prediction task on the Santander value dataset by combining data leakage detection with heavy ensemble averaging. The author's core insight is 
- `0.545` **nb_1489321** (santander-value-prediction-challenge) This notebook tackles a regression prediction problem on tabular data where a data leak—test samples that appear in training data under different IDs—constitutes a major component 
- `0.538` **nb_12214766** (riiid-test-answer-prediction) This notebook addresses predicting student answer correctness in the RIIID educational platform, where the main challenge is avoiding data leakage when creating target-encoded feat
- `0.538` **nb_18261372** (tabular-playground-series-jul-2021) This notebook addresses a multi-target air-quality prediction problem where three pollutant concentrations (carbon monoxide, benzene, nitrogen oxides) must be predicted from sensor
- `0.526` **nb_18747834** (tabular-playground-series-jul-2021) This notebook addresses the problem of predicting air quality measurements for carbon monoxide, benzene, and nitrogen oxides given meteorological and sensor features. The author di
- `0.525` **nb_18506994** (tabular-playground-series-jul-2021) This notebook addresses multioutput regression for air quality prediction, where the task is to predict three pollutant concentrations—carbon monoxide, benzene, and nitrogen oxides
- `0.518` **nb_1276318** (santander-value-prediction-challenge) The notebook tackles a regression problem on the Santander value prediction dataset by building a two-stage pipeline that combines individual feature screening, leak signal exploit
- `0.517` **nb_1345525** (santander-value-prediction-challenge) This notebook addresses a regression problem on the Santander value prediction dataset where the target variable exhibits a strong data leak—portions of the feature space contain e

### C: advise

*query:* `model architecture training approach multi-target regression Root Mean Squared Logarithmic Error (RMSLE), column-wise averaged Predict formation energy and band`

*13 docs | similarity max=0.591 median=0.550 min=0.543*

- `0.591` **nb_18964729** (tabular-playground-series-jul-2021) This notebook tackles a multi-target air quality prediction problem with three continuous regression targets: carbon monoxide, benzene, and nitrogen oxides. The core challenge was 
- `0.573` **nb_18452923** (tabular-playground-series-jul-2021) This notebook addresses a multi-output regression problem predicting three correlated air quality targets (carbon monoxide, benzene, nitrogen oxides) from sensor data. The distinct
- `0.561` **nb_18402214** (tabular-playground-series-jul-2021) This notebook demonstrates building an ensemble-based regression solution for air quality pollutant prediction using LightAutoML on a multi-target tabular dataset. The problem invo
- `0.560` **nb_9533389** (trends-assessment-prediction) This notebook addresses multi-target regression on neuroimaging data (brain functional connectivity and component loadings) to predict five continuous outcomes: age and four domain
- `0.559` **nb_11687179** (stanford-covid-vaccine) The notebook tackles a multi-output regression problem predicting RNA structural properties—specifically reactivity and degradation rates at two conditions (pH10 and 50C)—from enco
- `0.554` **nb_9871943** (trends-assessment-prediction) This notebook addresses multi-target regression on functional connectivity and loading data, predicting five continuous outcomes (age and four domain variables) with unequal loss w
- `0.550` **nb_12288625** (house-prices-advanced-regression-techniques) This notebook addresses housing price regression by training a complex multi-task neural network that generates multiple predictions and learns to weight them. The author worked wi
- `0.548` **nb_18261372** (tabular-playground-series-jul-2021) This notebook addresses a multi-target air-quality prediction problem where three pollutant concentrations (carbon monoxide, benzene, nitrogen oxides) must be predicted from sensor
- `0.547` **nb_9953638** (trends-assessment-prediction) This notebook addresses multi-target regression on neuroimaging data, combining functional network connectivity (FNC) features with loading features to predict five distinct target
- `0.547` **nb_18288993** (tabular-playground-series-jul-2021) This notebook addresses a multioutput regression problem predicting three air quality pollutants (carbon monoxide, benzene, and nitrogen oxides) from meteorological and sensor feat
- `0.546` **nb_9084991** (trends-assessment-prediction) This notebook tackles a multi-output regression problem predicting five continuous behavioral assessment scores from two heterogeneous neural data sources: functional connectivity 
- `0.544` **nb_11535683** (lish-moa) This notebook addresses the mechanism of action prediction problem for drug compounds, where the task is to predict multiple binary labels (206 scored targets) from molecular featu
- `0.543` **nb_9098292** (trends-assessment-prediction) This notebook tackles a multi-target regression problem predicting five continuous outcomes (age, domain1_var1, domain1_var2, domain2_var1, domain2_var2) from functional network co

*distinct summaries across C's stages: 24 (parity target 24)*

## dogs-vs-cats-redux-kernels-edition

### Condition B (flat, single query)

### B: metadata

*query:* `# Overview ## Overview ### Description In 2013, we hosted one of our favorite for-fun competitions: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats). Much`

*5 docs | similarity max=0.657 median=0.595 min=0.568*

- `0.657` **code4ml_dogs-vs.-cats-redux:-kernels-edition_0** (dogs-vs.-cats-redux:-kernels-edition) '`In 2013, we hosted one of our favorite for-fun competitions: Dogs vs. Cats. Much has since changed in the machine learning landscape, particularly in deep learning and image anal
- `0.619` **mlebench_dog-breed-identification_0** (dog-breed-identification) # Overview ## Description Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have *all* the answers. However, maybe they can answer th
- `0.595` **code4ml_atml2020-assignment-2_0** (atml2020-assignment-2) '`ImageNet is a well known dataset with 1000 image classes. We will be working on a subset of the dataset (60k images, 100 classes, 600 images per class 8080 pixels, RGB) and trai
- `0.569` **mlebench_aptos2019-blindness-detection_2** (aptos2019-blindness-detection) ![public vs private](https://storage.googleapis.com/kaggle-media/competitions/general/public_vs_private.png) - **train.csv** - the training labels - **test.csv** - the test set (yo
- `0.568` **code4ml_transferlearning-competition-appliedai_0** (transferlearning-competition-appliedai) '`Use transfer learning to properly predict flowers. See demo kernel for a starting point. Acknowledgements Dataset provided by Kaggle.`'

### B: notebook summaries

*query:* `# Overview ## Overview ### Description In 2013, we hosted one of our favorite for-fun competitions: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats). Much`

*24 docs | similarity max=0.594 median=0.577 min=0.573*

- `0.594` **nb_4722040** (aerial-cactus-identification) This notebook tackles a binary image classification task: detecting the presence of cacti in photographs. The dataset consists of training images with binary labels and unlabeled t
- `0.593` **nb_3132650** (petfinder-adoption-prediction) The notebook builds a convolutional neural network to predict adoption speed categories from pet photos in a multi-class classification task with five outcome classes. The core pro
- `0.592` **nb_18927591** (dog-breed-identification) This notebook builds a convolutional neural network to classify dog breeds from images. The dataset contains labeled training images across 120 dog breed classes and unlabeled test
- `0.587` **nb_4141242** (aerial-cactus-identification) This notebook addresses a binary image classification problem: identifying whether images contain cacti. The author built a custom convolutional neural network to solve this, treat
- `0.587` **nb_17260183** (dog-breed-identification) This notebook addresses dog breed classification from images using a custom convolutional neural network trained on labeled image data. The core problem is multi-class image classi
- `0.584` **nb_15105221** (dog-breed-identification) This notebook tackles a dog breed classification problem with 120 breeds using convolutional neural networks on image data. The author approaches it as a multiclass image classific
- `0.583` **nb_13773639** (dog-breed-identification) This notebook addresses multiclass image classification for dog breed identification using a custom convolutional neural network built with Keras. The dataset contains labeled trai
- `0.580` **nb_3986787** (aerial-cactus-identification) This notebook tackles a binary image classification task: determining whether aerial photographs contain cacti. The dataset consists of 32x32 pixel RGB images with binary labels, s
- `0.579` **nb_5211968** (aerial-cactus-identification) This notebook addresses binary image classification on the Aerial Cactus Identification dataset, where the task is to detect whether 32x32 pixel aerial images contain cacti. The au
- `0.578` **nb_4173015** (aerial-cactus-identification) This notebook addresses binary image classification on a cactus detection dataset containing 17,500 training images and a separate test set. The task is to predict whether each ima
- `0.578` **nb_12903455** (aerial-cactus-identification) This notebook addresses binary image classification of aerial photographs to detect cacti, achieving predictions on a held-out test set. The dataset consists of 32x32 RGB images th
- `0.577` **nb_4405683** (aerial-cactus-identification) This notebook addresses binary image classification on a cactus detection dataset, where the task is to predict whether photographs contain cacti. The author built a custom convolu
- `0.577` **nb_3985517** (aerial-cactus-identification) This notebook builds a binary image classifier to detect the presence of cacti in 32x32 pixel photographs. The problem is a straightforward supervised learning task where the train
- `0.576` **nb_4469558** (histopathologic-cancer-detection) This notebook addresses a binary image classification problem on a dataset of 96x96 pixel images, building and training a custom convolutional neural network from scratch. The auth
- `0.576` **nb_4781335** (aerial-cactus-identification) This notebook tackles binary image classification on a cactus detection dataset using a straightforward convolutional neural network built in Keras. The core task is to predict whe
- `0.576` **nb_3437580** (aerial-cactus-identification) This notebook addresses binary image classification on a dataset of photographs to detect the presence of cacti. The author starts with a simple fully-connected baseline model, the
- `0.575` **nb_13965434** (dog-breed-identification) This notebook tackles the dog breed identification problem using transfer learning with a pre-trained MobileNetV2 model from TensorFlow Hub. The author took a pragmatic experimenta
- `0.575` **nb_21575021** (petfinder-pawpularity-score) This notebook tackles the PetFinder pawpularity prediction task, which requires predicting a popularity score for pet photos based on image content. The author's approach frames th
- `0.575` **nb_22503151** (petfinder-pawpularity-score) This notebook addresses the problem of predicting a photo's appeal score (Pawpularity) from images of pets, framing it as a regression task rather than classification. The dataset 
- `0.574` **nb_20911790** (petfinder-pawpularity-score) This notebook addresses the PetFinder Pawpularity prediction problem, which asks to predict a popularity score (ranging from 1 to 100) for pet images. The author treats this as a 1
- `0.574` **nb_3326181** (petfinder-adoption-prediction) This notebook addresses pet adoption speed prediction by building a convolutional neural network for image classification. The problem is multiclass (five adoption speed categories
- `0.574` **nb_14229079** (aerial-cactus-identification) This notebook addresses binary image classification on aerial photographs to detect the presence of cacti. The problem involves classifying 32x32 pixel RGB images as containing cac
- `0.574` **nb_16671792** (dog-breed-identification) This notebook addresses multi-class image classification on the dog breed identification dataset, which contains 10,222 training images across 120 dog breeds and requires predictin
- `0.573` **nb_22085444** (petfinder-pawpularity-score) This notebook tackles the PetFinder Pawpularity problem, a regression task to predict a pet photograph's engagement score (0-100) from image data alone. The author built and traine

### Condition C (staged, directed queries)

### C: parse (metadata)

*query:* `# Overview ## Overview ### Description In 2013, we hosted one of our favorite for-fun competitions: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats). Much`

*5 docs | similarity max=0.657 median=0.595 min=0.568*

- `0.657` **code4ml_dogs-vs.-cats-redux:-kernels-edition_0** (dogs-vs.-cats-redux:-kernels-edition) '`In 2013, we hosted one of our favorite for-fun competitions: Dogs vs. Cats. Much has since changed in the machine learning landscape, particularly in deep learning and image anal
- `0.619` **mlebench_dog-breed-identification_0** (dog-breed-identification) # Overview ## Description Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have *all* the answers. However, maybe they can answer th
- `0.595` **code4ml_atml2020-assignment-2_0** (atml2020-assignment-2) '`ImageNet is a well known dataset with 1000 image classes. We will be working on a subset of the dataset (60k images, 100 classes, 600 images per class 8080 pixels, RGB) and trai
- `0.569` **mlebench_aptos2019-blindness-detection_2** (aptos2019-blindness-detection) ![public vs private](https://storage.googleapis.com/kaggle-media/competitions/general/public_vs_private.png) - **train.csv** - the training labels - **test.csv** - the test set (yo
- `0.568` **code4ml_transferlearning-competition-appliedai_0** (transferlearning-competition-appliedai) '`Use transfer learning to properly predict flowers. See demo kernel for a starting point. Acknowledgements Dataset provided by Kaggle.`'

### C: surface

*query:* `binary image classification log loss (binary cross-entropy) Classify images as dogs or cats and predict the probability that each test image is a dog`

*8 docs | similarity max=0.608 median=0.596 min=0.585*

- `0.608` **nb_9994824** (padhai-module1-level2) This notebook addresses binary image classification on a multilingual document dataset, distinguishing between background images and documents (in Tamil, Hindi, or English). The co
- `0.603` **nb_3219666** (padhai-module1-level4b) This notebook addresses a binary image classification task distinguishing background images from script text images (Tamil, Hindi, and English) using a sigmoid neuron trained with 
- `0.599` **nb_3204918** (padhai-module1-level4b) This notebook tackles a binary image classification task distinguishing text images (Tamil, Hindi, English) from background images using a simple sigmoid neuron trained with stocha
- `0.598` **nb_3229249** (padhai-module1-level4b) This notebook addresses a binary image classification task distinguishing text-containing images from background images across three language datasets (Tamil, Hindi, English). The 
- `0.593` **nb_3231902** (padhai-module1-level4a) This notebook addresses binary image classification on a dataset of multilingual text and background images, building a sigmoid neuron classifier from scratch to distinguish betwee
- `0.592` **nb_3284527** (padhai-module1-level4b) This notebook addresses binary classification of document images, specifically distinguishing background images from text-bearing documents (Tamil, Hindi, and English). The author 
- `0.591` **nb_3294704** (padhai-module1-level4b) This notebook addresses a binary image classification problem where the task is to distinguish text characters (in English, Tamil, and Hindi) from background images. The dataset co
- `0.585` **nb_3599227** (aerial-cactus-identification) This notebook addresses a binary image classification task: detecting the presence of cacti in photographs. The dataset comprises labeled training images and unlabeled test images,

### C: flag

*query:* `validation leakage overfitting pitfalls binary image classification log loss (binary cross-entropy) Classify images as dogs or cats and predict the probability `

*9 docs | similarity max=0.597 median=0.587 min=0.581*

- `0.597` **nb_3468349** (aerial-cactus-identification) This notebook tackles binary image classification on a dataset of 32x32 photographs to detect the presence of cacti. The author builds a straightforward convolutional neural networ
- `0.596` **nb_5211968** (aerial-cactus-identification) This notebook addresses binary image classification on the Aerial Cactus Identification dataset, where the task is to detect whether 32x32 pixel aerial images contain cacti. The au
- `0.595` **nb_4335296** (aerial-cactus-identification) This notebook addresses binary image classification on aerial photographs to detect the presence of cacti. The dataset consists of 32x32 pixel training images with binary labels, s
- `0.588` **nb_3709172** (aerial-cactus-identification) This notebook addresses binary image classification on 32x32 pixel photographs to detect whether an image contains a cactus. The author loads a training set from CSV metadata paire
- `0.587` **nb_4722040** (aerial-cactus-identification) This notebook tackles a binary image classification task: detecting the presence of cacti in photographs. The dataset consists of training images with binary labels and unlabeled t
- `0.586` **nb_3599227** (aerial-cactus-identification) This notebook addresses a binary image classification task: detecting the presence of cacti in photographs. The dataset comprises labeled training images and unlabeled test images,
- `0.585` **nb_3935716** (aerial-cactus-identification) This notebook addresses binary image classification on a cactus detection task, where 32×32 images must be labeled as containing a cactus or not. The author uses transfer learning 
- `0.582` **nb_3809084** (aerial-cactus-identification) This notebook addresses binary image classification on a cactus detection dataset, where the task is to predict whether 32x32 pixel images contain cacti. The author loads training 
- `0.581` **nb_4006542** (aerial-cactus-identification) This notebook tackles binary image classification to detect the presence of cacti in aerial photographs. The problem is a straightforward supervised learning task where the goal is

### C: advise

*query:* `model architecture training approach binary image classification log loss (binary cross-entropy) Classify images as dogs or cats and predict the probability tha`

*11 docs | similarity max=0.654 median=0.645 min=0.641*

- `0.654` **nb_3599227** (aerial-cactus-identification) This notebook addresses a binary image classification task: detecting the presence of cacti in photographs. The dataset comprises labeled training images and unlabeled test images,
- `0.649` **nb_4512708** (aerial-cactus-identification) This notebook addresses binary image classification on a dataset of 32x32 pixel images, tasking the model to distinguish whether images contain cacti. The author loads 17,500 train
- `0.648` **nb_3809084** (aerial-cactus-identification) This notebook addresses binary image classification on a cactus detection dataset, where the task is to predict whether 32x32 pixel images contain cacti. The author loads training 
- `0.647` **nb_4722040** (aerial-cactus-identification) This notebook tackles a binary image classification task: detecting the presence of cacti in photographs. The dataset consists of training images with binary labels and unlabeled t
- `0.645` **nb_7627112** (aerial-cactus-identification) This notebook addresses binary image classification on aerial photographs to detect the presence of cacti. The dataset contains 19,500 training images of 32x32 pixels with binary l
- `0.645` **nb_3283695** (aerial-cactus-identification) This notebook addresses binary image classification on a dataset of aerial photos to detect the presence of cacti. The author built a custom convolutional neural network from scrat
- `0.643` **nb_4055899** (aerial-cactus-identification) This notebook addresses binary image classification on a dataset of aerial photographs to detect whether each image contains a cactus. The author built a custom convolutional neura
- `0.642` **nb_3614756** (aerial-cactus-identification) This notebook addresses binary image classification on 32x32 images where the task is to detect the presence of cacti. The author built a four-layer convolutional neural network tr
- `0.641` **nb_4094573** (aerial-cactus-identification) This notebook addresses a binary image classification problem where the task is to predict whether images contain cacti. The author loads 17,500 training images and 4,000 test imag
- `0.641` **nb_3684914** (aerial-cactus-identification) This notebook addresses binary image classification on a cactus presence detection task. The dataset consists of 32x32 pixel RGB images with binary labels indicating whether a cact
- `0.641` **nb_4447135** (aerial-cactus-identification) This notebook addresses binary image classification: determining whether 32x32 pixel images contain cacti or not. The dataset consists of 17,500 training images with labels and a t

*distinct summaries across C's stages: 24 (parity target 24)*
