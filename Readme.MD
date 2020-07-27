# dicgen
### Automatic dictionary generation ideas
>  pip3 install -r requirements.txt
- Clone this repo to your local machine using  `https://github.com/imdbo/dicgen.git`
- This project uses Django 3.0.2, it needs to be installed along with MySQL `https://docs.djangoproject.com/en/3.0/ref/databases/`

### Migrate the django project database
```shell
$ python3 manage.py makemigrations
$ python3 manage.py migrate
$ python3 manage.py runserver
```
### Download and pre-process a Wikipedia text dump
```shell
$ python3 dump_wiki.py
```
### Download a word2vec or fasttext model
- you can find models for different languages here `https://embeddings.sketchengine.co.uk/static/index.html`
```shell
$ wget https://embeddings.sketchengine.co.uk/static/models/lemma_lc/ententen13_tt2_1.vec
```

### Perform the summarization of the data
```shell
$ python3 summarization.py
```

### Generate examples for each entry
```shell
$ python3 gen_examples.py
```

### Optional. F1 score of the results against Merriam Webster
```shell
$ python3 quantitative_comparison.py
```
