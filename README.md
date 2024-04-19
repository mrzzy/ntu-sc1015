# Opening a Business

## Problem Statement
### What Makes a Business Stay Open?
As potential business owners with no ingenious business idea, what business should we open and how do we ensure it stays open? Using our new skills from SC1015, we attempt to take a data-driven approach to answer the question. The analysis will be based on a dataset provided by Yelp.

## Contributors
- @mrzzy - Data Preparation, Machine Learning
- @jyorien - EDA Analsys (1 - 3)
- @RogerKwek - EDA Analsys (4 - 5)

## Data Preparation
[Data Preparation](dataprep.ipynb): Dataset sampling & denormalisation with Apache Spark & Parquet.

## Exploratory Data Analysis Questions
1) Do Closed businesses have more negative reviews? [Q1](https://github.com/mrzzy/ntu-sc1015/blob/main/nlp/sentiment_analysis.ipynb)
2) Do Open businesses have more reviews than Closed businesses? [Q2](https://github.com/mrzzy/ntu-sc1015/blob/main/nlp/businesses.ipynb)
3) What do people say about businesses that are Opened vs Closed? [Q3](https://github.com/mrzzy/ntu-sc1015/blob/main/nlp/tf_analysis.ipynb)
4) Does the Category of the Business Affect Whether it will Remain Open? [Q4](https://github.com/mrzzy/ntu-sc1015/blob/main/category%20and%20states/Categories.ipynb)
5) Does Location Affect Whether a Business Remains Open? [Q5](https://github.com/mrzzy/ntu-sc1015/blob/main/category%20and%20states/States.ipynb)

## Machine Learning
[Modeling Notebook](model.ipynb): Class imbalance, Feature Engineering, Model Selection, Hyperparameter tuning and Model Analysis.

## Conclusion
- Encourage customers to provide more feedback and reviews on Yelp.
- Category of businesses that are likely to stay open are: `Shopping`, `Home Services`, and `Health & Medical`
- Focus for business should be on delivering good food (assuming it is a food establishment), and providing good customer service
- Business should be situated in `Pennsylvania` and `Florida` for a good chance of staying open


**References**

- https://www.yelp.com/dataset/
- https://parquet.apache.org/docs/file-format/data-pages/encodings/
- https://blog.yelp.com/businesses/yelp_category_list/
- https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://blog.yelp.com/businesses/yelp_category_list/

**Appendix** 
- [Detailed README with our analyses combined](https://github.com/mrzzy/ntu-sc1015/blob/main/detailed_README.md)

