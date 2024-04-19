# Opening a Business

## Problem Statement
### What Makes a Business Stay Open?
As potential business owners with no ingenious business idea, what business should we open and how do we ensure it stays open? Using our new skills from SC1015, we attempt to take a data-driven approach to answer the question. The analysis will be based on a dataset provided by Yelp.

## Contributors
- @mrzzy - [Data Preparation](/Data%20Preparation/dataprep.ipynb), [Machine Learning](/Machine%20Learning/model.ipynb)
- @jyorien - [EDA Analysis (1 - 3)](/NLP/)
- @RogerKwek - [EDA Analysis (4 - 5)](/Category%20and%20States/)

## Data Preparation
[Data Preparation](/Data%20Preparation/dataprep.ipynb): Dataset sampling & denormalisation with Apache Spark & Parquet.

[Yelp Dataset Website](https://www.yelp.com/dataset): Yelp Main website with the raw data

[Review json](https://ntu-sc1015-yelp.s3.ap-southeast-1.amazonaws.com/yelp_reviews.parquet/part-00000-tid-8578034499791727984-cc969862-514e-46d4-990e-74c842c22502-1008-1-c000.snappy.parquet): Data used in our EDA

[Business json](Dataset/yelp_academic_dataset_business.json): Data used in our EDA


## Exploratory Data Analysis Questions
1) [Do Closed businesses have more negative reviews?](/NLP/sentiment_analysis.ipynb)
2) [Do Open businesses have more reviews than Closed businesses?](/NLP/businesses.ipynb)
3) [What do people say about businesses that are Opened vs Closed?](/NLP/tf_analysis.ipynb)
4) [Does the Category of the Business Affect Whether it will Remain Open?](/Category%20and%20States/Categories.ipynb)
5) [Does Location Affect Whether a Business Remains Open?](/Category%20and%20States/States.ipynb)

## Machine Learning
[Modeling Notebook](/Machine%20Learning/model.ipynb): Class imbalance, Feature Engineering, Model Selection, Hyperparameter tuning and Model Analysis.

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
- [Detailed README with our analyses combined](/README_detailed.md)

