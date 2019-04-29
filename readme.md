# Lifelong Sequential Modeling with Personalized Memorizationfor User Response Prediction

A `tensorflow` implementation of all the compared models for the SIGIR 2019 paper: Lifelong Sequential Modeling with Personalized Memorizationfor User Response Prediction.

The experiments are supported by Alimama Rank Group from Alibaba Inc.
Paper will be published ASAP.

If you have any problems, please propose an issue or contact the authors: [Kan Ren](http://saying.ren/), [Jiarui Qin](http://apex.sjtu.edu.cn/members/qinjr) and [Yuchen Fang](http://apex.sjtu.edu.cn/members/arthur_fyc).

### Data Preparation
- In the `data/amazon` folder, we give three small sample dataset that has been preprocessed, the sample code is running on the sample data. The `dataset_crop.pkl` is for the baseline `SHAN` (cut a short-term and a long-term sequence) and `dataset_hpmn.pkl` (padding in the front) is for our `HPMN` model, all the other baselines are based on the `dataset.pkl`
- For the full dataset, the raw dataset link are [Amazon](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz) and [Taobao](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649). 
- After you downloaded the full datasets, e.g. `Amazon`, you should extract the files into data/raw_data/amazon and do the following to preprocess the dataset:
```
python preprocess_amazon.py
```
### Run the Codes
- To run the code of the models:
```
python hpmn.py [DATASET] # To run HPMN, [DATASET] can be amazon or taobao
python train.py [MODEL] [DATASET] [GPU] # To run DNN/SVD++/GRU4Rec/DIEN/Caser/LSTM/, [GPU] is the GPU env 
python shan.py [DATASET] [GPU] # To run SHAN
python rum.py [DATASET] [GPU] # To run RUM
```
