## Environment

+ cpu: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz

+ mem: 126GB

+ system: Ubuntu 20.04.1 LTS

+ gpu: Nvidia 2080ti*4

+ nvcc: cuda_11.3.r11.3/compiler.29745058_0

  [attention] the 30-series architecture might not officially support cuda11.

+ conda env:

  `conda_env.yaml`(pkgs installed by conda)

  `spec-list.txt`(pkgs installed by pip)

### Migrating to another dataset

1. prepare dataset:

    use knn to generate graphs with patient infos: tag, label(lbl), survtime

    the graphs have ndata keys : adjv lst(type of nodes) feats and edata keys : lst(type of edges)

    format like that:

      ```python
      data = BRCADataset(graph_bin_path="./graphs.bin", info_pkl_path="./info.pkl")
      data.info()
      """
          tags[:3]      : ['TCGA-3C-AALI' 'TCGA-3C-AALJ' 'TCGA-3C-AALK']
          lbls[:3]      : tensor([0, 0, 0])
          survtimes[:3] : tensor([126.7000,  40.9333,  40.5667], dtype=torch.float64)
          graphs[0]     : 
              Graph(
                  num_nodes=900, 
                  num_edges=45000,
                  ndata_schemes={
                    'adjv': Scheme(shape=(9,), dtype=torch.float32), 
                    'lst': Scheme(shape=(), dtype=torch.int32), 
                    'feats': Scheme(shape=(1000,), dtype=torch.float32)
                  }
                  edata_schemes={'lst': Scheme(shape=(), dtype=torch.int32)}
              )
      """
      ```

2. exec feature_embedding

   > feature_embedding1000.py
   > 
   > set cmd params (--cuda --dataset and etc.) and then run that file
   > 

3. exec run.py

   > pipeline/brca/run.py
   >
   > set cmd params (--fold --device and etc.) and then run that file
   >
