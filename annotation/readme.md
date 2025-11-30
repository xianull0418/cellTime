本项目开发单细胞时序模型cellTime。将其中下游任务-细胞注释的性能与其他单细胞模型如scfoundation进行评估，


输入数据集：Zheng 68k PBMC（外周血单个核细胞）

总体流程：
1  输入 Zheng 68k PBMC 数据集；提取细胞嵌入（目前因为时序模型还没训练好，先取高变基因，然后取每个细胞的高变基因表达作为嵌入；假如后续有了时序模型，就用时序模型的嵌入）
2  输入细胞嵌入，训练注释分类器与预测（简单的两层MLP）
    【训练/评测数据，要先训练才能评测，能否用评测数据集如Zheng 68k PBMC进行训练。假如不能，训练数据是不是还得准备训练数据】
        重点结果细胞类型标签结果*.pkl 形式提供（如 seg-emb.pkl 、 zheng-emb-2mlp.pkl ）。这些 pkl 内含每个细胞的类别得分/概率矩阵，采用 argmax 得到预测类别索引，再用 *-str_label.npy 映射成细胞类型字符串
3 基于细胞嵌入聚类绘图，分组颜色使用细胞注释标签（真实标签true和预测标签pred）。
4 比较不同模型（如cellTime、scfoundation）在细胞注释任务上的性能差异。

运行环境scgpt_py3.9

## train_mlp_annotation.py 
- 读取数据与生成嵌入 cells_embedding.npy
- 训练两层 MLP、保存 pred_labels.npy 与 zheng-emb-2mlp.pkl, 
- 预测结果metrics.txt

python train_mlp_annotation.py --data_path ./data/celltypist_0806_zheng68k.h5ad --label_path ./data/zheng-test-label.npy --label_names_path ./data/zheng-str_label.npy --save_dir ./result/ --hvg_num 2000 --hvg_flavor seurat --epochs 30 --hidden_dim 256 --batch_size 512 --pkl_name zheng-emb-2mlp.pkl
## plot_umap_labels.py 基于细胞嵌入聚类绘图，分组颜色使用细胞注释标签（真实标签true和预测标签pred）。
python plot_umap_labels.py --embedding_path ./result/cells_embedding.npy --true_label_path ./data/zheng-test-label.npy --pred_label_path ./result/pred_labels.npy --label_names_path ./data/zheng-str_label.npy --save_dir ./result/ --prefix umapcf_zheng


## 还需补充，与其他模型（如scgpt，cellfm、scfoundation）在细胞注释任务上的性能的比较。