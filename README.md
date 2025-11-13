# Fake News Propagation Modeling with a Temporal GNN

*A research / demo project by Steve, Suhana, Joel*

This repository moved from a classic text-classification demo to a small research prototype that models how news (real or fake) spreads as cascades and predicts cascade size using a **Temporal Graph Neural Network (GNN)** over snapshots of user-interaction graphs.

---

# 📌 What this project does (short)

* Synthesises social “events” (retweets / reposts) for many articles and builds time-ordered snapshots of user interaction graphs.
* Encodes per-event text with a sentence transformer and aggregates per-user embeddings.
* Builds graph snapshots (node features = user embedding + node degree) and feeds sequences of snapshot graphs into a Temporal GNN (GCN per snapshot + GRU across time) to predict the final cascade size (log-scaled).
* Trains and evaluates the model, saves `temporal_gnn_model.pth`, and visualises propagation via network animations.

---

# 🔧 Key files / structure

```
Data-Mining-Project/
│
├── data/                             ← (optional) store real datasets here
│
├── notebook.ipynb                    ← Jupyter notebook containing the pipeline and visualizations
├── temporal_gnn_model.pth            ← saved PyTorch model (after training)
├── requirements.txt                  ← pip dependencies (see below)
└── README.md                         ← this file
```

---

# 🧰 Main components (what’s in the code)

* **Synthetic data generator**

  * `generate_synthetic_events()` — creates synthetic cascades with timestamps, user ids, parent_user_id (reply/retweet edges), and small text labels (`"original"` / `"retweet"`).

* **Snapshot construction**

  * `build_snapshots_for_article(events, snapshot_count, window_seconds, embedder)` — slices events into `snapshot_count` cumulative time windows, creates node lists, edge_index, and per-node feature vectors (mean of that user’s sentence embeddings + node degree).

* **Dataset / dataloader**

  * `CascadeSequenceDataset` — groups events by article and returns a sequence of `torch_geometric.data.Data` graphs plus a log-scaled target (final cascade size).
  * `collate_with_counts` — custom collate fn that batches per-time graphs while tracking node counts so we can later un-batch per-article.

* **Model**

  * `SnapshotEncoder` — two-layer GCN that encodes a snapshot into node embeddings.
  * `TemporalGNN` — applies SnapshotEncoder at each time step, aggregates node embeddings to per-graph vectors, sequences them into a GRU, then an MLP to regress final cascade size.

* **Embedding**

  * `SentenceTransformer('all-MiniLM-L6-v2')` — produces per-event text embeddings used to build node features.

* **Training & evaluation**

  * Training loop (Adam + StepLR), MSE loss on log1p targets, simple validation, saving state to `temporal_gnn_model.pth`.
  * Quick scatter plot of true vs predicted cascade sizes (denormalised with `expm1`) and a textual printout of a few predictions.

* **Visualization**

  * `visualize_fake_news_propagation(...)` — NetworkX + `matplotlib.animation.FuncAnimation` to show a simple SIR-like diffusion (for demo/illustration).

---

# ✅ Features

* End-to-end pipeline: synthetic data → snapshot graphs → temporal GNN → evaluation.
* Uses sentence embeddings to incorporate textual signals in node features.
* Graph neural networks (GCN) for structural modeling per snapshot.
* Temporal modeling with GRU to capture evolution across snapshots.
* Graph animation helper to visualise propagation dynamics.

---

# 📦 Dependencies

Minimum libraries used in the notebook / script:

* `python >= 3.8`
* `numpy`, `pandas`
* `matplotlib`, `networkx`
* `tqdm`
* `sentence-transformers`
* `torch` (matching your CUDA / CPU), `torchvision`, `torchaudio`
* `torch_geometric` and required PyG support packages

Example `requirements.txt` snippet:

```
numpy
pandas
matplotlib
networkx
tqdm
sentence-transformers
torch>=2.0
torchvision
torchaudio
torch-geometric
```

> Note: installing `torch_geometric` requires matching wheels for your `torch` version/platform. See PyG install docs if you hit wheel issues.

Quick pip install (CPU, example):

```bash
pip install numpy pandas matplotlib networkx tqdm sentence-transformers
pip install torch torchvision torchaudio               # choose CPU vs CUDA as needed
# PyG wheel depends on torch version; example (CPU) may be:
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

---

# 🚀 How to run

1. (Optional) Clone the repo

   ```bash
   git clone https://github.com/SamDarkKnight/Data-Mining-Project.git
   cd Data-Mining-Project
   ```

2. Install dependencies (see above).

3. Launch the notebook:

   ```bash
   jupyter notebook notebook.ipynb
   ```

   or run the script cells in order. The code auto-detects device (`cuda` if available).

4. The notebook will:

   * Generate synthetic cascades,
   * Build sentence embeddings,
   * Create the dataset and dataloaders,
   * Train the Temporal GNN for `EPOCHS` (default in code: 5),
   * Save the trained model to `temporal_gnn_model.pth`,
   * Show prediction scatter and animate propagation.

---

# 🔬 Example hyperparameters (from the code)

* `SENT_EMB_MODEL = 'all-MiniLM-L6-v2'`
* `TIME_WINDOW_SECONDS = 3600`
* `SNAPSHOT_COUNT = 8`
* `EPOCHS = 5`, batch size = 8, optimizer = Adam (lr=1e-4)
* Loss = MSE on `log1p(cascade_size)` target

These are easy to change at the top of the notebook for experiments.

---

# 📈 Results (example / demo)

When run with the provided synthetic setup the notebook prints training/validation loss each epoch, saves the model, and produces a scatter plot of true vs predicted cascade sizes (expm1 to denormalise). The printed sample predictions show the model’s ability to approximate cascade sizes on held-out synthetic data (results will vary with random seed and hyperparameters).

---

# 🧠 Notes, limitations & next steps

* **Synthetic data** is used for prototyping — for production / research you should use real social trace datasets (with timestamps, user IDs, reply/reshare relationships).
* Sentence embeddings for `"original"` / `"retweet"` are placeholders in the demo. Replace with actual article headlines / content for meaningful text signals.
* The current target is *cascade size* (regression). For classification (will it go viral?), you can threshold size or change the MLP to output logits.
* Consider richer features: user metadata, temporal inter-arrival times, edge weights, or attention mechanisms across nodes.
* For larger graphs, batching strategies and memory optimisations will be necessary (PyG supports many utilities).

---

# 🤝 Contributing

Contributions welcome — ideas to try:

* Replace synthetic generator with a real dataset (put CSVs in `data/`).
* Add classification heads (viral / non-viral).
* Experiment with other GNN layers (GraphSAGE, GAT), transformer encoders across snapshots, or contrastive pretraining.
* Improve the visualization (save animation to mp4/gif).

Workflow:

1. Fork
2. Branch with feature
3. Open PR with a clear description

---

# 📄 License

Provided for educational and research use. Cite or attribute if you use the code or ideas.

---

If you want, I can:

* produce an updated `requirements.txt` with exact pinned versions used in the notebook,
* convert the notebook into a runnable Python script (`train.py`) with CLI args,
* or trim the README into a short project blurb for GitHub — tell me which one and I’ll produce it.
