import numpy as np
from numpy.random import sample
from collections import Counter
import torch
import faiss
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from logging import getLogger
from recbole.utils import set_color

class ICPSampler():
    def __init__(self, config, train_data, model):
        self.data = train_data
        self.model = model
        self.logger = getLogger()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_clusters = config['num_clusters']
        self.popularity_alpha = config['popularity_alpha']
        self.saved_model_file = config['saved_model_path']
        self.cluster_method = config['cluster_method']
        self.strategy = config['strategy'].lower()
        
        self.strategy_map = {
            'icpns': self.sample_from_cached_distribution,
        }
        
        self.user_2cluster = None
        self.cluster_item_weights = None
        self.user_pos_items = None
        self.user_alias_cache = []

    def cluster_users(self, user_emb_np):
        if self.cluster_method == 'kmeans':
            model = KMeans(n_clusters=self.num_clusters, random_state=2026, n_init='auto')
        elif self.cluster_method == 'minibatch':
            model = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=2026)
        elif self.cluster_method == 'gmm':
            model = GaussianMixture(n_components=self.num_clusters, random_state=2026)
        elif self.cluster_method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        elif self.cluster_method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=self.num_clusters)
        elif self.cluster_method == 'spectral':
            model = SpectralClustering(n_clusters=self.num_clusters, assign_labels='discretize', random_state=2026)
        elif self.cluster_method == 'faiss':
            d = user_emb_np.shape[1]
            kmeans = faiss.Kmeans(d, self.num_clusters, niter=20, verbose=False)
            kmeans.train(user_emb_np.astype('float32'))
            _, labels = kmeans.index.search(user_emb_np.astype('float32'), 1)
            return labels.squeeze()
        return model.fit_predict(user_emb_np)

    def prepare_data(self):
        self.logger.info("Preparing data for ICPNS fine-tuning")
        dataset = self.data.dataset
        
        if not hasattr(dataset, 'user_positive_item_dict'):
            uid = dataset.inter_feat[dataset.uid_field].numpy()
            iid = dataset.inter_feat[dataset.iid_field].numpy()
            user_pos = {}
            for u, i in zip(uid, iid):
                user_pos.setdefault(u, []).append(i)
            dataset.user_positive_item_dict = user_pos
        
        self.user_pos_items = dataset.user_positive_item_dict
        n_users, n_items = dataset.user_num, dataset.item_num
        
        if not hasattr(dataset, 'user_interaction_matrix'):
            uids = dataset.inter_feat[dataset.uid_field].cpu().numpy()
            iids = dataset.inter_feat[dataset.iid_field].cpu().numpy()
            dataset.user_interaction_matrix = csr_matrix(
                (np.ones(len(uids), dtype=np.float32), (uids, iids)),
                shape=(n_users, n_items)
            )
        interaction_matrix = dataset.user_interaction_matrix

        with torch.no_grad():
            self.model.eval()
            user_embeddings, _ = self.model.forward()
            self.user_2cluster = self.cluster_users(user_embeddings.cpu().numpy())
        
        self.cluster_item_weights = np.zeros((self.num_clusters, n_items), dtype=np.float32)
        for cid in tqdm(range(self.num_clusters)):
            users_in_cluster = np.where(self.user_2cluster == cid)[0]
            if len(users_in_cluster) == 0:
                continue
            item_counts = np.array(interaction_matrix[users_in_cluster].sum(axis=0)).flatten()
            self.cluster_item_weights[cid] = np.power(item_counts, self.popularity_alpha)

        self.build_user_neg_cache_alias(n_users, n_items)

    def build_user_neg_cache_alias(self, n_users, n_items):
        self.user_alias_cache = [None] * n_users
        for u in tqdm(range(n_users)):
            cid = self.user_2cluster[u]
            w = self.cluster_item_weights[cid].copy()
            pos_items = self.user_pos_items.get(u, [])
            if len(pos_items) > 0:
                w[pos_items] = 0.0
            valid = np.flatnonzero(w)
            if len(valid) == 0:
                w = np.ones(n_items, dtype=np.float32)
                if len(pos_items) > 0:
                    w[pos_items] = 0.0
                valid = np.flatnonzero(w)
            probs = w[valid]
            probs = probs / probs.sum()
            K = len(probs)
            q = probs * K
            J = np.zeros(K, dtype=np.int32)
            small = []
            large = []
            for i, qi in enumerate(q):
                if qi < 1.0:
                    small.append(i)
                else:
                    large.append(i)
            while small and large:
                s = small.pop()
                l = large.pop()
                J[s] = l
                q[l] = q[l] + q[s] - 1.0
                if q[l] < 1.0:
                    small.append(l)
                else:
                    large.append(l)
            self.user_alias_cache[u] = (q.astype(np.float32), J, valid.astype(np.int32))

    def sample_from_cached_distribution(self, users_tensor):
        users = users_tensor.cpu().numpy()
        n = len(users)
        kk = np.random.randint(0, high=1 << 30, size=n)
        uu = np.random.rand(n).astype(np.float32)
        out = np.empty(n, dtype=np.int64)
        for i, u in enumerate(users):
            q, J, items = self.user_alias_cache[u]
            K = len(q)
            k = kk[i] % K
            out[i] = items[k] if uu[i] < q[k] else items[J[k]]
        return torch.from_numpy(out).to(self.device)

    def sample(self, users):
        if self.strategy in self.strategy_map:
            return self.strategy_map[self.strategy](users)
        else:
            raise ValueError(f"Unsupported sampling strategy: {self.strategy}")
