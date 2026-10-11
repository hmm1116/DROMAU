import numpy as np
import torch


class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per, n_query):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.n_query = n_query

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        n_support = self.n_per - self.n_query
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[: self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                shuffled = torch.randperm(len(l))
                query_pos = shuffled[: self.n_query]
                remaining = shuffled[self.n_query :]
                if len(remaining) < n_support:
                    support_pos = remaining[torch.randint(len(remaining), (n_support,))]
                else:
                    support_pos = remaining[:n_support]
                pos = torch.cat((support_pos, query_pos))
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
