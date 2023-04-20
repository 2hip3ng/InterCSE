import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer


class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.dense = nn.Linear(768, 1)
        self.activation = nn.Sigmoid()
        self.step = 0
        self.all_step = 144*10*2*2*2
    
    def get_alpha(self):
        alpha = 0.001
        if self.step / self.all_step < 0.2:
            alpha = 10
        elif self.step / self.all_step < 0.4:
            alpha = 1 
        elif self.step / self.all_step < 0.6:
            alpha = 0.1 
        elif self.step / self.all_step < 0.8:
            alpha = 0.01
        elif self.step / self.all_step < 1.0:
            alpha = 0.001
        return alpha

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        for i, sentence_feature in enumerate(sentence_features):
            sentence_feature["use_cls"] = False
            if i == 2:
                sentence_feature["use_cls"] = True
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        pooling = embeddings[2]
        inter = self.dense(pooling)
        inter = self.activation(inter)
        
        self.step += 1
        
        loss1 = self.loss_fct(output, labels.view(-1))
        loss1 = loss1 
        loss2 = self.loss_fct(inter, labels.view(-1))
        loss2 = loss2
        
        alpha = self.get_alpha()
        loss = loss1 + alpha * loss2
        return loss
