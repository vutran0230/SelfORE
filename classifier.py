import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import random
import numpy as np
from tqdm import tqdm

class Classifier:
    def __init__(self, k, sentence_path, max_len, batch_size, epoch, fp16=False):
        with open(sentence_path) as f:
            sentences = f.readlines()
        self.k = k
        self.epoch = epoch
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = self.get_tokenizer()
        self.fp16 = fp16
        self.device = self.get_device()
        self.model = self.get_model().to(self.device)
        self.using_parallel_model = False
        if torch.cuda.device_count()>1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
            self.batch_size = batch_size * torch.cuda.device_count()
            self.using_parallel_model = True
        
        self.input_ids, self.attention_masks, self.entity_idx = self.prepare_data(sentences)
        self.dataset = TensorDataset(self.input_ids, self.attention_masks, self.entity_idx)

        self.dataloader = DataLoader(
            self.dataset,
            sampler=SequentialSampler(self.dataset),
            batch_size=self.batch_size
        )

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    @staticmethod
    def get_tokenizer():
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)
        special_tokens_dict = {'additional_special_tokens': [
            '[E1]', '[E2]', '[/E1]', '[/E2]']}  # add special token
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    def get_model(self):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=self.k,
            output_attentions=False,
            output_hidden_states=True,
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def prepare_data(self, sentences):
        input_ids = []
        attention_masks = []
        entity_idx = []
        e1_tks_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        e2_tks_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        e2e_tks_id = self.tokenizer.convert_tokens_to_ids('[/E2]')
        lastsent = None
        brokensents_bar = tqdm(total=len(sentences),desc='broken sentences')
        
        for sent in tqdm(sentences,desc='preparing'):
            encoded_dict = self.tokenizer.encode_plus(
                sent,                        # Sentence to encode.
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                max_length=self.max_len,     # Pad & truncate all sentences.
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',         # Return pytorch tensors.
            )
            input_id = encoded_dict['input_ids']
            e1_idx = (input_id == e1_tks_id).nonzero(as_tuple=False)
            e2_idx = (input_id == e2_tks_id).nonzero(as_tuple=False)
            e2e_idx = (input_id == e2e_tks_id).nonzero(as_tuple=False)
            if not len(e2e_idx): 
                brokensents_bar.update(1)
                if not len(e1_idx):
                    e1_idx = [[0,0]]
                if not len(e2_idx):
                    e2_idx = [[0,0]]
            entity_idx.append([e1_idx[0][1], e2_idx[0][1]])
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            lastsent = sent
        brokensents_bar.close()

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        entity_idx = torch.LongTensor(entity_idx)
        print('Original: ', lastsent)
        print('Token IDs:', input_ids[-1] if len(input_ids) else [])
        print('Entity positions:', entity_idx[-1] if len(entity_idx) else [])
        print('Evocab IDs',e1_tks_id,e2_tks_id)
        print('WARN','broken sentences missing [E1] or [E2] may be represented as [CLS], [CLS].')
        return input_ids, attention_masks, entity_idx

    def get_hidden_state(self, *, verbose=2):
        self.model.eval()
        dataiter = self.dataloader
        outs = None
        nouts = 0
        if verbose>=2:
            dataiter = tqdm(dataiter)
        with torch.no_grad():
            for x, a, epos in dataiter:
                x = x.to(device=self.device)
                a = a.to(device=self.device)
                epos = epos.numpy()
                out0 = self.model(x, a)[1][-1].detach().cpu().numpy()
                out1 = np.array([v[p] for v,p in zip(out0,epos)])
                out = out1.reshape(x.shape[0], -1)
                
                if outs is None:
                    outs = np.empty((len(self.dataset),out.shape[1]),dtype='float32' if not self.fp16 else 'float16')
                outs[nouts:nouts+out.shape[0]] = out
                nouts += out.shape[0]
        return outs

    def train(self, labels, *, verbose=2):
        labels = torch.tensor(labels).long()
        dataset = TensorDataset(self.input_ids, self.attention_masks, labels)

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size])
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size
        )

        self.validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size
        )
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        epochs = self.epoch
        total_steps = len(self.train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        self.model.cuda()
        for epoch_i in range(0, epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            self.train_epoch(verbose=verbose)

    def train_epoch(self, *, verbose=2):
        total_train_loss = 0
        self.model.train()
        dataiter = self.train_dataloader
        if verbose>1:
            dataiter=tqdm(dataiter,desc='train batch')
        for batch in dataiter:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            self.model.zero_grad()
            loss, logits, _ = self.model(b_input_ids,
                                         token_type_ids=None,
                                         attention_mask=b_input_mask,
                                         labels=b_labels)
            loss = loss.mean()
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
        avg_train_loss = total_train_loss / len(self.train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("")
        print("Running Validation...")
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            with torch.no_grad():
                (loss, logits, _) = self.model(b_input_ids,
                                               token_type_ids=None,
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
            total_eval_loss += loss.mean().item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / \
            len(self.validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(self.validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def save(self,path):
        modeltosave = self.model.module if self.using_parallel_model else self.model
        modeltosave.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        

