import os
from utils import load_vertical_tagged_data, load_labeled_data
import torch
from torch.nn.utils.rnn import pad_sequence
import statistics as stat

class CLDataset():
    """

    """

    def __init__(self,
                 data_dir='./data/datasets/ade',
                 data_name='ade',
                 batch_size=2,
                 device='cuda',
                 lower=True,
                 vocab_size=1000000000,
                 pad='<pad>',
                 unk='<unk>'):
        """

        :param data_dir:
        :param data_name:
        :param batch_size:
        :param device:
        :param lower:
        :param vocab_size:
        :param pad:
        :param unk:
        """

        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.device = device
        self.lower = lower
        self.vocab_size = vocab_size
        self.PAD = pad
        self.UNK = unk
        self.PAD_ind = 0
        self.UNK_ind = 1
        self.populate_attributes()

    def populate_attributes(self):
        """

        """

        # Load training portion.
        (self.Ys_train, self.wordseqs_train, self.charseqslist_train, self.wordcounter_train, self.charcounter_train)\
            = load_labeled_data(os.path.join(self.data_dir, "neg_" + self.data_name + '_train_dev.json'),
                                os.path.join(self.data_dir, self.data_name + '_train_dev.json'))

        # Create index maps from training portion.
        self.word2x = self.get_imap(
            self.wordcounter_train,  max_size=self.vocab_size, lower=self.lower, pad_unk=True)
        #self.tag2y = self.get_imap(self.tagcounter_train, max_size=None, lower=False, pad_unk=True)
        #self.relation2y = self.get_imap(self.relcounter_train, max_size=None, lower=False, pad_unk=False)
        self.char2c = self.get_imap(
            self.charcounter_train, max_size=None, lower=False, pad_unk=True)

        # Load validation and test portions.
        (self.Ys_val, self.wordseqs_val, self.charseqslist_val, _, _) = load_labeled_data(
            os.path.join(self.data_dir, "neg_" + self.data_name + '_dev.json'),
            os.path.join(self.data_dir, self.data_name + '_dev.json'))
        (self.Ys_test, self.wordseqs_test, self.charseqslist_test, _, _) = load_labeled_data(
            os.path.join(self.data_dir, "neg_" + self.data_name + '_test.json'),
            os.path.join(self.data_dir, self.data_name + '_test.json'))

        # Prepare batches.
        self.batches_train = self.batchfy(
            self.Ys_train, self.wordseqs_train, self.charseqslist_train)
        self.batches_val = self.batchfy(
            self.Ys_val, self.wordseqs_val, self.charseqslist_val)
        self.batches_test = self.batchfy(
            self.Ys_test, self.wordseqs_test, self.charseqslist_test)

    def batchfy(self, _Ys, wordseqs, charseqslist):
        #print("_ys: "+ str(len(_Ys)))
        #print("wordseqs: "+ str(len(wordseqs)))

        """

        :param wordseqs:
        :param charseqslist:
        :return:
        """

        batches = []

        def add_batch(xseqs, Ys, cseqslist, raw_sentence):
            if not xseqs:
                return
            X = torch.stack(xseqs).to(self.device)  # B x T
            Y = torch.stack(Ys).to(self.device)  # B x T
            # List of BT tensors of varying lengths
            flattened_cseqs = [
                item for sublist in cseqslist for item in sublist]
            C = pad_sequence(flattened_cseqs, padding_value=self.PAD_ind,
                             batch_first=True).to(self.device)  # BT x T_char
            C_lens = torch.LongTensor([s.shape[0]
                                      for s in flattened_cseqs]).to(self.device)
            batches.append((X, Y, C, C_lens, raw_sentence))

        xseqs = []
        Ys = []
        cseqslist = []
        prev_length = float('inf')
        raw_sentence = []

        for i in range(len(wordseqs)):
            length = len(wordseqs[i])
            assert length <= prev_length  # Assume sequences in decr lengths
            wordseq = [word.lower()
                       for word in wordseqs[i]] if self.lower else wordseqs[i]
            
            xseq = torch.LongTensor(
                [self.word2x.get(word, self.UNK_ind) for word in wordseq])
            y = torch.LongTensor([_Ys[i]])

            cseqs = [torch.LongTensor([self.char2c[c] for c in word if c in self.char2c])  # Skip unknown
                     for word in wordseqs[i]]  # Use original words

            if length < prev_length or len(xseqs) >= self.batch_size:
                add_batch(xseqs, Ys, cseqslist, raw_sentence)
                xseqs = []
                Ys = []
                cseqslist = []
                raw_sentence = []

            xseqs.append(xseq)
            Ys.append(y)
            cseqslist.append(cseqs)
            prev_length = length
            raw_sentence.append(wordseqs[i])

        add_batch(xseqs, Ys, cseqslist, raw_sentence)

        return batches

    def get_imap(self, counter, max_size=None, lower=False, pad_unk=True):
        """

        :param counter:
        :param max_size:
        :param lower:
        :param pad_unk:
        :return:
        """

        if pad_unk:
            imap = {self.PAD: self.PAD_ind, self.UNK: self.UNK_ind}
        else:
            imap = {}
        if max_size is None or len(counter) <= max_size:
            strings = counter.keys()
        else:
            strings = list(zip(*sorted(counter.items(), key=lambda x: x[1],
                                       reverse=True)[:max_size]))[0]
        for string in strings:
            if lower:
                string = string.lower()
            if not string in imap:
                imap[string] = len(imap)
        return imap

    def log(self, logger):
        logger.log('CL Dataset')
        logger.log('-'*79)
        train_lengths = [len(xseq) for xseq in self.wordseqs_train]
        logger.log('Num train seqs: %d' % len(self.wordseqs_train))
        logger.log('\tAvg length: %d' % stat.mean(train_lengths))
        logger.log('\tMax length: %d' % max(train_lengths))
        logger.log('\tMin length: %d' % min(train_lengths))
        logger.log('\tStd length: %g' % stat.stdev(train_lengths))
        logger.log('Num val seqs: %d' % len(self.wordseqs_val))
        logger.log('Num test seqs: %d' % len(self.wordseqs_test))
        logger.log('')
        logger.log('Num word types: %d (including PAD/UNK)' %
                   len(self.word2x))
        #logger.log('Num CL label types: %d' %
        #           len(self.Ys))
        logger.log('Num char types: %d (including PAD/UNK)' %
                   len(self.char2c))
        logger.log('\t%s' % ' '.join(self.char2c.keys()))

class REDataset():
    """

    """

    def __init__(self,
                 data_dir='./data/datasets/conll04',
                 data_name='conll04',
                 batch_size=64,
                 device='cpu',
                 lower=True,
                 vocab_size=1000000000,
                 pad='<pad>',
                 unk='<unk>'):
        """

        :param data_dir:
        :param data_name:
        :param batch_size:
        :param device:
        :param lower:
        :param vocab_size:
        :param pad:
        :param unk:
        """

        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.device = device
        self.lower = lower
        self.vocab_size = vocab_size
        self.PAD = pad
        self.UNK = unk
        self.PAD_ind = 0
        self.UNK_ind = 1
        self.populate_attributes()

    def populate_attributes(self):
        """

        """

        # Load training portion.
        (self.wordseqs_train, self.tagseqs_train, self.relseqs_train, self.charseqslist_train,\
        self.wordcounter_train, self.tagcounter_train, self.relcounter_train, self.charcounter_train)\
         = load_vertical_tagged_data(os.path.join(self.data_dir, self.data_name + '_train_dev.json'))

        # Create index maps from training portion.
        self.word2x = self.get_imap(self.wordcounter_train, max_size=self.vocab_size, lower=self.lower, pad_unk=True)
        self.tag2y = self.get_imap(self.tagcounter_train, max_size=None, lower=False, pad_unk=True)
        self.relation2y = self.get_imap(self.relcounter_train, max_size=None, lower=False, pad_unk=False)
        self.char2c = self.get_imap(self.charcounter_train, max_size=None, lower=False, pad_unk=True)

        # Load validation and test portions.
        (self.wordseqs_val, self.tagseqs_val, self.relseqs_val, self.charseqslist_val, _, _, _, _) = load_vertical_tagged_data(
                                                                                    os.path.join(self.data_dir, self.data_name + '_dev.json'))
        (self.wordseqs_test, self.tagseqs_test, self.relseqs_test, self.charseqslist_test, _, _, _, _) = load_vertical_tagged_data(
                                                                                    os.path.join(self.data_dir, self.data_name + '_test.json'))

        # Prepare batches.
        self.batches_train = self.batchfy(self.wordseqs_train, self.tagseqs_train, self.relseqs_train, self.charseqslist_train)
        self.batches_val = self.batchfy(self.wordseqs_val, self.tagseqs_val, self.relseqs_val, self.charseqslist_val)
        self.batches_test = self.batchfy(self.wordseqs_test, self.tagseqs_test, self.relseqs_test, self.charseqslist_test)

    def batchfy(self, wordseqs, tagseqs, relseqs, charseqslist):
        """

        :param wordseqs:
        :param tagseqs:
        :param relseqs:
        :param charseqslist:
        :return:
        """

        batches = []
        def add_batch(xseqs, yseqs, rstartseqs, rendseqs, rseqs, cseqslist, raw_sentence):
            if not xseqs:
                return
            X = torch.stack(xseqs).to(self.device)  # B x T
            Y = torch.stack(yseqs).to(self.device)  # B x T
            flattened_cseqs = [item for sublist in cseqslist for item in sublist]  # List of BT tensors of varying lengths
            C = pad_sequence(flattened_cseqs, padding_value=self.PAD_ind, batch_first=True).to(self.device)  # BT x T_char
            C_lens = torch.LongTensor([s.shape[0] for s in flattened_cseqs]).to(self.device)
            batches.append((X, Y, C, C_lens, rstartseqs, rendseqs, rseqs, raw_sentence))

        xseqs = []
        yseqs = []
        rstartseqs = []
        rendseqs = []
        rseqs = []
        cseqslist = []
        prev_length = float('inf')
        raw_sentence = []

        for i in range(len(wordseqs)):
            length = len(wordseqs[i])
            assert length <= prev_length  # Assume sequences in decr lengths
            wordseq = [word.lower() for word in wordseqs[i]] if self.lower else wordseqs[i]
            xseq = torch.LongTensor([self.word2x.get(word, self.UNK_ind) for word in wordseq])
            yseq = torch.LongTensor([self.tag2y.get(tag, self.UNK_ind) for tag in tagseqs[i]])
            rstartseq = []
            rendseq = []
            rseq = []
            for rel in relseqs[i]:
                rstartseq.append(rel[0])
                rendseq.append(rel[1])
                rseq.append(self.relation2y[rel[2]])
            rstartseq = torch.LongTensor(rstartseq)
            rendseq = torch.LongTensor(rendseq)
            rseq = torch.LongTensor(rseq)

            cseqs = [torch.LongTensor([self.char2c[c] for c in word if c in self.char2c])  # Skip unknown
                     for word in wordseqs[i]]  # Use original words

            if length < prev_length or len(xseqs) >= self.batch_size:
                add_batch(xseqs, yseqs, rstartseqs, rendseqs, rseqs, cseqslist, raw_sentence)
                xseqs = []
                yseqs = []
                rstartseqs = []
                rendseqs= []
                rseqs = []
                cseqslist = []
                raw_sentence = []

            xseqs.append(xseq)
            yseqs.append(yseq)
            rstartseqs.append(rstartseq)
            rendseqs.append(rendseq)
            rseqs.append(rseq)
            cseqslist.append(cseqs)
            prev_length = length
            raw_sentence.append(wordseqs[i])

        add_batch(xseqs, yseqs, rstartseqs, rendseqs, rseqs, cseqslist, raw_sentence)

        return batches

    def get_imap(self, counter, max_size=None, lower=False, pad_unk=True):
        """

        :param counter:
        :param max_size:
        :param lower:
        :param pad_unk:
        :return:
        """

        if pad_unk:
            imap = {self.PAD: self.PAD_ind, self.UNK: self.UNK_ind}
        else:
            imap = {}
        if max_size is None or len(counter) <= max_size:
            strings = counter.keys()
        else:
            strings = list(zip(*sorted(counter.items(), key=lambda x: x[1],
                                       reverse=True)[:max_size]))[0]
        for string in strings:
            if lower:
                string = string.lower()
            if not string in imap:
                imap[string] = len(imap)
        return imap

    def log(self, logger):
        logger.log('-'*79)
        train_lengths = [len(xseq) for xseq in self.wordseqs_train]
        logger.log('Num train seqs: %d' % len(self.wordseqs_train))
        logger.log('\tAvg length: %d' % stat.mean(train_lengths))
        logger.log('\tMax length: %d' % max(train_lengths))
        logger.log('\tMin length: %d' % min(train_lengths))
        logger.log('\tStd length: %g' % stat.stdev(train_lengths))
        logger.log('Num val seqs: %d' % len(self.wordseqs_val))
        logger.log('Num test seqs: %d' % len(self.wordseqs_test))
        logger.log('')
        logger.log('Num word types: %d (including PAD/UNK)' %
                   len(self.word2x))
        logger.log('Num NER label types: %d (including PAD/UNK)' %
                   len(self.tag2y))
        logger.log('\t%s' % ' '.join(self.tag2y.keys()))
        logger.log('Num RE label types: %d ' %
                   len(self.relation2y))
        logger.log('\t%s' % ' '.join(self.relation2y.keys()))
        logger.log('Num char types: %d (including PAD/UNK)' %
                   len(self.char2c))
        logger.log('\t%s' % ' '.join(self.char2c.keys()))
