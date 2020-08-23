from torchtext import data


def load_dsl_data(text_field, label_field):
    train, dev, test = data.TabularDataset.splits(
        path='data/DSLCC-v2017/', train='DSL-DEV.txt.tok',
        validation='DSL-DEV.txt.tok', test='DSL-TEST-GOLD.txt.tok',
        format='tsv', fields=[('text', text_field), ('label', label_field)]
    )
    text_field.build_vocab(train)
    label_field.build_vocab(train)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_sizes=(32, 256, 256),
        sort_key=lambda x: len(x.text))
    return train_iter, dev_iter, test_iter


if __name__ == "__main__":
    text_field = data.Field(tokenize=list)
    label_field = data.Field()
    train_iter, dev_iter, test_iter = load_dsl_data(text_field, label_field)
    batch = next(iter(dev_iter))
    batch_input, batch_label = batch.text, batch.label
    print(batch_input[0])
    print([text_field.vocab.itos[i] for i in batch_input[0]])
