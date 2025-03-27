def check_forward_pass(model, train_loader, num_classes=200):
    sample_batch, _ = next(iter(train_loader))  # Get one batch
    sample_batch = sample_batch.cuda()  # If you're using a GPU

    output = model(sample_batch)
    assert output.shape == (train_loader.batch_size, num_classes), "Forward Pass Failed"
    print("Forward Pass works!")
