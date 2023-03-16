def predict(data_loader,
            model,
            loss_fn,
            metric_fns,
            device,
            output_save_path=None,
            log_save_path=None):
    '''
    Run the model on the data_loader, calculate metrics, and log

    Arg(s):
        data_loader : torch Dataloader
            data to test on
        model : torch.nn.Module
            model to run
        loss_fn : module
            loss function
        metric_fns : list[model.metric modules]
            list of metric functions
        device : torch.device
            GPU device
        output_save_path : str or None
            if not None, save model_outputs to save_path
        log_save_path : str or None
            if not None, save metrics to save_path


    Returns :
        log : dict{} of metrics
    '''

    # Hold data for calculating metrics
    outputs = []
    targets = []

    # Ensure model is in eval mode
    if model.training:
        model.eval()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data_loader)):
            if len(item) == 3:
                data, target, path = item
            else:
                data, target = item
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Store outputs and targets
            outputs.append(output)
            targets.append(target)

    # Concatenate predictions and targets
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # Calculate loss
    loss = loss_fn(outputs, targets).item()
    n_samples = len(data_loader.sampler)
    log = {'loss': loss}

    # Calculate predictions based on argmax
    predictions = torch.argmax(outputs, dim=1)

    # Move predictions and target to cpu and convert to numpy to calculate metrics
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Calculate metrics
    log = module_metric.compute_metrics(
        metric_fns=metric_fns,
        prediction=predictions,
        target=targets)

    if output_save_path is not None:
        ensure_dir(os.path.dirname(output_save_path))
        torch.save(outputs, output_save_path)

    if log_save_path is not None:
        ensure_dir(os.path.dirname(log_save_path))
        torch.save(log, log_save_path)

    return_data = {
        'metrics': log,
        'logits': outputs
    }
    return return_data

def predict_with_bump(data_loader,
                      model,
                      target_class_idx,
                      bump_amount,
                      loss_fn,
                      metric_fns,
                      device,
                      output_save_path=None,
                      log_save_path=None):
    '''
    Run the model on the data_loader, calculate metrics, and log

    Arg(s):
        data_loader : torch Dataloader
            data to test on
        model : torch.nn.Module
            model to run
        loss_fn : module
            loss function
        metric_fns : list[model.metric modules]
            list of metric functions
        device : torch.device
        output_save_path : str or None
            if not None, save model_outputs to save_path
        log_save_path : str or None
            if not None, save metrics to save_path

    Returns :
        log : dict{} of metrics
    '''

    # Hold data for calculating metrics
    outputs = []
    targets = []

    # Ensure model is in eval mode
    if model.training:
        model.eval()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data_loader)):
            if len(item) == 3:
                data, target, path = item
            else:
                data, target = item
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Store outputs and targets
            outputs.append(output)
            targets.append(target)

    # Concatenate predictions and targets
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # Adjust output softmax by bump amount
    outputs[:, target_class_idx] += bump_amount

    # Calculate loss
    loss = loss_fn(outputs, targets).item()
    n_samples = len(data_loader.sampler)


    # Calculate predictions based on argmax
    predictions = torch.argmax(outputs, dim=1)

    # Move predictions and target to cpu and convert to numpy to calculate metrics
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Calculate metrics
    log = module_metric.compute_metrics(
        metric_fns=metric_fns,
        prediction=predictions,
        target=targets)

    # Add bump amount to log
    log.update({'loss': loss})
    log.update({'bump_amount': bump_amount})

    if output_save_path is not None:
        ensure_dir(os.path.dirname(output_save_path))
        torch.save(outputs, output_save_path)

    if log_save_path is not None:
        ensure_dir(os.path.dirname(log_save_path))
        torch.save(log, log_save_path)

    return {
        'metrics': log,
        'logits': outputs
    }
