import numpy as np
import torch
from torch.optim import Optimizer
from torchvision.transforms import transforms

import checkpointing
import cityscapes
from losses import MultiTaskLoss
from model import MultitaskLearner


def main(_run):
    train_loader, validation_loader = _create_dataloaders(_run.config)

    learner = MultitaskLearner(num_classes=_run.config['num_classes'], enabled_tasks=_run.config['enabled_tasks'],
                               loss_uncertainties=_run.config['loss_uncertainties'],
                               pre_train_encoder=_run.config['pre_train_encoder'],
                               aspp_dilations=_run.config['aspp_dilations'], resnet_type=_run.config['resnet_type'])

    device = "cuda:0" if _run.config['gpu'] and torch.cuda.is_available() else "cpu"
    learner.to(device)

    use_adam = _run.config['use_adam']
    reduce_lr_on_plateau = _run.config['reduce_lr_on_plateau']
    lr_plateau_scheduler = None
    lr_lambda_scheduler = None
    if use_adam:
        optimizer = torch.optim.Adam(learner.parameters(), lr=_run.config['learning_rate'],
                                     weight_decay=_run.config['weight_decay'])
    else:
        optimizer = torch.optim.SGD(learner.parameters(), lr=_run.config['initial_learning_rate'], momentum=0.9,
                                    nesterov=True, weight_decay=_run.config['weight_decay'])

        if not reduce_lr_on_plateau:
            lr_lambda = lambda x: (1 - x / _run.config['max_iter']) ** 0.9
            lr_lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if reduce_lr_on_plateau:
        lr_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    restore_run_id = _run.config['restore_sacred_run']
    if restore_run_id != -1:
        epoch, model_state_dict, optimizer_state_dict = checkpointing.load_state(_run, restore_run_id)
        learner.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        _run.run_logger.info('Restored from sacred run {} at epoch {}'.format(restore_run_id, epoch))
    else:
        epoch = 0

    criterion = MultiTaskLoss(_run.config['loss_type'], _get_uncertainties(_run.config, learner),
                              _run.config['enabled_tasks'])

    if _run.config['validate_only']:
        # The user may want to load a previous experiment from Sacred, validate it, and exit.
        _validate(_run, device, validation_loader, learner, criterion, epoch)
        return

    iterations = 0
    while iterations < _run.config['max_iter']:

        # polynomial learning rate decay
        # print(f'Learning rate: {lr_scheduler.get_lr()}')

        num_training_batches = 0

        running_loss = 0.0
        training_semantic_loss = 0
        training_instance_loss = 0
        training_depth_loss = 0

        # Training loop
        for i, data in enumerate(train_loader, 0):
            inputs, semantic_labels, instance_centroid, instance_mask, depth, depth_mask = data

            learner.set_output_size(inputs.shape[2:])

            # Keep count of number of batches
            num_training_batches += 1

            inputs = inputs.to(device)
            semantic_labels = semantic_labels.to(device)
            instance_centroid = instance_centroid.to(device)
            instance_mask = instance_mask.to(device)
            depth = depth.to(device)
            depth_mask = depth_mask.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            output = learner(inputs)
            loss, task_loss = criterion(output, semantic_labels, instance_centroid, instance_mask, depth, depth_mask)
            loss.backward()
            optimizer.step()

            if lr_lambda_scheduler is not None:
                lr_lambda_scheduler.step()

            # Print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            logvars = learner.get_loss_params()
            print('[%d, %5d] Training loss: %.3f - (%.3f, %.3f, %.3f)' % (
                epoch + 1, i + 1, running_loss, logvars[0].item(), logvars[1].item(), logvars[2].item()))
            running_loss = 0.0

            # compute gradient of an output pixel with respect to input
            # for class 0
            # inputs.requires_grad = True
            # learner.zero_grad()
            # output, _, _ = learner(inputs)
            # output = F.softmax(output, dim=1)
            # output[0, 0, 0, 0].backward(retain_graph=True)
            # print(inputs.grad)

            training_semantic_loss += task_loss[0]
            training_instance_loss += task_loss[1]
            training_depth_loss += task_loss[2]

            iterations += 1

        # Save statistics to Sacred
        _run.log_scalar('training_semantic_loss', training_semantic_loss / num_training_batches, epoch)
        _run.log_scalar('training_instance_loss', training_instance_loss / num_training_batches, epoch)
        _run.log_scalar('training_depth_loss', training_depth_loss / num_training_batches, epoch)

        _run.log_scalar('learning_rate', _get_learning_rate(optimizer))

        # print(f'Training losses: {training_semantic_loss / num_training_batches, training_instance_loss / num_training_batches, training_depth_loss / num_training_batches}')

        if _run.config['validate_epochs'] != 0 and ((epoch + 1) % _run.config['validate_epochs'] == 0 or epoch == 0):
            loss = _validate(_run=_run, device=device, validation_loader=validation_loader, learner=learner,
                             criterion=criterion, epoch=epoch)
            if reduce_lr_on_plateau:
                lr_plateau_scheduler.step(loss)

        if _run.config['model_save_epochs'] != 0 and (epoch + 1) % _run.config['model_save_epochs'] == 0:
            checkpointing.save_model(_run, learner, optimizer, epoch, iterations)

        epoch += 1


def _get_learning_rate(optimizer: Optimizer):
    assert len(optimizer.state_dict()['param_groups']) == 1
    return optimizer.state_dict()['param_groups'][0]['lr']


def _create_dataloaders(config):
    train_loader = cityscapes.get_loader_from_dir(config['root_dir_train'], config,
                                                  transform=_get_training_transforms(config))

    validation_loader = cityscapes.get_loader_from_dir(config['root_dir_validation'], config)

    assert len(train_loader.dataset) >= 3, 'Must have at least 3 train images (had {})'.format(
        len(train_loader.dataset))
    if config['validate_epochs'] >= 1:
        assert len(validation_loader.dataset) >= 3, 'Must have at least 3 validation images(had {})'.format(
            len(validation_loader.dataset))

    return train_loader, validation_loader


def _get_training_transforms(config):
    transform_list = []
    if config['crop']:
        assert len(config['crop_size']) == 2, 'Wrong crop size' + config['crop_size']
        transform_list.append(cityscapes.RandomCrop(config['crop_size']))

    if config['flip']:
        transform_list.append(cityscapes.RandomHorizontalFlip())

    if len(transform_list) > 0:
        return transforms.Compose(transform_list)
    else:
        return cityscapes.NoopTransform()


def _get_uncertainties(config, learner: MultitaskLearner):
    if config['loss_type'] == 'learned':
        return learner.get_loss_params()
    elif config['loss_type'] == 'fixed':
        return config['loss_uncertainties']
    else:
        raise ValueError('Unknown loss_type {}'.format(config["loss_type"]))


def _validate(_run, device, validation_loader, learner, criterion, epoch) -> float:
    val_total_loss = 0
    val_semantic_loss = 0
    val_instance_loss = 0
    val_depth_loss = 0
    val_iou = 0

    num_val_batches = 0

    # Validation loop
    with torch.no_grad():  # Exclude gradients
        for i, data in enumerate(validation_loader, 0):
            inputs, semantic_labels, instance_centroid, instance_mask, depth, depth_mask = data

            learner.set_output_size(inputs.shape[2:])

            inputs = inputs.to(device)
            semantic_labels = semantic_labels.to(device)
            instance_centroid = instance_centroid.to(device)
            instance_mask = instance_mask.to(device)
            depth = depth.to(device)
            depth_mask = depth_mask.to(device)

            # Keep count of number of batches
            num_val_batches += 1

            # Forward + backward + optimize
            output_semantic, output_instance, output_depth = learner(inputs.float())
            val_loss, val_task_loss = criterion((output_semantic, output_instance, output_depth),
                                                semantic_labels.long(), instance_centroid, instance_mask, depth,
                                                depth_mask)

            # TODO: this batch size might break
            batch_size = semantic_labels.shape[0]

            # Calculate accuracy measures
            # Segmentation IoU
            # Only compute IoU if semantic segmentation is enabled.
            batch_iou = 0
            if _run.config['enabled_tasks'][0]:
                for image_index in range(batch_size):
                    batch_iou += _compute_image_iou(semantic_labels[image_index], output_semantic[image_index],
                                                    _run.config['num_classes'])

            # instance mean error
            instance_error = val_task_loss[1]

            # inverse depth mean error
            depth_error = val_task_loss[2]

            # print('Batch iou %', batch_iou * 100)
            # print('Batch instance_error', instance_error)
            # print('Batch depth_error', depth_error)

            # Print every 2000 mini-batches
            # if i % 2000 == 1999:
            print('[%d, %5d] Validation loss: %.3f' % (epoch + 1, i + 1, val_loss.item()))

            val_total_loss += val_loss.item()
            val_semantic_loss += val_task_loss[0]
            val_instance_loss += val_task_loss[1]
            val_depth_loss += val_task_loss[2]
            val_iou += batch_iou / batch_size

    # save statistics to Sacred
    _run.log_scalar('val_semantic_loss', val_semantic_loss / num_val_batches, epoch)
    # _run.run_logger.debug('val_semantic_loss', val_semantic_loss / num_val_batches)
    _run.log_scalar('val_instance_loss', val_instance_loss / num_val_batches, epoch)
    # _run.run_logger.debug('val_instance_loss', val_instance_loss / num_val_batches, epoch)
    _run.log_scalar('val_depth_loss', val_depth_loss / num_val_batches, epoch)
    # _run.run_logger.debug('val_depth_loss', val_depth_loss / num_val_batches, epoch)

    _run.log_scalar('val_iou', val_iou / num_val_batches, epoch)
    # _run.run_logger.debug('val_iou', val_iou / num_val_batches, epoch)

    if _run.config['loss_type'] == 'learned':
        _log_loss_uncertainties_and_weights(_run, epoch, learner)

    return val_total_loss / num_val_batches


def _log_loss_uncertainties_and_weights(_run, epoch, learner):
    sem_uncertainty = learner.get_loss_params()[0].item()
    inst_uncertainty = learner.get_loss_params()[1].item()
    depth_uncertainty = learner.get_loss_params()[2].item()

    # Convert from uncertainty = log (sigma^2) into the actual weights of the losses
    sem_weight = np.exp(-sem_uncertainty)
    inst_weight = 0.5 * np.exp(-inst_uncertainty)
    depth_weight = 0.5 * np.exp(-depth_uncertainty)

    _run.log_scalar('S_semantic', sem_uncertainty, epoch)
    _run.log_scalar('S_instance', inst_uncertainty, epoch)
    _run.log_scalar('S_depth', depth_uncertainty, epoch)

    print('S: (%.5f, %.5f, %.5f)' % (sem_uncertainty, inst_uncertainty, depth_uncertainty))

    _run.log_scalar('weight_semantic', sem_weight, epoch)
    _run.log_scalar('weight_instance', inst_weight, epoch)
    _run.log_scalar('weight_depth', depth_weight, epoch)

    print('Weights: (%.5f, %.5f, %.5f)' % (sem_weight, inst_weight, depth_weight))

    sem_var = np.exp(sem_uncertainty)
    inst_var = np.exp(inst_uncertainty)
    depth_var = np.exp(depth_uncertainty)

    _run.log_scalar('var_semantic', sem_var, epoch)
    _run.log_scalar('var_instance', inst_var, epoch)
    _run.log_scalar('var_depth', depth_var, epoch)

    print()


def _compute_image_iou(truth, output_softmax, num_classes: int):
    # Convert the softmax to the id of the class.
    output_classes = torch.argmax(output_softmax, dim=0)

    class_count = 0
    iou = 0.0
    for c in range(num_classes):
        # Create tensors with 1 for every pixel labelled with this class, and 0 otherwise. We then
        # add these tensors. The result has 2 for the intersection, and 1 or 2 for the union.

        truth_for_class = torch.where(truth == c, torch.ones_like(truth, dtype=torch.int),
                                      torch.zeros_like(truth, dtype=torch.int))

        output_for_class = torch.where(output_classes == c, torch.ones_like(output_classes, dtype=torch.int),
                                       torch.zeros_like(output_classes, dtype=torch.int))

        result = truth_for_class + output_for_class
        # View in 1D as bincount only supports 1D.
        # We expect values 0, 1, 2 for no object, one object and both objects respectively.
        counts = torch.bincount(result.view(-1), minlength=3)

        assert counts.size(0) == 3, 'Wrong number of bins: {}'.format(counts)

        intersection = counts[2].item()
        union = counts[1].item() + counts[2].item()

        if union > 0:
            class_count += 1
            iou += intersection / union

    return iou / class_count
