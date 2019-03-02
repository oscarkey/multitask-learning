import cityscapes
import torch
import torch.nn as nn
from decoders import Decoders
from encoder import Encoder
from losses import MultiTaskLoss


class MultitaskLearner(nn.Module):
    def __init__(self, num_classes):
        super(MultitaskLearner, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoders(num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main(_run):
    train_loader = cityscapes.get_loader_from_dir(_run.config['root_dir_train'], _run.config)
    validation_loader = cityscapes.get_loader_from_dir(_run.config['root_dir_validation'],
                                                       _run.config)

    learner = MultitaskLearner(_run.config['num_classes'])

    device = "cuda:0" if _run.config['gpu'] and torch.cuda.is_available() else "cpu"
    learner.to(device)

    criterion = MultiTaskLoss(_run.config['loss_type'], _run.config['loss_weights'])

    initial_learning_rate = 2.5e-3

    optimizer = torch.optim.SGD(learner.parameters(),
                                lr=initial_learning_rate,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1e4)
    lr_lambda = lambda x: initial_learning_rate * (1 - x / _run.config['max_iter']) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(_run.config['max_iter']):  # loop over the dataset multiple times

        # polynomial learning rate decay
        lr_scheduler.step()

        running_loss = 0.0

        # training loop
        for i, data in enumerate(train_loader, 0):
            inputs, semantic_labels, instance_centroid, instance_mask = data

            inputs = inputs.to(device)
            semantic_labels = semantic_labels.to(device)
            instance_centroid = instance_centroid.to(device)
            instance_mask = instance_mask.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_semantic, output_instance, output_depth = learner(inputs)
            loss, task_losses = criterion((output_semantic, output_instance, output_depth),
                             semantic_labels, instance_centroid, instance_mask)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

        # validation loop
        with torch.no_grad(): # exclude gradients
            for i, data in enumerate(validation_loader, 0):
                inputs, semantic_labels, instance_centroid, instance_mask = data

                # forward + backward + optimize
                output_semantic, output_instance, output_depth = learner(inputs.float())
                val_loss, val_task_loss = criterion((output_semantic, output_instance, output_depth),
                                    semantic_labels.long(), instance_centroid, instance_mask)

                # calculate accuracy measures

                # segmentation IoU
                iou = 0 # calculated for each class separately and averaged over all classes
                max_args = torch.argmax(output_semantic, dim=1) # find the predicted class for each pixel

                for batch in range(_run.config['batch_size']):
                    iou_dict = {str(i): {'intersection': 0, 'union': 0} for i in range(_run.config['num_classes'])}
                    for height in range(128): # TODO: get height and width from main.py
                        for width in range(256):
                            gt_class = semantic_labels.long()[batch][height][width].item()
                            predicted_class = max_args[batch][height][width].item()
                            if predicted_class == gt_class:
                                # Add to intersection and union
                                iou_dict[str(gt_class)]['intersection'] += 1
                                iou_dict[str(gt_class)]['union'] += 1
                            else:
                                # Add only to the union of each
                                iou_dict[str(predicted_class)]['union'] += 1
                                if gt_class != 255:
                                    iou_dict[str(gt_class)]['union'] += 1
                    # Average across all classes
                    for key, dict in iou_dict.items():
                        if dict['union'] != 0:
                            iou += dict['intersection'] / (dict['union']* _run.config['num_classes'])
                iou = iou / _run.config['batch_size']

                # instance mean error
                instance_error = val_task_loss[1].item()

                # inverse depth mean error
                depth_error = val_task_loss[2]

                print('Batch iou %', iou * 100)
                print('Batch instance_error', instance_error)
                print('Batch depth_error', depth_error)

                # print statistics
                running_loss += val_loss.item()
                # if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0
