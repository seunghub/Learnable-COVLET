import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

### For train_test
def train_and_test(max_epoch, train_loader, test_loader, model, opt, scheduler, device, covlet=None):
    
    if covlet is not None:
        print(f'INITIAL SCALE : {list(covlet.parameters())[1]}')
    
    criterion = torch.nn.CrossEntropyLoss()
    best_acc, best_rec, best_prc = 0,0,0

    for epoch in range(1, max_epoch + 1):
        model.train()
        if covlet is not None:
            covlet.train()
        train_true_y_list = []
        train_pred_y_list = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            target = target.long().to(device)
            opt.zero_grad()

            if covlet is not None:
                CMD = covlet(data)
                output = model(CMD)
            else:
                output = model(data)

            loss = criterion(output, target)

            if epoch %200==0:
                pred_y = torch.argmax(output.detach(), axis=1)
                train_pred_y_list.extend(pred_y.tolist())
                train_true_y_list.extend(target.tolist())

            loss.backward()
            opt.step()
            scheduler.step()


        if (epoch > 1000 and (epoch % 5 == 0)) or (epoch%200 == 0):

            model.eval()
            if covlet is not None:
                covlet.eval()
            test_true_y_list = []
            test_pred_y_list = []
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.float().to(device)
                    target = target.long().to(device)
                    if covlet is not None:
                        CMD = covlet(data)
                        output = model(CMD)
                    else:
                        output = model(data)
                    pred_y = torch.argmax(output.detach(), axis=1)
                    test_pred_y_list.extend(pred_y.tolist())
                    test_true_y_list.extend(target.tolist())
                
                test_acc = accuracy_score(test_true_y_list, test_pred_y_list)
                test_rec = recall_score(test_true_y_list, test_pred_y_list,average='macro')
                test_prc = precision_score(test_true_y_list, test_pred_y_list,average='macro')

            if epoch > 1000 and epoch %5 ==0:
                    if test_acc > best_acc:
                        best_acc = test_acc
                        best_rec = test_rec
                        best_prc = test_prc

    return best_acc, best_rec, best_prc