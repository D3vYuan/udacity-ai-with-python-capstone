import torch
import time
from workspace_utils import active_session

class ModelTraining():
    def __init__(self, model, optimizer, criterion, epochs, is_gpu, progress_intervals=20):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.progress_intervals = progress_intervals
        self.device = torch.device("cuda" if is_gpu and torch.cuda.is_available() else "cpu")

    def batch_training(self, inputs, labels):
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        logps = self.model.forward(inputs)
        loss = self.criterion(logps, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate_training(self, test_loader):
        total_test_loss = 0
        total_test_accuracy = 0

        with torch.no_grad():
            self.model.eval()
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterion(logps, labels)

                total_test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                total_test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        self.model.train()
        return total_test_loss, total_test_accuracy

    def train(self, train_loader, test_loader):
        train_losses = []
        test_losses = []
        total_train_loss = 0
        start_time = None

        try:
            self.model.to(self.device)
            print("== training in progress ===")
            with active_session():
                for epoch in range(self.epochs):
                    steps = 0
                    for inputs, labels in train_loader:
                        if start_time is None:
                            start_time = time.time()
                        steps += 1
                        total_train_loss += self.batch_training(inputs, labels)

                        if steps % self.progress_intervals == 0:
                            end_time = time.time()
                            total_test_loss, total_test_accuracy = self.validate_training(test_loader)
                            
                            train_loss = total_train_loss/self.progress_intervals
                            test_loss = total_test_loss / len(test_loader)
                            test_accuracy = total_test_accuracy/len(test_loader)
                            
                            train_losses.append(train_loss)
                            test_losses.append(test_loss)
                            
                            training_duration = end_time - start_time
                            start_time = None
                            
                            train_progress=f"Epoch {epoch+1}/{self.epochs}.. Step #{steps}.. " \
                                f"Train loss: {train_loss:.3f}.. "  \
                                f"Test loss: {test_loss:.3f}.. "  \
                                f"Test accuracy: {test_accuracy:.3f}.. "  \
                                f"Took: {training_duration:.2f}s "
                            print(train_progress)
                            
                            total_train_loss = 0
                    
                print("== training completed ===")
        except Exception as e:
            print(f"Model NOT trained due to {e}")
        finally:
            return self.model, self.optimizer, train_losses, test_losses
