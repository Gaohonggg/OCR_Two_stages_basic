import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
from sklearn.model_selection import train_test_split

import os
import random
import time
import timm
import xml.etree.ElementTree as ET


def encode(label, char_to_idx, max_label_len):
    encoded_labels = torch.tensor(
        [char_to_idx[char] for char in label],
        dtype= torch.float32
    )

    label_len = len(encoded_labels)
    lengths = torch.tensor(
        label_len,
        dtype= torch.int32
    )

    padded_labels = F.pad(
        encoded_labels,
        (0, max_label_len - label_len),
        value=0
    )

    return padded_labels, lengths

def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None

        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]

                if char != blank_char:
                    if char != prev_char or prev_char == blank_char:
                        decoded_label.append( char )
                prev_char = char
        
        decoded_sequences.append("".join(decoded_label))
    
    return decoded_sequences

class STRDataset(Dataset):
    def __init__(self, X, y, char_to_idx, max_label_len,
                 label_encoder=None, transform=None):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len( self.img_paths )
    
    def __getitem__(self, idx):
        label = self.labels[ idx ]
        img_path = self.img_paths[ idx ]
        img = Image.open( img_path ).convert("RGB")

        img = self.transform( img )
        encoded_label, label_len = self.label_encoder(
            label, self.char_to_idx, self.max_label_len
        )

        return img, encoded_label, label_len


class CRNN(nn.Module):
    def __init__(self,vocab_size, hidden_size, n_layers, 
                 dropout=0.2, unfreeze_layers=3):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet152", in_chans=1, pretrained=True)
        modules = list( backbone.children() )[:-2]
        modules.append( nn.AdaptiveAvgPool2d((1,None)) )
        self.backbone = nn.Sequential( *modules )

        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True
        
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout( dropout )
        )

        self.gru = nn.GRU(
            512, hidden_size,
            num_layers= n_layers,
            bidirectional= True,
            batch_first= True,
            dropout= dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm( hidden_size*2 )
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, vocab_size),
            nn.LogSoftmax(dim=2)
        )
    
    @torch.autocast( device_type="cuda" )
    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view( x.size(0), x.size(1), -1)
        x = self.mapSeq( x )
        x, _ = self.gru( x )
        x = self.layer_norm(x)
        x = self.out( x )
        x = x.permute(1, 0 ,2)

        return x 

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []

    with torch.no_grad():
        for inputs, labels, labels_len in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            outputs = model(inputs)
            logits_lens = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs, labels, logits_lens, labels_len)
            losses.append(loss.item())

    loss = sum(losses) / len(losses)
    return loss


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        start = time.time()
        batch_train_losses = []

        model.train()
        for idx, (inputs, labels, labels_len) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            logits_lens = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs, labels, logits_lens, labels_len)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(
            f"EPOCH {epoch + 1}: \tTrain loss: {train_loss:.4f} \tVal loss: {val_loss:.4f} \tTime: {time.time() - start:.2f} seconds"
        )

        scheduler.step()
    return train_losses, val_losses


if __name__ == "__main__":
    root_dir = "ocr_dataset"

    img_paths = []
    labels = []

    with open( os.path.join(root_dir,"labels.txt"), "r") as f:
        for pattern in f:
            img_paths.append( pattern.strip().split("\t")[0] )
            labels.append( pattern.strip().split("\t")[1] )

    letters = [char.split(".")[0].lower() for char in labels]
    letters = "".join(letters)
    letters = sorted( list( set(list(letters)) ) )
    
    chars = "".join( letters )
    blank_char = "-"
    chars += blank_char

    vocab_size = len( chars )

    char_to_idx = {
        char : idx + 1 for idx, char in enumerate( sorted(chars) )
    }

    idx_to_char = {
        idx : char for char, idx in char_to_idx.items()
    }

    max_label_len = max( [len(label) for label in labels] )

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize( (100,420) ),
                transforms.ColorJitter(
                    brightness= 0.5,
                    contrast= 0.5,
                    saturation= 0.5,
                ),
                transforms.Grayscale(
                    num_output_channels= 1,
                ),
                transforms.GaussianBlur(3),
                transforms.RandomAffine(
                    degrees= 1,
                    shear= 1,
                ),
                transforms.RandomPerspective(
                    distortion_scale= 0.3,
                    p= 0.5,
                    interpolation= 3,
                ),
                transforms.RandomRotation(degrees= 2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ),(0.5, )),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize( (100,420) ),
                transforms.Grayscale( num_output_channels= 1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ),(0.5, )),
            ]
        ),
    }

    seed = 0
    val_size = 0.1
    test_size = 0.1
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        img_paths, labels,
        test_size= val_size,
        random_state= seed,
        shuffle= is_shuffle
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size= test_size,
        random_state= seed,
        shuffle= is_shuffle
    )

    train_dataset = STRDataset(
        X_train, y_train,
        char_to_idx= char_to_idx,
        max_label_len= max_label_len,
        label_encoder= encode,
        transform= data_transforms["train"]
    )
    val_dataset = STRDataset(
        X_val, y_val,
        char_to_idx= char_to_idx,
        max_label_len= max_label_len,
        label_encoder= encode,
        transform= data_transforms["val"]
    )
    test_dataset = STRDataset(
        X_test, y_test,
        char_to_idx= char_to_idx,
        max_label_len= max_label_len,
        label_encoder= encode,
        transform= data_transforms["val"]
    )

    train_batch_size = 64
    test_batch_size = 64*2

    train_loader = DataLoader(
        train_dataset,
        batch_size= train_batch_size,
        shuffle= True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size= test_batch_size,
        shuffle= False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size= test_batch_size,
        shuffle= False
    )

    hidden_size = 256
    n_layers = 3
    dropout_prob = 0.2
    unfreeze_layers = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CRNN(
        vocab_size= vocab_size,
        hidden_size= hidden_size,
        n_layers= n_layers,
        dropout= dropout_prob,
        unfreeze_layers= unfreeze_layers
    ).to( device )

    # Hyperparameters
    epochs = 50
    lr = 5e-4
    weight_decay = 1e-5
    scheduler_step_size = int(epochs * 0.5)

    # Criterion
    criterion = nn.CTCLoss(
        blank=char_to_idx[blank_char],
        zero_infinity=True,
        reduction="mean",
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=0.1
    )

    # Training
    train_losses, val_losses = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs
    )

    val_loss = evaluate(model, val_loader, criterion, device)
    test_loss = evaluate(model, test_loader, criterion, device)

    print("Evaluation on val / test dataset")
    print("Val loss: ", val_loss)
    print("Test loss: ", test_loss)

    # Save model
    save_model_path = "ocr_crnn.pt"
    torch.save(model.state_dict(), save_model_path)
  