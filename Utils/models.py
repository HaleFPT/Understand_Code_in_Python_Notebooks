import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# ✨ EnhancedModel Class ✨
class EnhancedModel(nn.Module):  #  Inheriting from nn.Module for superpowers!
    """
    A model refined to the utmost clarity and secrecy 
    - Detailed comments with illustrative icons ✨
    - Structural enhancements for potential greatness 
    """

    def __init__(self, model_name, num_classes=1, seq_length=96, pretrained=True):
        super().__init__()  #  Harnessing the power of inheritance

        # --- Encoder Configuration ⚙️ ---
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.attention_probs_dropout_prob = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.max_position_embeddings = 4096 * 2 

        # --- Encoder Instantiation  ---
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                model_name, config=self.config, ignore_mismatched_sizes=True
            )
        else:
            #  Building with clarity
            self.base_encoder = AutoModel.from_config(self.config)

        # --- Extracting Input Dimension  ---
        self.in_dim = self.encoder.config.hidden_size
        print(self.in_dim)

        # --- BiLSTM Layer  ---
        self.bilstm = nn.LSTM(
            self.in_dim,
            self.in_dim,
            num_layers=1,
            dropout=self.config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )

        # --- Output Layer  ---
        self.output_layer = nn.Sequential(
            nn.Linear(self.in_dim * 2, num_classes),
            nn.Sigmoid(),
        )  #  Combining forces for clarity

        # --- Initialization ⚡️ ---
        torch.nn.init.normal_(self.output_layer[0].weight, std=0.02)

    def forward(self, input_ids, attention_mask):
        """
        Guiding data through the model's pathways ️
        """

        # --- Encoder Embeddings  ---
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = encoder_output["last_hidden_state"]

        # --- BiLSTM Processing  ---
        bilstm_output, _ = self.bilstm(last_hidden_state)

        # --- Final Output  ---
        output = self.output_layer(bilstm_output)
        output = output.squeeze(-1)  # ✂️ Trimming the excess
        return output

# --- Testing (Optional) ---
if __name__ == "__main__":
    model = EnhancedModel("roberta-base")
    print(model)
    print(model.config)
    print(model.in_dim)
    print(model.output_layer)
    print(model.forward)
    print(model.forward(torch.ones(1, 96).long(), torch.ones(1, 96).long()))