local train_size = 129974;
local batch_size = 128;
local num_epochs = 60;
local seed = 18551;
local encoder_hidden_size = 300;
local projection_hidden_size = 300;
local key_projection_feedforward_hidden_size = 300;
local inference_encoder_hidden_size = 300;
local link_key_encoder_hidden_size = 300;
local key_compare_feedforward_hidden_size = 600;
local output_feedforward_hidden_size = 300;
local learning_rate = 2e-4;
{

  "numpy_seed":seed,
  "pytorch_seed":seed,

  "dataset_reader": {
    "type": "mac_seq_dataset_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
   },
  "train_data_path": "/home/amitra7/multihop/coverage_mcq_abductive_train.jsonl",
  "validation_data_path": "/home/amitra7/multihop/coverage_mcq_abductive_dev.jsonl",
  "iterator": {
        "type": "basic",
        "batch_size": batch_size
  },

  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": learning_rate
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 2,
    "num_epochs": num_epochs,
    "grad_norm": 10.0,
    "patience": 10,
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": 0.2,
      "num_steps_per_epoch": std.ceil(train_size / batch_size),
    },
  },

  "model": {
    "type": "mac_seq_model",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "encoder": {
      "type": "lstm",
        "input_size": 300,
        "hidden_size": encoder_hidden_size,
        "num_layers": 1,
        "bidirectional": true
    },
    "projection_feedforward": {
        "input_dim": 8*encoder_hidden_size,
        "hidden_dims": projection_hidden_size,
        "num_layers": 1,
        "activations": "relu"
    },
    "key_projection_feedforward": {
        "input_dim": 8*encoder_hidden_size,
        "hidden_dims": key_projection_feedforward_hidden_size,
        "num_layers": 1,
        "activations": "relu"
    },
    "inference_encoder": {
        "type": "lstm",
        "input_size": projection_hidden_size,
        "hidden_size": inference_encoder_hidden_size,
        "num_layers": 1,
        "bidirectional": true
    },
    "link_key_encoder": {
        "type": "lstm",
        "input_size": key_projection_feedforward_hidden_size,
        "hidden_size": link_key_encoder_hidden_size,
        "num_layers": 1,
        "bidirectional": false
    },
    "key_compare_feedforward": {
      "input_dim": 2*(inference_encoder_hidden_size + link_key_encoder_hidden_size),
      "num_layers": 1,
      "hidden_dims": key_compare_feedforward_hidden_size,
      "activations": "relu",
    },
     "output_feedforward": {
        "input_dim": 2*key_compare_feedforward_hidden_size + 4*inference_encoder_hidden_size,
        "num_layers": 1,
        "hidden_dims": output_feedforward_hidden_size,
        "activations": "relu",
        "dropout": 0.5
     },
     "output_logit": {
        "input_dim": output_feedforward_hidden_size,
        "num_layers": 1,
        "hidden_dims": 1,
        "activations": "linear"
     },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   }
}
