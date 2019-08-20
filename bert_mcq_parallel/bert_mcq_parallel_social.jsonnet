local train_size = 33410;
local batch_size = 4;
local grad_accumulate = 8;
local num_epochs = 9;
local bert_tokenizer = "bert-large-uncased";
local bert_model = "/home/amitra7/multihop/bert_large_uncased_whole_word_masking.tar.gz";
local seed = 42;

{

    "numpy_seed":seed,
    "pytorch_seed":seed,

    "dataset_reader": {
        "type": "bert_mcq_parallel",
        "tokenizer": {
            "type": "bert-multinli",
            "pretrained_model": bert_tokenizer
        },
        "token_indexers": {
            "tokens": {
                "type": "bert-multinli",
                "pretrained_model": bert_tokenizer
            }
        },
        "max_pieces":64
    },

    "train_data_path": "/home/amitra7/multihop/mcq_sc_sim_train.jsonl",
    "validation_data_path": "/home/amitra7/multihop/mcq_sc_sim_dev.jsonl",
    "model": {
        "type": "bert_mcq_parallel",
        "bert_model": bert_model,
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_normal"}]
        ],
        //"dropout": 0.3
    },
    "iterator": {
        "type": "basic",
        "batch_size": batch_size
    },
   "trainer": {

    "optimizer": {
      "type": "bert_adam",
      "weight_decay_rate": 0.009,
      "parameter_groups": [[["bias", "gamma", "beta"], {"weight_decay_rate": 0}]],
      "lr": 2e-6
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": 0.1,
      "num_steps_per_epoch": std.ceil(train_size / batch_size),
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 2,
    "should_log_learning_rate": true,
    "num_epochs": num_epochs,
    "grad_accumulate_epochs": grad_accumulate,
    "cuda_device": 0,
    "grad_norm": 5.0
  }
}