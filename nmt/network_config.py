class NetworkConfig:
    def __init__(self, inference_input_file=None):
        # network
        self.num_units = 1024
        self.num_layers = 2
        self.num_encoder_layers = 4
        self.num_decoder_layers = 4
        self.encoder_type = "bi"
        self.residual = False
        self.time_major = True
        self.num_embeddings_partitions = 0
        # attention mechanisms
        self.attention = "normed_bahdanau"
        self.attention_architecture = "standard"
        self.output_attention = True
        self.pass_hidden_state = True
        # optimizer
        self.optimizer = "sgd"
        self.learning_rate = 1.0
        self.warmup_steps = 0
        self.warmup_scheme = "t2t"
        self.decay_scheme = "luong234"
        self.num_train_steps = 340000
        self.colocate_gradients_with_ops = True
        # initializer
        self.init_op = "uniform"
        self.init_weight = 0.1
        # data
        self.src = "en"
        self.tgt = "fr"
        self.train_prefix = "nmt/wmt15/train.tok.clean.bpe.36000"
        self.dev_prefix = "nmt/wmt15/newstest2013.tok.bpe.36000"
        self.test_prefix = "nmt/wmt15/newstest2015.tok.bpe.36000"
        self.out_dir = "models/en_fr_attention"
        # vocab
        self.vocab_prefix = "nmt/wmt15/vocab.bpe.36000"
        self.embed_prefix = None
        self.sos = "<s>"
        self.eos = "</s>"
        self.share_vocab = False
        self.check_special_token = True
        # sequence length
        self.src_max_len = 50
        self.tgt_max_len = 50
        self.src_max_len_infer = None
        self.tgt_max_len_infer = None
        # default setting
        self.unit_type = "lstm"
        self.forget_bias = 1.0
        self.dropout = 0.1
        self.max_gradient_norm = 5.0
        self.batch_size = 128
        self.steps_per_stats = 100
        self.max_train = 0
        self.num_buckets = 5
        self.num_sampled_softmax = 0
        # SPM
        self.subword_option = "bpe"
        # experimental encoding feature
        self.use_char_encode = False
        # misc
        self.num_gpus = 1
        self.log_device_placement = False
        self.metrics = "bleu"
        self.steps_per_external_eval = None
        self.scope = None
        self.hparams_path = None
        self.random_seed = None
        self.override_loaded_hparams = False
        self.num_keep_ckpts = 5
        self.avg_ckpts = False
        self.language_model = False
        # inference
        self.ckpt = None
        self.inference_input_file = None if inference_input_file is None else 'inference/input/' + inference_input_file
        self.inference_list = None
        self.infer_batch_size = 32
        self.inference_output_file = None if inference_input_file is None else 'inference/output/' + inference_input_file
        self.inference_ref_file = None
        # advance inference args
        self.infer_mode = "beam_search"
        self.beam_width = 10
        self.length_penalty_weight = 1.0
        self.sampling_temperature = 0.0
        self.num_translations_per_input = 1
        # job info
        self.jobid = 0
        self.num_workers = 1
        self.num_inter_threads = 0
        self.num_intra_threads = 0
        # added
        self.num_encoder_residual_layers = None
        self.num_decoder_residual_layers = None
        self.src_vocab_size = None
        self.tgt_vocab_size = None
        self.src_vocab_file = None
        self.tgt_vocab_file = None
        self.num_embeddings_partitions = 0
        self.num_enc_emb_partitions = 0
        self.num_dec_emb_partitions = 0
        self.src_embed_file = None
        self.tgt_embed_file = None
        self.inference_indices = None
