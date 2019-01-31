# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from . import inference
from . import train
from . import network_config
from .utils import evaluation_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

FLAG = None
INFERENCE_KEYS = ["src_max_len_infer", "tgt_max_len_infer", "subword_option",
                  "infer_batch_size", "beam_width",
                  "length_penalty_weight", "sampling_temperature",
                  "num_translations_per_input", "infer_mode"]


def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        embed_prefix=flags.embed_prefix,
        out_dir=flags.out_dir,

        # Networks
        num_units=flags.num_units,
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,
        num_embeddings_partitions=flags.num_embeddings_partitions,

        # Attention mechanisms
        attention=flags.attention,
        attention_architecture=flags.attention_architecture,
        output_attention=flags.output_attention,
        pass_hidden_state=flags.pass_hidden_state,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        decay_scheme=flags.decay_scheme,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
        num_sampled_softmax=flags.num_sampled_softmax,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,

        # Advanced inference arguments
        infer_mode=flags.infer_mode,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        sampling_temperature=flags.sampling_temperature,
        num_translations_per_input=flags.num_translations_per_input,

        # Vocab
        sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,
        subword_option=flags.subword_option,
        check_special_token=flags.check_special_token,
        use_char_encode=flags.use_char_encode,

        # Misc
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(","),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        override_loaded_hparams=flags.override_loaded_hparams,
        num_keep_ckpts=flags.num_keep_ckpts,
        avg_ckpts=flags.avg_ckpts,
        language_model=flags.language_model,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
    )


def _add_argument(hparams, key, value, update=True):
    """Add an argument to hparams; if exists, change the value if update==True."""
    if hasattr(hparams, key):
        if update:
            setattr(hparams, key, value)
    else:
        hparams.add_hparam(key, value)


def extend_hparams(hparams):
    """Add new arguments to hparams."""
    # Sanity checks
    if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
        raise ValueError("For bi, num_encoder_layers %d should be even" %
                         hparams.num_encoder_layers)
    if (hparams.attention_architecture in ["gnmt"] and
            hparams.num_encoder_layers < 2):
        raise ValueError("For gnmt attention architecture, "
                         "num_encoder_layers %d should be >= 2" %
                         hparams.num_encoder_layers)
    if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
        raise ValueError("subword option must be either spm, or bpe")
    if hparams.infer_mode == "beam_search" and hparams.beam_width <= 0:
        raise ValueError("beam_width must greater than 0 when using beam_search"
                         "decoder.")
    if hparams.infer_mode == "sample" and hparams.sampling_temperature <= 0.0:
        raise ValueError("sampling_temperature must greater than 0.0 when using"
                         "sample decoder.")

    # Different number of encoder / decoder layers
    assert hparams.num_encoder_layers and hparams.num_decoder_layers
    if hparams.num_encoder_layers != hparams.num_decoder_layers:
        hparams.pass_hidden_state = False
        utils.print_out("Num encoder layer %d is different from num decoder layer"
                        " %d, so set pass_hidden_state to False" % (
                            hparams.num_encoder_layers,
                            hparams.num_decoder_layers))

    # Set residual layers
    num_encoder_residual_layers = 0
    num_decoder_residual_layers = 0
    if hparams.residual:
        if hparams.num_encoder_layers > 1:
            num_encoder_residual_layers = hparams.num_encoder_layers - 1
        if hparams.num_decoder_layers > 1:
            num_decoder_residual_layers = hparams.num_decoder_layers - 1

        if hparams.encoder_type == "gnmt":
            # The first unidirectional layer (after the bi-directional layer) in
            # the GNMT encoder can't have residual connection due to the input is
            # the concatenation of fw_cell and bw_cell's outputs.
            num_encoder_residual_layers = hparams.num_encoder_layers - 2

            # Compatible for GNMT models
            if hparams.num_encoder_layers == hparams.num_decoder_layers:
                num_decoder_residual_layers = num_encoder_residual_layers
    _add_argument(hparams, "num_encoder_residual_layers",
                  num_encoder_residual_layers)
    _add_argument(hparams, "num_decoder_residual_layers",
                  num_decoder_residual_layers)

    # Language modeling
    if getattr(hparams, "language_model", None):
        hparams.attention = ""
        hparams.attention_architecture = ""
        hparams.pass_hidden_state = False
        hparams.share_vocab = True
        hparams.src = hparams.tgt
        utils.print_out("For language modeling, we turn off attention and "
                        "pass_hidden_state; turn on share_vocab; set src to tgt.")

    ## Vocab
    # Get vocab file names first
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    else:
        raise ValueError("hparams.vocab_prefix must be provided.")

    # Source vocab
    check_special_token = getattr(hparams, "check_special_token", True)
    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)

    # Target vocab
    if hparams.share_vocab:
        utils.print_out("  using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
            tgt_vocab_file,
            hparams.out_dir,
            check_special_token=check_special_token,
            sos=hparams.sos,
            eos=hparams.eos,
            unk=vocab_utils.UNK)
    _add_argument(hparams, "src_vocab_size", src_vocab_size)
    _add_argument(hparams, "tgt_vocab_size", tgt_vocab_size)
    _add_argument(hparams, "src_vocab_file", src_vocab_file)
    _add_argument(hparams, "tgt_vocab_file", tgt_vocab_file)

    # Num embedding partitions
    num_embeddings_partitions = getattr(hparams, "num_embeddings_partitions", 0)
    _add_argument(hparams, "num_enc_emb_partitions", num_embeddings_partitions)
    _add_argument(hparams, "num_dec_emb_partitions", num_embeddings_partitions)

    # Pretrained Embeddings
    _add_argument(hparams, "src_embed_file", "")
    _add_argument(hparams, "tgt_embed_file", "")
    if getattr(hparams, "embed_prefix", None):
        src_embed_file = hparams.embed_prefix + "." + hparams.src
        tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

        if tf.gfile.Exists(src_embed_file):
            utils.print_out("  src_embed_file %s exist" % src_embed_file)
            hparams.src_embed_file = src_embed_file

            utils.print_out(
                "For pretrained embeddings, set num_enc_emb_partitions to 1")
            hparams.num_enc_emb_partitions = 1
        else:
            utils.print_out("  src_embed_file %s doesn't exist" % src_embed_file)

        if tf.gfile.Exists(tgt_embed_file):
            utils.print_out("  tgt_embed_file %s exist" % tgt_embed_file)
            hparams.tgt_embed_file = tgt_embed_file

            utils.print_out(
                "For pretrained embeddings, set num_dec_emb_partitions to 1")
            hparams.num_dec_emb_partitions = 1
        else:
            utils.print_out("  tgt_embed_file %s doesn't exist" % tgt_embed_file)

    # Evaluation
    for metric in hparams.metrics:
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        tf.gfile.MakeDirs(best_metric_dir)
        _add_argument(hparams, "best_" + metric, 0, update=False)
        _add_argument(hparams, "best_" + metric + "_dir", best_metric_dir)

        if getattr(hparams, "avg_ckpts", None):
            best_metric_dir = os.path.join(hparams.out_dir, "avg_best_" + metric)
            tf.gfile.MakeDirs(best_metric_dir)
            _add_argument(hparams, "avg_best_" + metric, 0, update=False)
            _add_argument(hparams, "avg_best_" + metric + "_dir", best_metric_dir)

    return hparams


def create_or_load_hparams(
        out_dir, default_hparams, hparams_path, save_hparams=True):
    """Create hparams or load hparams from out_dir."""
    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(
            hparams, hparams_path)
    else:
        hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)
    hparams = extend_hparams(hparams)

    # Save HParams
    if save_hparams:
        utils.save_hparams(out_dir, hparams)
        for metric in hparams.metrics:
            utils.save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams)

    # Print HParams
    utils.print_hparams(hparams)
    return hparams


def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
    """Run main."""
    # Job
    jobid = flags.jobid
    num_workers = flags.num_workers
    utils.print_out("# Job id %d" % jobid)

    # GPU device
    utils.print_out(
        "# Devices visible to TensorFlow: %s" % repr(tf.Session().list_devices()))

    # Random
    random_seed = flags.random_seed
    if random_seed is not None and random_seed > 0:
        utils.print_out("# Set random seed to %d" % random_seed)
        random.seed(random_seed + jobid)
        np.random.seed(random_seed + jobid)

    # Model output directory
    out_dir = flags.out_dir
    if out_dir and not tf.gfile.Exists(out_dir):
        utils.print_out("# Creating output directory %s ..." % out_dir)
        tf.gfile.MakeDirs(out_dir)

    # Load hparams.
    loaded_hparams = False
    if flags.ckpt:  # Try to load hparams from the same directory as ckpt
        ckpt_dir = os.path.dirname(flags.ckpt)
        ckpt_hparams_file = os.path.join(ckpt_dir, "hparams")
        if tf.gfile.Exists(ckpt_hparams_file) or flags.hparams_path:
            hparams = create_or_load_hparams(
                ckpt_dir, default_hparams, flags.hparams_path,
                save_hparams=False)
            loaded_hparams = True
    if not loaded_hparams:  # Try to load from out_dir
        assert out_dir
        hparams = create_or_load_hparams(
            out_dir, default_hparams, flags.hparams_path,
            save_hparams=(jobid == 0))

    ## Train / Decode
    if flags.inference_input_file:
        # Inference output directory
        trans_file = flags.inference_output_file
        assert trans_file
        trans_dir = os.path.dirname(trans_file)
        if not tf.gfile.Exists(trans_dir): tf.gfile.MakeDirs(trans_dir)

        # Inference indices
        hparams.inference_indices = None
        if flags.inference_list:
            (hparams.inference_indices) = (
                [int(token) for token in flags.inference_list.split(",")])

        # Inference
        ckpt = flags.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        inference_fn(ckpt, flags.inference_input_file,
                     trans_file, hparams, num_workers, jobid)

        # Evaluation
        ref_file = flags.inference_ref_file
        if ref_file and tf.gfile.Exists(trans_file):
            for metric in hparams.metrics:
                score = evaluation_utils.evaluate(
                    ref_file,
                    trans_file,
                    metric,
                    hparams.subword_option)
                utils.print_out("  %s: %.1f" % (metric, score))
    else:
        # Train
        train_fn(hparams, target_session=target_session)


def ensure_compatible_hparams(hparams, default_hparams, hparams_path=""):
    """Make sure the loaded hparams is compatible with new changes."""
    default_hparams = utils.maybe_parse_standard_hparams(
        default_hparams, hparams_path)

    # Set num encoder/decoder layers (for old checkpoints)
    if hasattr(hparams, "num_layers"):
        if not hasattr(hparams, "num_encoder_layers"):
            hparams.add_hparam("num_encoder_layers", hparams.num_layers)
        if not hasattr(hparams, "num_decoder_layers"):
            hparams.add_hparam("num_decoder_layers", hparams.num_layers)

    # For compatible reason, if there are new fields in default_hparams,
    #   we add them to the current hparams
    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

    # Update all hparams' keys if override_loaded_hparams=True
    if getattr(default_hparams, "override_loaded_hparams", None):
        overwritten_keys = default_config.keys()
    else:
        # For inference
        overwritten_keys = INFERENCE_KEYS

    for key in overwritten_keys:
        if getattr(hparams, key) != default_config[key]:
            utils.print_out("# Updating hparams.%s: %s -> %s" %
                            (key, str(getattr(hparams, key)),
                             str(default_config[key])))
            setattr(hparams, key, default_config[key])
    return hparams


def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    train_fn = train.train
    inference_fn = inference.inference
    run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
    FLAGS = network_config.NetworkConfig()
    tf.app.run(main=main, argv=[sys.argv[0]])
