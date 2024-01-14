import json
from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("output_filename", None, "Output filename.", required=True)

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def main(_):
    config = FLAGS.config
    config.bridgedata_config = FLAGS.bridgedata_config
    with open(f"{FLAGS.output_filename}.json", "w") as f:
        json.dump(config.to_dict(), f)


if __name__ == "__main__":
    app.run(main)