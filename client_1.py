import argparse
import warnings

import flwr as fl
from flwr_datasets import FederatedDataset
from tensorflow import keras as keras

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="192.168.1.1:8085",
    help=f"gRPC server address (default '192.168.1.2:8085')",
)
parser.add_argument(
    "--cid", # Container id, this is ideal to ideantify each container and make the dataset partition with IID data
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 2 


def prepare_dataset():
    """Download and partition the MNIST dataset."""
    # MNNIST
    fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS}) # dataset="cifar10"   --> if you want to use dataset cifar10
    img_key = "image" # img  --> if you want to use dataset cifar10
    partitions = []
    
    for partition_id in range(NUM_CLIENTS): # the dataset is partitioned deppending on the number of clients available
        partition = fds.load_partition(partition_id, "train")
        partition.set_format("numpy")
        # Divide data on each node: 80% train, 20% test
        partition = partition.train_test_split(test_size=0.2, seed=42)
        x_train, y_train = (
            partition["train"][img_key] / 255.0,
            partition["train"]["label"],
        )
        x_test, y_test = partition["test"][img_key] / 255.0, partition["test"]["label"]
        partitions.append(((x_train, y_train), (x_test, y_test)))
    
    data_centralized = fds.load_split("test")
    data_centralized.set_format("numpy")
    x_centralized = data_centralized[img_key] / 255.0
    y_centralized = data_centralized["label"]
    
    return partitions, (x_centralized, y_centralized)


class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that uses a smaller CNN for MNIST."""
    
    def __init__(self, trainset, valset):
        self.x_train, self.y_train = trainset
        self.x_val, self.y_val = valset
        # Instantiate model for MNIST
        self.model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),  # (32, 32, 3)   ---> if you want to use dataset CIFAR10
                keras.layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(), # Unidimensional vector
                keras.layers.Dropout(0.5), # turn off 50% of perceptrons to improve the generalization - Randomly
                keras.layers.Dense(10, activation="softmax"),
            ]
        )
        self.model.compile(
            "adam", "sparse_categorical_crossentropy", metrics=["accuracy"]
        )

    def get_parameters(self, config): # Server gets the weights from the trained model.
        return self.model.get_weights()

    def set_parameters(self, params): #  weights are updated in the client sent by the server
        self.model.set_weights(params)

    def fit(self, parameters, config): # Train the model with the client data
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        # Set hyperparameters from config sent by server/strategy
        batch, epochs = config["batch_size"], config["epochs"]
        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch)
        return self.get_parameters({}), len(self.x_train), {}

    def evaluate(self, parameters, config): # Evaluate the model with the validation data of the client
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val)
        return loss, len(self.x_val), {"accuracy": accuracy}


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS # Be aware that there are not more clients that the user requests

    # Always use MNIST dataset
    partitions, _ = prepare_dataset()
    trainset, valset = partitions[args.cid]

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainset, valset=valset
        ).to_client(),
    )


if __name__ == "__main__":
    main()