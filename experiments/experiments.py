
from core.dataset import PerfumeDatasetGenerator
from core.model import PerfumeTechModel
from core.discovery import DiscoveryEngine


DATASET_SIZE = 6000
EPOCHS = 300
MODEL_PATH = "perfume_model.pkl"


def run_experiments():
    print("\n[1] Generating dataset...")

    generator = PerfumeDatasetGenerator()
    X, y = generator.generate_dataset(DATASET_SIZE)

    print("Dataset:", X.shape, y.shape)

    print("\n[2] Training model...")
    model = PerfumeTechModel(input_size=X.shape[1])
    model.train(X, y, epochs=EPOCHS, lr=0.001)

    print("\n[3] Discovery phase...")
    engine = DiscoveryEngine(model)
    engine.discover(rounds=60)


if __name__ == "__main__":
    run_experiments()
