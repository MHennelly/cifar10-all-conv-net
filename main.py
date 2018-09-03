from format import format
from test_train import test
from test_train import train

def main():
    while True:
        switch = {
            1: test,
            2: train,
            3: format
        }
        choice = int(input("1: Test\n2: Train\n3: Format Batches\n"))
        switch[choice]()

if __name__ == "__main__":
    main()
