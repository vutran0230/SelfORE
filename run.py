from selfore import SelfORE
import yaml


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    selfore = SelfORE(config)
    selfore.start()
    selfore.save(config['save_path'])


if __name__ == "__main__":
    main()
